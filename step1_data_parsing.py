"""
battery_pipeline.py
===================
Format-agnostic ingestion + parsing for battery cycler data.

Design goals:
  - No format-specific assumptions baked into the parser core.
  - Raw data preserved untouched; all transforms happen on copies.
  - Fail loud on missing critical columns; never silently interpolate.
  - Cycle detection prefers an explicit cycle index, with a rest-aware
    sign-change fallback (NOT naive zero-crossing).

The only thing the parser *requires* you to supply that can't be inferred
from a raw file is cell-level physics (nominal capacity for C-rate). That
is passed via CellConfig, never guessed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("battery_pipeline")


# ---------------------------------------------------------------------------
# 1. Configuration & column aliasing
# ---------------------------------------------------------------------------

# Canonical schema. Everything downstream uses ONLY these names.
CANONICAL_COLUMNS = ("time", "voltage", "current", "capacity", "cycle_index", "step")

# Required to do anything meaningful. Missing -> hard error, no interpolation.
REQUIRED_COLUMNS = ("time", "voltage", "current")

# Alias registry: maps lowercased/stripped source names -> canonical.
# Extend this per cycler vendor instead of writing per-format branches.
# Keys are matched case-insensitively and after stripping units in [].
DEFAULT_ALIASES: dict[str, str] = {
    # time
    "time": "time", "time/s": "time", "test_time": "time", "test_time(s)": "time",
    "test time": "time", "time_s": "time", "timestamp": "time", "step_time": "time",
    "total_time": "time", "elapsed_time": "time", "datetime": "time",
    # voltage
    "voltage": "voltage", "voltage/v": "voltage", "ewe": "voltage", "ewe/v": "voltage",
    "potential": "voltage", "cell_voltage": "voltage", "voltage(v)": "voltage", "v": "voltage",
    # current
    "current": "current", "current/a": "current", "i": "current", "i/ma": "current",
    "current(a)": "current", "cell_current": "current", "current/ma": "current",
    # capacity
    "capacity": "capacity", "capacity/mah": "capacity", "capacity/ah": "capacity",
    "q": "capacity", "charge_capacity": "capacity", "discharge_capacity": "capacity",
    "capacity(ah)": "capacity", "capacity(mah)": "capacity",
    # cycle index
    "cycle_index": "cycle_index", "cycle": "cycle_index", "cycle_number": "cycle_index",
    "cycle number": "cycle_index", "cyc#": "cycle_index", "cycle_no": "cycle_index",
    # step
    "step": "step", "step_index": "step", "step_type": "step", "step_no": "step",
    "ns": "step", "mode": "step",
}


@dataclass
class CellConfig:
    """Cell-level physics the raw file cannot tell you. Never guessed."""
    nominal_capacity_ah: Optional[float] = None  # required for C-rate / ICA filtering
    # Explicit unit declarations override auto-inference. Use when ambiguous.
    capacity_unit: Optional[str] = None   # "Ah" | "mAh"
    current_unit: Optional[str] = None    # "A"  | "mA"
    time_unit: Optional[str] = None       # "s"  | "h" | "min"


@dataclass
class ParseConfig:
    aliases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ALIASES))
    min_points_per_cycle: int = 20        # below this -> flagged
    ica_max_c_rate: float = 0.1           # "low-rate" ceiling for ICA suitability
    fast_charge_c_rate: float = 1.0       # reject as fast-charge above this
    rest_current_threshold_a: float = 1e-4  # |I| below this = rest (in Amps)
    dedupe_time: bool = True
    cache_dir: Path = Path("./_battery_cache")


# ---------------------------------------------------------------------------
# 2. Loading (format adapters)
# ---------------------------------------------------------------------------

def _read_any(path: Path) -> pd.DataFrame:
    """Dispatch on extension only. Add adapters here, not in the core logic."""
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".csv", ".txt"):
        # sep=None + python engine sniffs delimiter (handles tab/comma/semicolon dumps)
        return pd.read_csv(path, sep=None, engine="python")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    raise ValueError(
        f"Unsupported file type {suffix!r}. Add an adapter in _read_any() "
        f"rather than special-casing downstream."
    )


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load a cycler file. Returns the RAW frame, untouched."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = _read_any(path)
    if df.empty:
        raise ValueError(f"{path} loaded as an empty frame.")
    logger.info("Loaded %s: %d rows, %d cols", path.name, len(df), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# 3. Standardize, validate, type-coerce, normalize units
# ---------------------------------------------------------------------------

def _normalize_name(raw: str) -> str:
    """Lowercase, strip whitespace; used as the alias-lookup key."""
    return str(raw).strip().lower()


def _infer_unit_from_name(raw_name: str) -> Optional[str]:
    """Pull a unit hint out of a column name like 'Capacity/mAh' or 'I/mA'."""
    n = _normalize_name(raw_name)
    for token, unit in (("/mah", "mAh"), ("(mah)", "mAh"), ("/ah", "Ah"), ("(ah)", "Ah"),
                        ("/ma", "mA"), ("(ma)", "mA"), ("/a", "A"), ("(a)", "A")):
        if n.endswith(token):
            return unit
    return None


def standardize_columns(raw: pd.DataFrame, cfg: ParseConfig
                       ) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Rename source columns to the canonical schema.
    Returns (renamed_copy, detected_units) where detected_units maps
    canonical name -> inferred source unit string (or None).
    Operates on a COPY; raw is never mutated.
    """
    rename_map: dict[str, str] = {}
    detected_units: dict[str, Optional[str]] = {}
    for col in raw.columns:
        key = _normalize_name(col)
        if key in cfg.aliases:
            canonical = cfg.aliases[key]
            # Don't overwrite an already-mapped canonical col (e.g. two voltage cols)
            if canonical not in rename_map.values():
                rename_map[col] = canonical
                detected_units[canonical] = _infer_unit_from_name(col)
    df = raw.rename(columns=rename_map).copy()
    # Drop any non-canonical leftovers? No -- keep them, they may be useful for debug.
    return df, detected_units


def validate_required(df: pd.DataFrame) -> None:
    """Hard fail if a required column is absent. No interpolation, no guessing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after aliasing: {missing}. "
            f"Available: {list(df.columns)}. Add the source name to ParseConfig.aliases. "
            f"Refusing to interpolate or fabricate these."
        )


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical numeric columns to float. Non-parseable -> NaN, then row dropped."""
    numeric_cols = [c for c in ("time", "voltage", "current", "capacity",
                                "cycle_index", "step") if c in df.columns]
    for c in numeric_cols:
        # 'time' may be a datetime string -> convert to epoch seconds
        if c == "time" and not np.issubdtype(df[c].dtype, np.number):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.5:  # mostly parseable as datetime
                df[c] = (parsed - parsed.min()).dt.total_seconds()
                continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_units(df: pd.DataFrame, detected: dict[str, str],
                    cell: CellConfig) -> pd.DataFrame:
    """
    Bring everything to SI-ish base units: Volts, Amps, Ah, seconds.
    Explicit CellConfig units win; otherwise use the unit parsed from the
    column name. If neither is available AND the unit is ambiguous, we DO NOT
    guess from magnitude -- we leave it and warn loudly.
    """
    # Current: target Amps
    cur_unit = cell.current_unit or detected.get("current")
    if cur_unit == "mA":
        df["current"] = df["current"] / 1000.0
    elif cur_unit is None:
        logger.warning("Current unit unknown; assuming Amps. Set CellConfig.current_unit "
                       "to silence. (mA vs A magnitude-guessing is disabled by design.)")

    # Capacity: target Ah
    if "capacity" in df.columns:
        cap_unit = cell.capacity_unit or detected.get("capacity")
        if cap_unit == "mAh":
            df["capacity"] = df["capacity"] / 1000.0
        elif cap_unit is None:
            logger.warning("Capacity unit unknown; assuming Ah. Set CellConfig.capacity_unit.")

    # Time: target seconds
    if cell.time_unit == "h":
        df["time"] = df["time"] * 3600.0
    elif cell.time_unit == "min":
        df["time"] = df["time"] * 60.0
    return df


# ---------------------------------------------------------------------------
# 4. Clean: sort by time, drop dupes / corrupted rows
# ---------------------------------------------------------------------------

def clean_rows(df: pd.DataFrame, cfg: ParseConfig) -> pd.DataFrame:
    """
    Sort strictly by time, drop rows where required fields are NaN (corrupted),
    and remove duplicate timestamps (keep first).
    """
    before = len(df)
    # Drop rows with NaN in any required column (these are corrupted / unparseable)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    # Stable sort by time
    df = df.sort_values("time", kind="mergesort").reset_index(drop=True)
    if cfg.dedupe_time:
        df = df.drop_duplicates(subset="time", keep="first").reset_index(drop=True)
    logger.info("clean_rows: %d -> %d rows (%d removed)", before, len(df), before - len(df))
    return df


# ---------------------------------------------------------------------------
# 5. Cycle detection
# ---------------------------------------------------------------------------

def detect_cycles(df: pd.DataFrame, cfg: ParseConfig) -> pd.DataFrame:
    """
    Add a 'cycle_index' column if not already present/usable.

    Strategy:
      1. If a usable cycle_index exists (monotonic non-decreasing, >1 unique), trust it.
      2. Otherwise, derive cycles from a REST-AWARE current-sign state machine:
         a full cycle = a charge phase followed by a discharge phase (or vice
         versa). Rest periods (|I| < threshold) do NOT trigger a boundary;
         they inherit the current phase.

    Naive zero-crossing is deliberately avoided -- it shatters cycles at every
    rest step and noise spike.
    """
    if "cycle_index" in df.columns and df["cycle_index"].notna().any():
        ci = df["cycle_index"]
        if ci.is_monotonic_increasing and ci.nunique() > 1:
            logger.info("Using existing cycle_index (%d cycles).", ci.nunique())
            return df
        logger.warning("Existing cycle_index unusable (non-monotonic or single value); "
                       "deriving from current sign.")

    thr = cfg.rest_current_threshold_a
    sign = np.where(df["current"] > thr, 1,
                    np.where(df["current"] < -thr, -1, 0))  # +1 chg, -1 dis, 0 rest

    # Carry the last non-rest sign forward through rest periods
    phase = np.zeros(len(sign), dtype=int)
    last = 0
    for i, s in enumerate(sign):
        if s != 0:
            last = s
        phase[i] = last

    # A new cycle starts on a charge->discharge->charge return.
    # Count a cycle increment each time we transition from discharge(-1) back to charge(+1).
    cycle = np.zeros(len(phase), dtype=int)
    counter = 0
    prev = phase[0] if len(phase) else 0
    for i in range(len(phase)):
        if prev == -1 and phase[i] == 1:
            counter += 1
        cycle[i] = counter
        prev = phase[i]
    df = df.copy()
    df["cycle_index"] = cycle
    df["_phase"] = phase  # +1 charge, -1 discharge, kept for half-cycle split
    logger.info("Derived %d cycles from current sign.", df["cycle_index"].nunique())
    return df


# ---------------------------------------------------------------------------
# 6. Cycle object construction (+ half-cycle split, metadata, flags)
# ---------------------------------------------------------------------------

def _half_cycle_phase(sub: pd.DataFrame, cfg: ParseConfig) -> np.ndarray:
    """Return +1/-1/0 phase array for a sub-frame (used if _phase absent)."""
    if "_phase" in sub.columns:
        return sub["_phase"].to_numpy()
    thr = cfg.rest_current_threshold_a
    return np.where(sub["current"] > thr, 1, np.where(sub["current"] < -thr, -1, 0))


def _c_rate(sub: pd.DataFrame, cell: CellConfig) -> Optional[float]:
    """Mean |current| / nominal capacity. None if nominal capacity unknown."""
    if cell.nominal_capacity_ah is None or cell.nominal_capacity_ah <= 0:
        return None
    active = sub.loc[sub["current"].abs() > 0, "current"].abs()
    if active.empty:
        return 0.0
    return float(active.mean() / cell.nominal_capacity_ah)


def build_cycle_object(cycle_id: int, sub: pd.DataFrame,
                       cfg: ParseConfig, cell: CellConfig) -> dict[str, Any]:
    """
    Build one independent cycle dict with charge/discharge halves + metadata + flags.
    'sub' is the slice of the cleaned frame for this cycle.
    """
    sub = sub.reset_index(drop=True)
    phase = _half_cycle_phase(sub, cfg)
    charge = sub[phase == 1].reset_index(drop=True)
    discharge = sub[phase == -1].reset_index(drop=True)

    c_rate = _c_rate(sub, cell)
    n = len(sub)

    duration_s = float(sub["time"].iloc[-1] - sub["time"].iloc[0]) if n > 1 else 0.0
    v_min = float(sub["voltage"].min())
    v_max = float(sub["voltage"].max())

    # Capacity per half cycle: prefer reported capacity column, else integrate I dt (coulomb counting)
    def _capacity_ah(half: pd.DataFrame) -> Optional[float]:
        if half.empty:
            return 0.0
        if "capacity" in half.columns and half["capacity"].notna().any():
            return float(half["capacity"].abs().max())
        # Coulomb count: integral of |I| dt, seconds -> hours
        t = half["time"].to_numpy()
        i = half["current"].abs().to_numpy()
        if len(t) < 2:
            return 0.0
        return float(np.trapz(i, t) / 3600.0)

    flags: list[str] = []
    if n < cfg.min_points_per_cycle:
        flags.append(f"too_few_points({n}<{cfg.min_points_per_cycle})")
    if c_rate is None:
        flags.append("c_rate_unknown_no_nominal_capacity")

    # ICA suitability + fast-charge rejection (require known C-rate)
    ica_suitable = False
    fast_charge = False
    if c_rate is not None:
        ica_suitable = c_rate <= cfg.ica_max_c_rate
        fast_charge = c_rate >= cfg.fast_charge_c_rate
    if fast_charge:
        flags.append(f"fast_charge_rejected(C={c_rate:.2f})")

    return {
        "cycle_index": int(cycle_id),
        "data": sub,                     # full cleaned cycle frame
        "charge": charge,                # charge half-cycle
        "discharge": discharge,          # discharge half-cycle
        "meta": {
            "n_points": n,
            "duration_s": duration_s,
            "voltage_min": v_min,
            "voltage_max": v_max,
            "voltage_range": v_max - v_min,
            "charge_capacity_ah": _capacity_ah(charge),
            "discharge_capacity_ah": _capacity_ah(discharge),
            "c_rate_est": c_rate,
            "ica_suitable": ica_suitable,
            "fast_charge": fast_charge,
        },
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# 7. Orchestration + caching
# ---------------------------------------------------------------------------

@dataclass
class ParsedDataset:
    raw: pd.DataFrame                       # untouched, for debugging
    clean: pd.DataFrame                     # standardized + cleaned
    cycles: dict[int, dict[str, Any]]       # cycle_index -> cycle object
    source: str

    def ica_cycles(self) -> dict[int, dict[str, Any]]:
        return {k: v for k, v in self.cycles.items() if v["meta"]["ica_suitable"]}

    def flagged(self) -> dict[int, list[str]]:
        return {k: v["flags"] for k, v in self.cycles.items() if v["flags"]}


def parse_file(path: str | Path, cell: CellConfig,
               cfg: Optional[ParseConfig] = None,
               use_cache: bool = True) -> ParsedDataset:
    """Full pipeline: load -> standardize -> validate -> clean -> cycle -> objectify -> cache."""
    cfg = cfg or ParseConfig()
    path = Path(path)

    cache_file = cfg.cache_dir / f"{path.stem}.parsed.pkl"
    if use_cache and cache_file.exists():
        logger.info("Loading parsed cache: %s", cache_file)
        return pd.read_pickle(cache_file)

    raw = load_raw(path)
    df, detected = standardize_columns(raw, cfg)
    validate_required(df)
    df = coerce_numeric(df)
    df = normalize_units(df, detected, cell)
    df = clean_rows(df, cfg)
    df = detect_cycles(df, cfg)

    cycles: dict[int, dict[str, Any]] = {}
    for cid, sub in df.groupby("cycle_index"):
        cycles[int(cid)] = build_cycle_object(cid, sub, cfg, cell)

    result = ParsedDataset(raw=raw, clean=df, cycles=cycles, source=str(path))

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(result, cache_file)
    logger.info("Cached parsed dataset -> %s (%d cycles)", cache_file, len(cycles))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import sys
    if len(sys.argv) < 2:
        print("Usage: python battery_pipeline.py <cycler_file> [nominal_capacity_ah]")
        sys.exit(1)
    nominal = float(sys.argv[2]) if len(sys.argv) > 2 else None
    ds = parse_file(sys.argv[1], CellConfig(nominal_capacity_ah=nominal))
    print(f"\nParsed {len(ds.cycles)} cycles from {ds.source}")
    print(f"ICA-suitable cycles: {sorted(ds.ica_cycles())}")
    print(f"Flagged cycles: {ds.flagged()}")