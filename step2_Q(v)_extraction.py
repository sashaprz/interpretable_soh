"""
ica.py
======
Incremental Capacity Analysis (dQ/dV) on parsed cycle objects.

Consumes the cycle dicts produced by battery_pipeline.parse_file(). Keeps the
parser dumb: all ICA-specific logic lives here.

Pipeline per half-cycle:
    extract CC segment  ->  build Q(V)  ->  enforce monotonic V
    ->  interpolate onto uniform V grid  ->  smooth  ->  differentiate
    ->  trim edges  ->  detect peaks  ->  stability + distortion validation

Deviations from the naive spec (deliberate):
  * Default differentiation is Savitzky-Golay deriv=1, NOT smooth-then-gradient.
    savgol(deriv=1) takes the derivative analytically from the same local
    polynomial used for smoothing -> "smooth before differentiate" done
    consistently, without re-injecting noise via a separate finite difference.
  * Default interpolation is PCHIP (monotone cubic), NOT cubic spline.
    Cubic splines overshoot on steep Q(V) regions and manufacture phantom
    dQ/dV peaks. PCHIP cannot overshoot.
  * CC-segment extraction and monotonic-V enforcement are added because the
    spec is unusable without them.

Provenance: every ICACurve carries the SHA-256 hash of the ICAConfig used to
make it. assert_comparable() refuses to let you compare curves made with
different configs -- enforcing "do not compare differently-processed curves".
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter, find_peaks

logger = logging.getLogger("ica")

ICA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Config + provenance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ICAConfig:
    # --- segment selection ---
    max_c_rate_research: float = 0.10      # <= C/10 = research grade
    max_c_rate_fallback: float = 0.20      # accept up to C/5 but flag non-research-grade
    cc_current_tol: float = 0.05           # |I - I_cc| / I_cc allowed within "constant" current
    min_voltage_span_v: float = 0.05       # half-cycle must cover >= 50 mV to be usable

    # --- grid ---
    dv_mv: float = 2.0                      # uniform voltage spacing (1-5 mV typical)

    # --- smoothing (stored explicitly for reproducibility) ---
    interp: Literal["pchip", "linear"] = "pchip"
    smooth_method: Literal["savgol_deriv", "smooth_then_gradient"] = "savgol_deriv"
    savgol_window_mv: float = 20.0          # smoothing window in mV (converted to odd #pts)
    savgol_polyorder: int = 3

    # --- differentiation / edges ---
    edge_trim_windows: float = 1.0          # trim this many smoothing-windows from each end

    # --- validation thresholds ---
    peak_prominence_frac: float = 0.05      # peak prominence as fraction of max |dQ/dV|
    peak_stability_tol_mv: float = 10.0     # peak may not move more than this under window change
    max_residual_frac: float = 0.02         # smoothed-vs-raw Q RMSE / capacity ceiling

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def provenance(self, source: str, cycle_index: int, half: str) -> dict[str, Any]:
        return {
            "ica_version": ICA_VERSION,
            "config_hash": self.hash(),
            "config": asdict(self),
            "source": source,
            "cycle_index": cycle_index,
            "half": half,
        }


@dataclass
class ICACurve:
    cycle_index: int
    half: str                       # "charge" | "discharge"
    voltage_grid: np.ndarray        # uniform V grid
    q_raw_on_grid: np.ndarray       # raw Q(V) interpolated onto grid (pre-smoothing)
    q_smoothed: np.ndarray          # smoothed Q(V)
    dqdv: np.ndarray                # dQ/dV on grid (oriented positive-up; see direction)
    peaks_v: np.ndarray             # detected peak voltages
    c_rate: Optional[float]
    research_grade: bool
    flags: list[str]
    provenance: dict[str, Any]
    diagnostics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Segment selection
# ---------------------------------------------------------------------------

def _pick_cycle_for_ica(cycles: dict[int, dict], cfg: ICAConfig
                       ) -> tuple[Optional[int], bool]:
    """Prefer <= C/10; fall back to <= C/5 (flagged). Return (cycle_index, research_grade)."""
    rated = [(cid, c["meta"]["c_rate_est"]) for cid, c in cycles.items()
             if c["meta"]["c_rate_est"] is not None]
    if not rated:
        return None, False
    research = [(cid, cr) for cid, cr in rated if cr <= cfg.max_c_rate_research]
    if research:
        cid = min(research, key=lambda x: x[1])[0]
        return cid, True
    fallback = [(cid, cr) for cid, cr in rated if cr <= cfg.max_c_rate_fallback]
    if fallback:
        cid = min(fallback, key=lambda x: x[1])[0]
        logger.warning("No <=C/10 cycle; using C-rate %.3f (NOT research grade).",
                       dict(fallback)[cid])
        return cid, False
    return None, False


def _extract_cc_segment(half: pd.DataFrame, cfg: ICAConfig) -> Optional[pd.DataFrame]:
    """
    Keep only the constant-current portion (drop CV tail / current ramps).
    CC = current within cc_current_tol of the segment's median |current|.
    """
    if half.empty:
        return None
    i_abs = half["current"].abs()
    i_cc = float(i_abs.median())
    if i_cc <= 0:
        return None
    mask = (i_abs - i_cc).abs() / i_cc <= cfg.cc_current_tol
    seg = half[mask]
    if len(seg) < 5:
        return None
    return seg.reset_index(drop=True)


def _build_qv(seg: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Q(V) for a CC segment. Q = coulomb count of |I| dt (Ah), starting at 0.
    Returns (V, Q) time-ordered (V monotonic within CC, modulo noise).
    """
    t = seg["time"].to_numpy(float)
    i = seg["current"].abs().to_numpy(float)
    v = seg["voltage"].to_numpy(float)
    # cumulative trapezoid in Ah (time is seconds)
    dq = np.concatenate([[0.0], (i[1:] + i[:-1]) / 2.0 * np.diff(t) / 3600.0])
    q = np.cumsum(dq)
    return v, q


def _enforce_monotonic_v(v: np.ndarray, q: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Force V strictly monotonic so Q(V) is invertible. Determine direction from
    net trend; keep only points that advance V in that direction (drops noise
    reversals). Returns (V_inc, Q_inc, direction) with V ascending.
    """
    direction = "up" if v[-1] >= v[0] else "down"
    if direction == "down":
        v = v[::-1].copy()
        q = (q[-1] - q)[::-1].copy()   # re-reference Q so it increases with index
    keep = [0]
    last = v[0]
    for k in range(1, len(v)):
        if v[k] > last + 1e-9:
            keep.append(k)
            last = v[k]
    keep = np.array(keep)
    return v[keep], q[keep], direction


# ---------------------------------------------------------------------------
# Core ICA on a single half-cycle
# ---------------------------------------------------------------------------

def _odd_window(span_mv: float, dv_mv: float, n: int, polyorder: int) -> int:
    """Convert a mV smoothing window into a valid odd savgol window length."""
    w = max(int(round(span_mv / dv_mv)), polyorder + 2)
    if w % 2 == 0:
        w += 1
    # must be <= n and odd
    w = min(w, n if n % 2 == 1 else n - 1)
    if w <= polyorder:
        w = polyorder + 1 + (polyorder % 2)
    return w


def _differentiate(q_grid: np.ndarray, dv_v: float, cfg: ICAConfig, window: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_smoothed, dqdv). Default = savgol deriv=1; alt = smooth-then-gradient."""
    if cfg.smooth_method == "savgol_deriv":
        q_sm = savgol_filter(q_grid, window, cfg.savgol_polyorder)
        dqdv = savgol_filter(q_grid, window, cfg.savgol_polyorder, deriv=1, delta=dv_v)
    else:  # smooth_then_gradient (literal spec reading; noisier)
        q_sm = savgol_filter(q_grid, window, cfg.savgol_polyorder)
        dqdv = np.gradient(q_sm, dv_v)
    return q_sm, dqdv


def compute_ica_for_half(cycle: dict, half: str, cfg: ICAConfig,
                         source: str, research_grade: bool) -> Optional[ICACurve]:
    flags: list[str] = []
    seg = _extract_cc_segment(cycle[half], cfg)
    if seg is None:
        logger.info("cycle %s %s: no usable CC segment.", cycle["cycle_index"], half)
        return None

    v_raw, q_raw = _build_qv(seg)
    v_mono, q_mono, direction = _enforce_monotonic_v(v_raw, q_raw)

    span = v_mono[-1] - v_mono[0]
    if span < cfg.min_voltage_span_v or len(v_mono) < 8:
        logger.info("cycle %s %s: insufficient V span (%.3f V) / points (%d).",
                    cycle["cycle_index"], half, span, len(v_mono))
        return None

    # Uniform voltage grid
    dv_v = cfg.dv_mv / 1000.0
    grid = np.arange(v_mono[0], v_mono[-1] + dv_v, dv_v)

    # Interpolate Q onto grid (PCHIP = no overshoot; linear optional)
    if cfg.interp == "pchip":
        q_on_grid = PchipInterpolator(v_mono, q_mono)(grid)
    else:
        q_on_grid = np.interp(grid, v_mono, q_mono)

    n = len(grid)
    window = _odd_window(cfg.savgol_window_mv, cfg.dv_mv, n, cfg.savgol_polyorder)
    if window <= cfg.savgol_polyorder:
        flags.append("grid_too_short_for_smoothing")
        return None

    q_sm, dqdv = _differentiate(q_on_grid, dv_v, cfg, window)

    # Remove edge artifacts: trim edge_trim_windows * (window/2) points each side
    trim = int(cfg.edge_trim_windows * window // 2)
    if trim > 0 and n > 2 * trim + 5:
        sl = slice(trim, n - trim)
        grid, q_on_grid, q_sm, dqdv = grid[sl], q_on_grid[sl], q_sm[sl], dqdv[sl]

    # dQ/dV is oriented positive-up for both halves (discharge was sign-flipped
    # in _enforce_monotonic_v). Find peaks pointing up.
    prom = cfg.peak_prominence_frac * np.nanmax(np.abs(dqdv))
    pk_idx, _ = find_peaks(dqdv, prominence=max(prom, 1e-12))
    peaks_v = grid[pk_idx]

    # --- distortion validation: smoothed vs raw Q residual ---
    resid = float(np.sqrt(np.mean((q_sm - q_on_grid) ** 2)))
    cap = max(abs(q_on_grid[-1] - q_on_grid[0]), 1e-9)
    resid_frac = resid / cap
    if resid_frac > cfg.max_residual_frac:
        flags.append(f"smoothing_distortion(resid={resid_frac:.3f}>{cfg.max_residual_frac})")

    # --- peak stability: re-run with a wider window, peaks must not move much ---
    w2 = _odd_window(cfg.savgol_window_mv * 1.5, cfg.dv_mv, len(q_on_grid), cfg.savgol_polyorder)
    max_shift_mv = float("nan")
    if w2 != window and w2 > cfg.savgol_polyorder and w2 < len(q_on_grid):
        _, dqdv2 = _differentiate(q_on_grid, dv_v, cfg, w2)
        pk2, _ = find_peaks(dqdv2, prominence=max(prom, 1e-12))
        peaks_v2 = grid[pk2] if len(pk2) else np.array([])
        max_shift_mv = 0.0
        for p in peaks_v:
            if len(peaks_v2):
                max_shift_mv = max(max_shift_mv, float(np.min(np.abs(peaks_v2 - p)) * 1000.0))
        if max_shift_mv > cfg.peak_stability_tol_mv:
            flags.append(f"peak_unstable({max_shift_mv:.1f}mV>{cfg.peak_stability_tol_mv})")

    if not research_grade:
        flags.append("not_research_grade_c_rate")

    return ICACurve(
        cycle_index=int(cycle["cycle_index"]),
        half=half,
        voltage_grid=grid,
        q_raw_on_grid=q_on_grid,
        q_smoothed=q_sm,
        dqdv=dqdv,
        peaks_v=peaks_v,
        c_rate=cycle["meta"]["c_rate_est"],
        research_grade=research_grade,
        flags=flags,
        provenance=cfg.provenance(source, int(cycle["cycle_index"]), half),
        diagnostics={
            "n_grid": float(len(grid)),
            "window_pts": float(window),
            "voltage_span_v": float(span),
            "residual_frac": float(resid_frac),
            "peak_shift_mv": float(max_shift_mv),
            "direction": 1.0 if direction == "up" else -1.0,
        },
    )


# ---------------------------------------------------------------------------
# Dataset-level driver + comparability guard + caching
# ---------------------------------------------------------------------------

def assert_comparable(curves: list[ICACurve]) -> None:
    """Refuse to let differently-processed curves be compared."""
    hashes = {c.provenance["config_hash"] for c in curves}
    if len(hashes) > 1:
        raise ValueError(
            f"Curves were processed with DIFFERENT configs {hashes}. "
            f"Comparing them is invalid. Reprocess all with one ICAConfig."
        )


def run_ica(parsed, cfg: Optional[ICAConfig] = None,
            halves: tuple[str, ...] = ("charge", "discharge"),
            cycle_index: Optional[int] = None,
            cache_dir: Path = Path("./_ica_cache"),
            use_cache: bool = True) -> list[ICACurve]:
    """
    Run ICA over a ParsedDataset. If cycle_index is None, auto-pick the lowest
    C-rate cycle (<=C/10 preferred). Caches per (source, cycle, half, config_hash).
    """
    cfg = cfg or ICAConfig()
    cycles = parsed.cycles
    source = parsed.source

    if cycle_index is None:
        cycle_index, research = _pick_cycle_for_ica(cycles, cfg)
        if cycle_index is None:
            raise ValueError("No cycle has a known C-rate (set CellConfig.nominal_capacity_ah). "
                             "Cannot select a low-rate segment for ICA.")
    else:
        cr = cycles[cycle_index]["meta"]["c_rate_est"]
        research = cr is not None and cr <= cfg.max_c_rate_research

    out: list[ICACurve] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source).stem
    chash = cfg.hash()

    for half in halves:
        cache_file = cache_dir / f"{stem}_cyc{cycle_index}_{half}_{chash}.pkl"
        if use_cache and cache_file.exists():
            out.append(pd.read_pickle(cache_file))
            continue
        curve = compute_ica_for_half(cycles[cycle_index], half, cfg, source, research)
        if curve is not None:
            pd.to_pickle(curve, cache_file)
            out.append(curve)
    return out


def plot_ica(curve: ICACurve, save_path: Optional[str] = None):
    """Visual raw-vs-smoothed check. Top: Q(V) raw points + smoothed. Bottom: dQ/dV + peaks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(curve.voltage_grid, curve.q_raw_on_grid, ".", ms=2, alpha=0.4,
             label="raw Q(V) on grid")
    ax1.plot(curve.voltage_grid, curve.q_smoothed, "-", lw=1.5, label="smoothed Q(V)")
    ax1.set_ylabel("Q (Ah)")
    ax1.legend(fontsize=8)
    ax1.set_title(f"cyc {curve.cycle_index} {curve.half} | C-rate "
                  f"{curve.c_rate:.3f} | cfg {curve.provenance['config_hash']} | "
                  f"{'RESEARCH' if curve.research_grade else 'fallback'}", fontsize=9)

    ax2.plot(curve.voltage_grid, curve.dqdv, "-", lw=1.2)
    for pv in curve.peaks_v:
        ax2.axvline(pv, color="r", ls="--", lw=0.8, alpha=0.7)
    ax2.set_xlabel("Voltage (V)")
    ax2.set_ylabel("dQ/dV (Ah/V)")
    if curve.flags:
        ax2.text(0.02, 0.95, "FLAGS: " + ", ".join(curve.flags), transform=ax2.transAxes,
                 fontsize=7, va="top", color="darkred")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    return save_path