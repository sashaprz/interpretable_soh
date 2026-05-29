"""
step3_deltaQ(V)_feature_extraction.py
Leakage-free SOH feature engineering from delta Q(V) curves.
"""
from __future__ import annotations
import hashlib, json, logging, warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from feature_schema import (
    FEATURE_COLS as _SCHEMA_FEATURE_COLS,
    save_schema_json as _save_schema_json,
)

logger = logging.getLogger("battery.features")

_SOH_PROXY_NAMES = frozenset({
    "discharge_capacity_ah", "charge_capacity_ah",
    "coulombic_efficiency", "energy_wh",
    "soh", "capacity_fade", "capacity_retention",
})


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Single source of truth for all feature-engineering parameters.

    config_hash() ties every artifact to an exact config. Objects built with
    different hashes must never be compared or combined.
    Treat as immutable after construction.
    """
    # Shared voltage grid -- MUST be identical for every cell in an experiment.
    v_min: float = 2.5
    v_max: float = 4.2
    dv: float = 0.005           # V (5 mV default; 1-5 mV acceptable)

    # Savitzky-Golay smoothing + differentiation
    sg_window: int = 21         # points on the uniform v_grid; must be odd
    sg_polyorder: int = 3

    # Reference cycle: use the Nth ICA-suitable cycle (1-indexed, per cell).
    reference_cycle_rank: int = 1

    # CC region acceptance
    c_rate_max: float = 0.11        # |I| / C_nom <= this (approx C/10 + 10% margin)
    current_cv_max: float = 0.02    # max coeff. of variation of |I| in rolling window
    min_cc_fraction: float = 0.20   # reject if < 20% of half-cycle points pass CC gates

    # Half-cycle to build Q(V) from
    half_cycle: str = "discharge"   # "charge" | "discharge"

    # Outlier clipping applied to delta Q(V) before feature extraction
    outlier_sigma: float = 5.0

    # Flag cycle unusable if > this fraction of grid points are NaN
    max_nan_fraction: float = 0.10

    # Remove first/last N grid points after SG filter (eliminates ringing)
    edge_trim_pts: int = 5

    # Chemistry tag -- gates chemistry-specific feature subsets
    chemistry: str = "generic"

    cache_dir: Path = field(default_factory=lambda: Path("./_feature_cache"))

    def __post_init__(self) -> None:
        if self.sg_window % 2 == 0:
            raise ValueError(f"sg_window must be odd, got {self.sg_window}")
        if self.sg_polyorder >= self.sg_window:
            raise ValueError(f"sg_polyorder must be < sg_window")
        if self.half_cycle not in ("charge", "discharge"):
            raise ValueError(f"half_cycle must be 'charge' or 'discharge'")

    @property
    def v_grid(self) -> np.ndarray:
        return np.arange(self.v_min, self.v_max + self.dv / 2, self.dv)

    def config_hash(self) -> str:
        """SHA-256 of all semantic fields (cache_dir excluded)."""
        d = {k: str(v) for k, v in asdict(self).items() if k != "cache_dir"}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 2. Q(V) extraction from raw half-cycle DataFrames
# ---------------------------------------------------------------------------

class QVExtractor:
    """Build a smooth Q(V) curve from a raw half-cycle DataFrame.

    Required columns: time (s), voltage (V), current (A).
    nominal_capacity_ah is optional but enables C-rate gating.

    Pipeline:
      1. Select CC region (reject CV tails, pulses, unstable-current windows).
      2. Integrate |I| dt -> cumulative Q in Ah (trapezoid rule).
      3. Enforce monotonic voltage; remove duplicate/noisy reversals.
      4. PCHIP-interpolate Q(V) onto the shared v_grid (no overshoot).
      5. Savitzky-Golay smooth Q(V); dQ/dV via single SG pass (deriv=1).
      6. Trim edge artifacts.

    Never differentiates raw unevenly-sampled voltage data directly.
    """

    def __init__(self, cfg: FeatureConfig,
                 nominal_capacity_ah: Optional[float] = None) -> None:
        self.cfg = cfg
        self.nominal_capacity_ah = nominal_capacity_ah

    def _select_cc_region(self, half: pd.DataFrame
                          ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Return CC rows and a diagnostic dict.

        Gate 1: C-rate <= cfg.c_rate_max (prefer <= C/10).
        Gate 2: |dI/dt| <= 5x median |dI/dt| (rejects CV tails and pulses).
        Gate 3: rolling CV of |I| <= cfg.current_cv_max (rejects unstable regions).
        """
        i_abs = half["current"].abs()
        info: Dict[str, Any] = {}

        # Gate 1 -- C-rate
        if self.nominal_capacity_ah and self.nominal_capacity_ah > 0:
            rate_mask = (i_abs / self.nominal_capacity_ah) <= self.cfg.c_rate_max
        else:
            rate_mask = pd.Series(True, index=half.index)
            logger.warning("nominal_capacity_ah not set; C-rate gate skipped.")

        # Gate 2 -- pulse / CV tail detection via |dI/dt| spikes
        dt = half["time"].diff().clip(lower=1e-6)
        di_dt = i_abs.diff().abs() / dt
        med = di_dt.median()
        stable_mask = di_dt <= 5.0 * max(med, 1e-9)
        stable_mask.iloc[0] = True

        # Gate 3 -- rolling coefficient of variation of |I|
        win = max(5, len(half) // 20)
        r_mean = i_abs.rolling(win, center=True, min_periods=1).mean()
        r_std  = i_abs.rolling(win, center=True, min_periods=1).std().fillna(0.0)
        cv_mask = (r_std / r_mean.clip(lower=1e-12)) <= self.cfg.current_cv_max

        cc_mask = rate_mask & stable_mask & cv_mask
        cc_frac = float(cc_mask.mean())
        info["cc_fraction"]  = cc_frac
        info["n_cc_pts"]     = int(cc_mask.sum())
        info["cc_sufficient"] = cc_frac >= self.cfg.min_cc_fraction

        if not info["cc_sufficient"]:
            logger.warning("CC fraction %.1f%% < min %.0f%%.",
                           cc_frac * 100, self.cfg.min_cc_fraction * 100)
        return half.loc[cc_mask].reset_index(drop=True), info

    def extract(self, half: pd.DataFrame
                ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """Returns (q_smooth, dqdv, info) on cfg.v_grid, or (None, None, info) on failure."""
        info: Dict[str, Any] = {"ok": False}

        if len(half) < 10:
            info["fail_reason"] = "too_few_raw_points"
            return None, None, info

        cc, cc_info = self._select_cc_region(half)
        info.update(cc_info)
        if not cc_info.get("cc_sufficient", False) or len(cc) < 10:
            info["fail_reason"] = "insufficient_cc_region"
            return None, None, info

        t     = cc["time"].to_numpy(dtype=float)
        i_abs = cc["current"].abs().to_numpy(dtype=float)
        v     = cc["voltage"].to_numpy(dtype=float)

        # Cumulative charge Q(t) = integral |I| dt  (seconds -> Ah)
        dt = np.diff(t, prepend=t[0])
        dt[0] = dt[1] if len(dt) > 1 else 0.0
        q_cum = np.cumsum(i_abs * dt) / 3600.0

        # Monotonic voltage filter (allow tiny noise reversals within tol)
        tol = 1e-4
        if self.cfg.half_cycle == "discharge":
            mono_mask = np.concatenate([[True], v[1:] <= v[:-1] + tol])
        else:
            mono_mask = np.concatenate([[True], v[1:] >= v[:-1] - tol])

        # Remove duplicate voltages at 0.1 mV resolution
        _, uniq_idx = np.unique(np.round(v, 4), return_index=True)
        uniq_mask = np.zeros(len(v), dtype=bool)
        uniq_mask[uniq_idx] = True

        v_mono = v[mono_mask & uniq_mask]
        q_mono = q_cum[mono_mask & uniq_mask]

        if len(v_mono) < 4:
            info["fail_reason"] = "too_few_monotonic_points"
            return None, None, info

        v_grid = self.cfg.v_grid
        coverage = (v_mono.max() - v_mono.min()) / (v_grid.max() - v_grid.min())
        info["v_coverage"] = float(coverage)
        if coverage < 0.5:
            logger.warning("Voltage coverage only %.0f%% of grid.", coverage * 100)

        # PCHIP onto shared grid (sort ascending; no overshoot; no extrapolation)
        sort_idx = np.argsort(v_mono)
        interp   = PchipInterpolator(v_mono[sort_idx], q_mono[sort_idx], extrapolate=False)
        q_on_grid = interp(v_grid)

        nan_frac = float(np.isnan(q_on_grid).mean())
        info["nan_fraction"] = nan_frac
        if nan_frac > self.cfg.max_nan_fraction:
            info["fail_reason"] = f"too_many_nans({nan_frac:.1%})"
            return None, None, info

        q_filled = _inpaint_nans(q_on_grid)

        # SG smooth Q(V)
        q_smooth = savgol_filter(q_filled, self.cfg.sg_window, self.cfg.sg_polyorder)

        # dQ/dV: single SG pass deriv=1 -- smoothing + differentiation in one step.
        # delta=dv gives correct units (Ah/V). Never diff raw unevenly-sampled data.
        dqdv = savgol_filter(q_filled, self.cfg.sg_window, self.cfg.sg_polyorder,
                             deriv=1, delta=self.cfg.dv)

        # Edge trim -- SG ringing at voltage extremes
        trim = self.cfg.edge_trim_pts
        if trim > 0:
            q_smooth[:trim] = q_smooth[-trim:] = np.nan
            dqdv[:trim]     = dqdv[-trim:]     = np.nan

        # Restore NaN positions outside measured range
        orig_nan = np.isnan(q_on_grid)
        q_smooth[orig_nan] = np.nan
        dqdv[orig_nan]     = np.nan

        info["ok"] = True
        return q_smooth, dqdv, info


def _inpaint_nans(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation across interior NaNs; leading/trailing NaNs -> 0."""
    out  = arr.copy()
    nans = np.isnan(out)
    if not nans.any():
        return out
    x, valid = np.arange(len(out)), ~nans
    if valid.sum() < 2:
        out[:] = 0.0
        return out
    out[nans] = np.interp(x[nans], x[valid], out[valid])
    return out


# ---------------------------------------------------------------------------
# 3. Delta Q(V) computation
# ---------------------------------------------------------------------------

class DeltaQVComputer:
    """Compute delta Q(V) = Q_cycle(V) - Q_ref(V) for a single cell.

    Reference is fixed at fit_reference() time and never updated.
    All Q(V) curves must be on the same v_grid as the config.
    """

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        self._reference: Optional[np.ndarray] = None
        self._ref_cycle_index: Optional[int] = None

    def fit_reference(self, q_curves: Dict[int, Optional[np.ndarray]]) -> int:
        """Select the Nth valid Q(V) curve as reference (cfg.reference_cycle_rank)."""
        valid = {k: v for k, v in sorted(q_curves.items())
                 if v is not None and np.isfinite(v).any()}
        rank = self.cfg.reference_cycle_rank
        if len(valid) < rank:
            raise ValueError(f"Only {len(valid)} valid Q(V) curve(s); "
                             f"cannot select rank-{rank} reference.")
        ref_idx = list(valid.keys())[rank - 1]
        self._reference      = valid[ref_idx].copy()
        self._ref_cycle_index = int(ref_idx)
        logger.info("Reference fixed at cycle %d (rank %d of %d valid).",
                    ref_idx, rank, len(valid))
        return self._ref_cycle_index

    def set_reference(self, q_ref: np.ndarray, ref_cycle_index: int = -1) -> None:
        """Set an externally supplied Q(V) reference curve.

        Use instead of fit_reference() when the reference comes from outside
        the cell's own cycle data -- e.g. a cross-cell baseline, a nominal
        curve from literature, or a pre-agreed early-life snapshot.

        Parameters
        ----------
        q_ref           : reference Q(V) array on the same v_grid as the config
        ref_cycle_index : stored in metadata; use -1 to signal external origin
        """
        if q_ref.shape != (len(self.cfg.v_grid),):
            raise ValueError(
                f"q_ref shape {q_ref.shape} does not match v_grid length "
                f"{len(self.cfg.v_grid)}. Both must be on the same voltage grid."
            )
        self._reference       = q_ref.copy()
        self._ref_cycle_index = int(ref_cycle_index)
        logger.info("Reference set externally (ref_cycle_index=%d, shape=%s).",
                    ref_cycle_index, q_ref.shape)

    @property
    def reference_cycle_index(self) -> Optional[int]:
        return self._ref_cycle_index

    def compute(self, q_cycle: np.ndarray) -> np.ndarray:
        """delta Q(V) = Q_cycle - Q_ref. NaN wherever either curve is NaN."""
        if self._reference is None:
            raise RuntimeError("Call fit_reference() before compute().")
        if q_cycle.shape != self._reference.shape:
            raise ValueError(f"Shape mismatch: {q_cycle.shape} vs {self._reference.shape}. "
                             "Both must be on the same v_grid.")
        return q_cycle - self._reference


# ---------------------------------------------------------------------------
# 4. Feature extraction from delta Q(V)
# ---------------------------------------------------------------------------

# Core feature functions: (valid_array, dv) -> float
# 'valid' is already clipped and finite -- no NaN handling needed inside.

def _feat_variance(x, dv):      return float(np.var(x))
def _feat_log_variance(x, dv):  return float(np.log(np.var(x) + 1e-30))
def _feat_skewness(x, dv):      return float(scipy_skew(x))
def _feat_kurtosis(x, dv):      return float(scipy_kurtosis(x, fisher=True))
def _feat_integral_abs(x, dv):  return float(np.trapezoid(np.abs(x)) * dv)
def _feat_max_deviation(x, dv): return float(np.max(np.abs(x)))
def _feat_min(x, dv):           return float(np.min(x))
def _feat_max(x, dv):           return float(np.max(x))
def _feat_mean(x, dv):          return float(np.mean(x))
def _feat_rms(x, dv):           return float(np.sqrt(np.mean(x**2)))

_CORE_FEATURES: Dict[str, Any] = {
    "dqv_variance":      _feat_variance,
    "dqv_log_variance":  _feat_log_variance,
    "dqv_skewness":      _feat_skewness,
    "dqv_kurtosis":      _feat_kurtosis,
    "dqv_integral_abs":  _feat_integral_abs,
    "dqv_max_deviation": _feat_max_deviation,
    "dqv_min":           _feat_min,
    "dqv_max":           _feat_max,
    "dqv_mean":          _feat_mean,
    "dqv_rms":           _feat_rms,
}

# Guard: _CORE_FEATURES keys must exactly match feature_schema.FEATURE_COLS.
# This fires at import time so drift between the two files is caught immediately.
assert list(_CORE_FEATURES.keys()) == _SCHEMA_FEATURE_COLS, (
    f"_CORE_FEATURES keys don't match feature_schema.FEATURE_COLS\n"
    f"  step3   : {list(_CORE_FEATURES.keys())}\n"
    f"  schema  : {_SCHEMA_FEATURE_COLS}"
)

# Chemistry-specific additions -- extend per chemistry as needed.
_CHEMISTRY_FEATURES: Dict[str, Dict[str, Any]] = {
    "lfp": {
        "dqv_range": lambda x, dv: float(np.max(x) - np.min(x)),
    },
    "nmc": {
        "dqv_range":             lambda x, dv: float(np.max(x) - np.min(x)),
        "dqv_positive_integral": lambda x, dv: float(np.trapz(np.maximum(x, 0)) * dv),
        "dqv_negative_integral": lambda x, dv: float(np.trapz(np.minimum(x, 0)) * dv),
    },
}


class FeatureExtractor:
    """Compute statistical features from a single delta Q(V) array.

    Operates on the finite subset only. SOH-proxy names are blocked at init.
    Returns NaN for any feature that fails (never raises on individual failure).
    """

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        funs: Dict[str, Any] = dict(_CORE_FEATURES)
        funs.update(_CHEMISTRY_FEATURES.get(cfg.chemistry.lower(), {}))
        leaked = set(funs) & _SOH_PROXY_NAMES
        if leaked:
            raise RuntimeError(f"SOH leakage: feature names {leaked} are forbidden SOH proxies.")
        self._funs = funs

    @property
    def feature_names(self) -> List[str]:
        return list(self._funs)

    def extract(self, delta_qv: np.ndarray) -> Dict[str, float]:
        """Extract all features. Clips outliers before computation."""
        dv    = self.cfg.dv
        valid = delta_qv[np.isfinite(delta_qv)]

        if len(valid) < 10:
            logger.warning("delta Q(V) has only %d finite pts; returning NaN.", len(valid))
            return {n: float("nan") for n in self._funs}

        sigma = self.cfg.outlier_sigma
        if sigma > 0:
            mu, std = np.mean(valid), np.std(valid)
            if std > 0:
                valid = np.clip(valid, mu - sigma * std, mu + sigma * std)

        out: Dict[str, float] = {}
        for name, fn in self._funs.items():
            try:
                out[name] = fn(valid, dv)
            except Exception as exc:
                logger.warning("Feature %r failed (%s); NaN.", name, exc)
                out[name] = float("nan")
        return out


# ---------------------------------------------------------------------------
# 5. Feature matrix builder
# ---------------------------------------------------------------------------

@dataclass
class CycleRecord:
    """All inputs for one (cell, cycle) observation."""
    cell_id: str
    cycle_number: int
    half: pd.DataFrame           # raw half-cycle DataFrame: time, voltage, current
    soh: Optional[float] = None  # target label only -- never a feature
    temperature: Optional[float] = None
    protocol: Optional[str] = None
    extra_meta: Dict[str, Any] = field(default_factory=dict)


class FeatureMatrixBuilder:
    """Orchestrate Q(V) -> delta Q(V) -> features -> DataFrame per experiment.

    Output columns:
      cell_id, cycle_number, soh, temperature, protocol,
      ref_cycle_index, config_hash, extraction_ok, <feature columns>

    The returned DataFrame is raw -- no scaling applied.
    Scale only inside cross-validation folds via make_soh_pipeline().
    """

    def __init__(self, cfg: FeatureConfig,
                 nominal_capacity_ah: Optional[float] = None) -> None:
        self.cfg = cfg
        self._qv   = QVExtractor(cfg, nominal_capacity_ah)
        self._feat = FeatureExtractor(cfg)

    def build(
        self,
        records: List[CycleRecord],
        per_cell_reference: Optional[Dict[str, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Build the feature matrix. Failed cycles get NaN features + extraction_ok=False."""
        by_cell: Dict[str, List[CycleRecord]] = {}
        for rec in records:
            by_cell.setdefault(rec.cell_id, []).append(rec)

        rows: List[Dict[str, Any]] = []

        for cell_id, cell_recs in by_cell.items():
            cell_recs = sorted(cell_recs, key=lambda r: r.cycle_number)
            logger.info("Processing cell %s (%d cycles).", cell_id, len(cell_recs))

            # Q(V) extraction for all cycles in this cell
            q_curves: Dict[int, Optional[np.ndarray]] = {}
            for rec in cell_recs:
                q_smooth, _, info = self._qv.extract(rec.half)
                q_curves[rec.cycle_number] = q_smooth
                if not info.get("ok"):
                    logger.debug("  Cycle %d failed: %s",
                                 rec.cycle_number, info.get("fail_reason"))

            # Per-cell reference selection
            delta = DeltaQVComputer(self.cfg)
            if per_cell_reference is not None and cell_id in per_cell_reference:
                try:
                    delta.set_reference(per_cell_reference[cell_id])
                    ref_idx = delta.reference_cycle_index
                except ValueError as exc:
                    logger.error("Cell %s: external reference invalid: %s"
                                 " -- NaN features for all cycles.", cell_id, exc)
                    ref_idx = None
            else:
                try:
                    ref_idx = delta.fit_reference(q_curves)
                except ValueError as exc:
                    logger.error("Cell %s: %s -- NaN features for all cycles.", cell_id, exc)
                    ref_idx = None

            for rec in cell_recs:
                row: Dict[str, Any] = {
                    "cell_id": cell_id,
                    "cycle_number": rec.cycle_number,
                    "soh": rec.soh,
                    "temperature": rec.temperature,
                    "protocol": rec.protocol,
                    "ref_cycle_index": ref_idx,
                    "config_hash": self.cfg.config_hash(),
                    "extraction_ok": False,
                    **rec.extra_meta,
                }
                q_cycle = q_curves.get(rec.cycle_number)
                if q_cycle is None or ref_idx is None:
                    row.update({n: float("nan") for n in self._feat.feature_names})
                    rows.append(row)
                    continue

                dqv      = delta.compute(q_cycle)
                features = self._feat.extract(dqv)
                row.update(features)
                row["extraction_ok"] = any(np.isfinite(v) for v in features.values())
                rows.append(row)

        df = pd.DataFrame(rows)
        _check_no_soh_leakage(df, self._feat.feature_names)
        logger.info("Feature matrix: %d rows x %d features | %.1f%% ok.",
                    len(df), len(self._feat.feature_names),
                    100 * df["extraction_ok"].mean())
        return df


def _check_no_soh_leakage(df: pd.DataFrame, feature_names: List[str]) -> None:
    leaked = set(feature_names) & _SOH_PROXY_NAMES
    if leaked:
        raise RuntimeError(
            f"SOH LEAKAGE DETECTED: {leaked}. Remove from _CORE_FEATURES/_CHEMISTRY_FEATURES.")


# ---------------------------------------------------------------------------
# 6. sklearn transformers (NaN-aware; always fit on training data only)
# ---------------------------------------------------------------------------

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip feature values beyond +-n_sigma of the TRAINING mean/std.

    Preserves NaN positions so NaNFlagImputer can detect them downstream.
    """

    def __init__(self, n_sigma: float = 5.0) -> None:
        self.n_sigma = n_sigma

    def fit(self, X: np.ndarray, y=None) -> "OutlierClipper":
        X = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean_ = np.nanmean(X, axis=0)
            self.std_  = np.nanstd(X, axis=0)
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X = np.asarray(X, dtype=float).copy()
        for j in range(X.shape[1]):
            col    = X[:, j]
            finite = np.isfinite(col)
            if finite.any():
                col[finite] = np.clip(col[finite],
                                      self.mean_[j] - self.n_sigma * self.std_[j],
                                      self.mean_[j] + self.n_sigma * self.std_[j])
        return X


class NaNFlagImputer(BaseEstimator, TransformerMixin):
    """Append binary NaN-indicator columns, then median-impute.

    fit() records which columns had NaN in training; transform() appends
    the same indicator columns for any input, preserving missingness info.
    """

    def __init__(self, strategy: str = "median", add_indicator: bool = True) -> None:
        self.strategy      = strategy
        self.add_indicator = add_indicator

    def fit(self, X: np.ndarray, y=None) -> "NaNFlagImputer":
        X = np.asarray(X, dtype=float)
        self._n_in    = X.shape[1]
        self._nan_cols = np.where(np.isnan(X).any(axis=0))[0]
        self._imputer = SimpleImputer(strategy=self.strategy)
        self._imputer.fit(X)
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X     = np.asarray(X, dtype=float)
        parts = [self._imputer.transform(X)]
        if self.add_indicator and len(self._nan_cols) > 0:
            parts.append(np.isnan(X[:, self._nan_cols]).astype(float))
        return np.hstack(parts)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        base  = list(input_features) if input_features is not None \
                else [f"x{i}" for i in range(self._n_in)]
        names = list(base)
        if self.add_indicator:
            names += [f"{base[i]}_nan_flag" for i in self._nan_cols]
        return np.array(names)


# ---------------------------------------------------------------------------
# 7. Preprocessing pipeline factory
# ---------------------------------------------------------------------------

def make_soh_pipeline(cfg: FeatureConfig, estimator=None,
                      scaler_type: str = "robust") -> Pipeline:
    """
    Build: NaN-flag-impute -> outlier-clip -> scale [-> estimator].

    Scaling is encapsulated so it is always fit-on-train only when used
    inside CellGroupKFold. Never call fit() on the full dataset.

    Args:
        cfg         : FeatureConfig that produced the feature matrix.
        estimator   : optional sklearn estimator as the final step.
        scaler_type : "robust" (default) | "standard".
    """
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler_type {scaler_type!r}.")

    steps: List[Tuple[str, Any]] = [
        ("nan_imputer",  NaNFlagImputer(strategy="median", add_indicator=True)),
        ("outlier_clip", OutlierClipper(n_sigma=cfg.outlier_sigma)),
        ("scaler",       scaler),
    ]
    if estimator is not None:
        steps.append(("estimator", estimator))

    pipe = Pipeline(steps)
    pipe._config_hash = cfg.config_hash()  # type: ignore[attr-defined]
    return pipe


# ---------------------------------------------------------------------------
# 8. Cell-group-aware cross-validation splitter
# ---------------------------------------------------------------------------

class CellGroupKFold(BaseCrossValidator):
    """K-fold CV keeping all cycles from one cell in the same fold.

    Random cycle splits within a cell leak the degradation trajectory
    (cycles are autocorrelated in time) and inflate apparent generalisation.
    Use n_splits == n_cells for leave-one-cell-out CV.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True,
                 random_state: Optional[int] = None) -> None:
        self.n_splits     = n_splits
        self.shuffle      = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None
              ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if groups is None:
            raise ValueError("CellGroupKFold requires groups (cell_id per sample).")
        groups       = np.asarray(groups)
        unique_cells = np.unique(groups)
        n_cells      = len(unique_cells)

        if self.n_splits > n_cells:
            raise ValueError(f"n_splits ({self.n_splits}) > n_cells ({n_cells}).")

        if self.shuffle:
            rng          = np.random.RandomState(self.random_state)
            unique_cells = rng.permutation(unique_cells)

        for test_batch in np.array_split(unique_cells, self.n_splits):
            test_mask = np.isin(groups, test_batch)
            yield np.where(~test_mask)[0], np.where(test_mask)[0]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _iter_test_indices(self, X=None, y=None, groups=None):
        for _, test_idx in self.split(X, y, groups):
            yield test_idx

# ---------------------------------------------------------------------------
# 9. Pipeline persistence with config-hash validation
# ---------------------------------------------------------------------------

def save_pipeline(pipeline: Pipeline, cfg: FeatureConfig,
                  path) -> Path:
    """Save pipeline + FeatureConfig + hash. Refuses mismatched hashes.

    Also writes a feature_schema.json sibling so the consuming step always
    knows the exact column contract for this artifact.
    """
    path     = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = cfg.config_hash()
    stored   = getattr(pipeline, "_config_hash", None)
    if stored is not None and stored != expected:
        raise ValueError(f"Pipeline hash {stored!r} != config hash {expected!r}.")
    joblib.dump({"pipeline": pipeline, "feature_config": cfg,
                 "config_hash": expected}, path)
    schema_path = path.with_name(path.stem + "_schema.json")
    _save_schema_json(schema_path)
    logger.info("Saved pipeline -> %s  (hash=%s)", path, expected)
    logger.info("Saved feature schema -> %s", schema_path)
    return path


def load_pipeline(path) -> Tuple[Pipeline, FeatureConfig]:
    """Load pipeline and validate config hash. Raises on stale artifacts."""
    path   = Path(path)
    bundle = joblib.load(path)
    cfg    = bundle["feature_config"]
    stored = bundle["config_hash"]
    recomp = cfg.config_hash()
    if stored != recomp:
        raise ValueError(f"Stale artifact {path}: {stored!r} != {recomp!r}.")
    logger.info("Loaded pipeline <- %s  (hash=%s)", path, stored)
    return bundle["pipeline"], cfg

# ---------------------------------------------------------------------------
# 10. Feature stability monitoring
# ---------------------------------------------------------------------------

def monitor_feature_stability(
    feature_df: pd.DataFrame,
    feature_names: List[str],
    group_col: str = "cell_id",
    reference_group: Optional[str] = None,
) -> pd.DataFrame:
    """
    Assess feature transferability across cells or datasets.
    CV of per-group medians > 0.5 flags instability.
    """
    present = [f for f in feature_names if f in feature_df.columns]
    if not present:
        raise ValueError("None of feature_names found in feature_df.")
    groups  = feature_df[group_col].unique()
    records: List[Dict[str, Any]] = []
    for feat in present:
        medians, iqrs = [], []
        for grp in groups:
            vals = feature_df.loc[feature_df[group_col] == grp, feat].dropna().to_numpy()
            if len(vals) < 2:
                continue
            medians.append(float(np.median(vals)))
            iqrs.append(float(np.percentile(vals, 75) - np.percentile(vals, 25)))
        if not medians:
            records.append({"feature": feat, "stable": False, "note": "no_valid_data"})
            continue
        med_arr = np.array(medians)
        cv      = float(np.std(med_arr) / (np.abs(np.mean(med_arr)) + 1e-12))
        ref_shifts = None
        if reference_group is not None and reference_group in groups:
            rv = feature_df.loc[feature_df[group_col] == reference_group, feat].dropna().to_numpy()
            if len(rv) >= 2:
                ref_med    = float(np.median(rv))
                ref_shifts = {}
                for grp in groups:
                    g = feature_df.loc[feature_df[group_col] == grp, feat].dropna().to_numpy()
                    if len(g) >= 2:
                        ref_shifts[str(grp)] = float(np.median(g) - ref_med)
        records.append({
            "feature": feat, "n_groups": len(medians),
            "median_of_medians": float(np.mean(med_arr)),
            "cv_of_medians": cv, "mean_iqr": float(np.mean(iqrs)),
            "stable": cv < 0.5, "ref_shifts": ref_shifts,
        })
    stability  = pd.DataFrame(records).set_index("feature")
    n_unstable = int((~stability["stable"]).sum())
    if n_unstable:
        logger.warning("%d/%d features UNSTABLE (CV>0.5): %s", n_unstable, len(present),
                       list(stability.index[~stability["stable"]]))
    return stability

# ---------------------------------------------------------------------------
# 11. SOH target helper
# ---------------------------------------------------------------------------

def compute_soh(discharge_capacity_ah: float,
                nominal_capacity_ah: float) -> float:
    """SOH = discharge_capacity / nominal_capacity. Target label -- never a feature."""
    if nominal_capacity_ah <= 0:
        raise ValueError("nominal_capacity_ah must be positive.")
    return float(discharge_capacity_ah / nominal_capacity_ah)


# ---------------------------------------------------------------------------
# 12. Quick-start demo (synthetic data)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    rng = np.random.default_rng(42)
    cfg = FeatureConfig(v_min=2.5, v_max=4.2, dv=0.005, chemistry="generic")

    def _synthetic_discharge(cycle: int, soh: float, n: int = 800) -> pd.DataFrame:
        t = np.linspace(0.0, 36000.0 * soh, n)
        v = np.linspace(4.15, 2.55, n) + rng.normal(0, 0.003, n)
        i = np.full(n, -1.0 * soh) + rng.normal(0, 0.005, n)
        return pd.DataFrame({"time": t, "voltage": v, "current": i})

    records: List[CycleRecord] = []
    for cell_num in range(1, 4):
        for cyc in range(1, 16):
            soh = max(0.70, 1.0 - 0.02 * cyc + rng.normal(0, 0.004))
            records.append(CycleRecord(
                cell_id=f"cell_{cell_num:02d}",
                cycle_number=cyc,
                half=_synthetic_discharge(cyc, soh),
                soh=soh,
                temperature=25.0 + rng.normal(0, 0.5),
                protocol="C/5_discharge",
            ))

    builder    = FeatureMatrixBuilder(cfg, nominal_capacity_ah=10.0)
    feat_df    = builder.build(records)
    feat_names = FeatureExtractor(cfg).feature_names
    print(f"\nFeature matrix: {feat_df.shape}")
    print(feat_df[["cell_id", "cycle_number", "soh", "extraction_ok"]
                  + feat_names].head(6).to_string())

    X      = feat_df[feat_names].to_numpy()
    y      = feat_df["soh"].to_numpy()
    groups = feat_df["cell_id"].to_numpy()

    print("\nCross-validation folds (CellGroupKFold, n_splits=3):")
    for fold, (tr, te) in enumerate(
            CellGroupKFold(n_splits=3, random_state=0).split(X, y, groups)):
        pipe = make_soh_pipeline(cfg)
        pipe.fit(X[tr])
        X_te = pipe.transform(X[te])
        print(f"  Fold {fold}: train={len(tr)} {np.unique(groups[tr])} |",
              f"test={len(te)} {np.unique(groups[te])} | X_test {X_te.shape}")

    print("\nFeature stability:")
    stability = monitor_feature_stability(feat_df, feat_names)
    print(stability[["cv_of_medians", "stable"]].to_string())

    from sklearn.linear_model import Ridge
    full_pipe = make_soh_pipeline(cfg, estimator=Ridge(alpha=1.0))
    full_pipe.fit(X, y)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    sp = save_pipeline(full_pipe, cfg, cfg.cache_dir / "demo_ridge.joblib")
    _, loaded_cfg = load_pipeline(sp)
    print(f"\nSave/load round-trip OK. Hash: {loaded_cfg.config_hash()}")