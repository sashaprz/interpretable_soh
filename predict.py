"""
predict.py
==========
Inference-time SOH prediction and degradation explanation for a single cycle.

Flow 2: user provides one discharge half-cycle measurement for a cell of a known
chemistry, and receives:
  - Predicted SOH (0–1 float)
  - Physics explanation (LLI, LAM, resistance growth)
  - Feature vector used for the prediction

Typical usage::

    predictor = SOHPredictor.from_model("models/elasticnet_soh.joblib")
    predictor.set_reference(ref_df)          # healthy early-life cycle
    result = predictor.assess(curr_df)       # current cycle to evaluate
    print(result["soh_pred"])
    print(result["explanation"])
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from feature_schema import FEATURE_COLS

# ---------------------------------------------------------------------------
# Lazy module loader (handles parenthesised filenames)
# ---------------------------------------------------------------------------

_MODULE_CACHE: dict[str, types.ModuleType] = {}


def _load_step(slug: str, filename: str) -> types.ModuleType:
    if slug in _MODULE_CACHE:
        return _MODULE_CACHE[slug]
    here = Path(__file__).parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    path = here / filename
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {path}")
    spec   = importlib.util.spec_from_file_location(slug, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[slug] = module
    spec.loader.exec_module(module)          # type: ignore[union-attr]
    _MODULE_CACHE[slug] = module
    return module


def _s3():
    return _load_step("_step3_pred", "step3_deltaQ(V)_feature_extraction.py")


# ---------------------------------------------------------------------------
# Natural-language explanation generator
# ---------------------------------------------------------------------------

def _explain(physics: dict, soh_pred: float) -> str:
    parts: list[str] = []

    lli_shift = float(physics.get("LLI_mean_shift", 0.0) or 0.0)
    lli_conf  = float(physics.get("LLI_confidence",  0.0) or 0.0)
    lam       = float(physics.get("LAM_loss",         0.0) or 0.0)
    resist    = float(physics.get("Resistance_growth", 0.0) or 0.0)
    n_dis     = int(physics.get("LLI_n_disappeared", 0)    or 0)

    parts.append(f"Predicted SOH: {soh_pred:.1%}.")

    if lli_conf > 0.5 and abs(lli_shift) > 0.003:
        direction = "higher" if lli_shift > 0 else "lower"
        parts.append(
            f"Lithium inventory loss (LLI): ICA peaks shifted "
            f"{lli_shift * 1000:.1f} mV to {direction} voltage "
            f"(match confidence {lli_conf:.0%})."
        )
    elif n_dis > 0:
        parts.append(
            f"Lithium inventory loss (LLI): {n_dis} reference peak(s) "
            "disappeared, suggesting capacity-limiting Li plating or SEI growth."
        )

    if lam > 0.02:
        parts.append(
            f"Active material loss (LAM): {lam * 100:.1f}% reduction "
            "in total ICA area — electrode active sites consumed."
        )

    if resist > 0.05:
        parts.append(
            f"Resistance growth: ICA peak width increased {resist * 100:.1f}% "
            "(broadening = higher impedance / polarisation)."
        )

    if len(parts) == 1:
        parts.append(
            "No significant degradation signature detected at this cycle."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# SOHPredictor
# ---------------------------------------------------------------------------

class SOHPredictor:
    """Load a trained ElasticNet SOH model and predict on new half-cycle data.

    Parameters
    ----------
    pipeline
        Trained sklearn Pipeline (StandardScaler + ElasticNet).
    feat_cfg
        FeatureConfig instance from step3.  Controls voltage grid, smoothing,
        and CC selection.  Defaults to FeatureConfig() (generic chemistry).
    nominal_capacity_ah
        Optional nominal cell capacity for C-rate gating.
    """

    def __init__(
        self,
        pipeline,
        feat_cfg=None,
        nominal_capacity_ah: Optional[float] = None,
    ) -> None:
        self._pipeline = pipeline
        s3 = _s3()
        self._cfg = feat_cfg if feat_cfg is not None else s3.FeatureConfig()
        self._qv   = s3.QVExtractor(self._cfg, nominal_capacity_ah)
        self._feat = s3.FeatureExtractor(self._cfg)
        self._ref_q_smooth: Optional[np.ndarray] = None
        self._ref_dqdv:     Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model_path: str | Path,
        feat_cfg=None,
        nominal_capacity_ah: Optional[float] = None,
    ) -> "SOHPredictor":
        """Load a saved joblib pipeline and wrap it in SOHPredictor."""
        pipeline = joblib.load(Path(model_path))
        return cls(pipeline, feat_cfg=feat_cfg,
                   nominal_capacity_ah=nominal_capacity_ah)

    # ------------------------------------------------------------------
    # Reference setup
    # ------------------------------------------------------------------

    def set_reference(self, ref_half_df: pd.DataFrame) -> None:
        """Supply the early-life reference half-cycle for ΔQ(V) computation.

        Must be called before :meth:`assess`.  The reference should be an
        early, healthy cycle from the same cell chemistry so that ΔQ(V)
        captures degradation relative to a known-good baseline.

        Parameters
        ----------
        ref_half_df
            DataFrame with columns ``time`` (s), ``voltage`` (V),
            ``current`` (A, negative for discharge).
        """
        q_smooth, dqdv, info = self._qv.extract(ref_half_df)
        if not info.get("ok"):
            raise ValueError(
                f"Reference half-cycle extraction failed: "
                f"{info.get('fail_reason', 'unknown')}. "
                f"CC fraction: {info.get('cc_fraction', 'n/a'):.1%}. "
                f"Check that the data is a CC discharge at low C-rate "
                f"(≤ C/10 if nominal_capacity_ah is set)."
            )
        self._ref_q_smooth = q_smooth
        self._ref_dqdv     = dqdv

    def set_reference_from_q(
        self,
        q_smooth: np.ndarray,
        dqdv: np.ndarray,
    ) -> None:
        """Directly supply pre-extracted reference Q(V) and dQ/dV arrays.

        Use this when the reference was extracted externally (e.g. saved
        alongside a trained model artifact).
        """
        self._ref_q_smooth = np.asarray(q_smooth, dtype=float)
        self._ref_dqdv     = np.asarray(dqdv,     dtype=float)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def assess(self, curr_half_df: pd.DataFrame) -> dict:
        """Predict SOH and explain degradation for one half-cycle.

        Parameters
        ----------
        curr_half_df
            DataFrame with columns ``time`` (s), ``voltage`` (V),
            ``current`` (A) for the cycle being assessed.

        Returns
        -------
        dict with keys:
            soh_pred        : float in [0, 1] — predicted state of health
            features        : dict[str, float] — ΔQ(V) feature vector
            physics         : dict — LLI / LAM / resistance metrics
            explanation     : str — human-readable degradation summary
            extraction_info : dict — QV extraction diagnostics
        """
        if self._ref_q_smooth is None:
            raise RuntimeError(
                "Call set_reference() before assess()."
            )

        from step4_interpretation import interpret_cycle

        q_smooth, dqdv, info = self._qv.extract(curr_half_df)
        if not info.get("ok"):
            raise ValueError(
                f"Current half-cycle extraction failed: "
                f"{info.get('fail_reason', 'unknown')}."
            )

        # ΔQ(V) and statistical features
        delta_qv = q_smooth - self._ref_q_smooth
        features = self._feat.extract(delta_qv)

        X = np.array([[features.get(c, np.nan) for c in FEATURE_COLS]])
        soh_pred = float(self._pipeline.predict(X)[0])

        # Physics interpretation — restrict to positions finite in both ref and curr
        voltage_grid = self._cfg.v_grid
        try:
            finite_mask = np.isfinite(self._ref_dqdv) & np.isfinite(dqdv)
            if finite_mask.sum() >= 20:
                physics_metrics, _lli = interpret_cycle(
                    voltage_grid[finite_mask],
                    self._ref_dqdv[finite_mask],
                    dqdv[finite_mask],
                )
            else:
                physics_metrics = {}
        except Exception:
            physics_metrics = {}

        return {
            "soh_pred":        soh_pred,
            "features":        features,
            "physics":         physics_metrics,
            "explanation":     _explain(physics_metrics, soh_pred),
            "extraction_info": info,
        }

    # ------------------------------------------------------------------
    # Batch inference (list of half-cycle DataFrames)
    # ------------------------------------------------------------------

    def assess_trajectory(
        self,
        half_cycle_dfs: list[pd.DataFrame],
        cycle_numbers: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """Run :meth:`assess` over a sequence of cycles.

        Parameters
        ----------
        half_cycle_dfs
            Ordered list of half-cycle DataFrames (earliest first).
        cycle_numbers
            Optional cycle index labels; defaults to 0, 1, 2, …

        Returns
        -------
        DataFrame with columns:
            cycle_number, soh_pred, explanation, <physics cols>, <feature cols>
        """
        if cycle_numbers is None:
            cycle_numbers = list(range(len(half_cycle_dfs)))

        rows = []
        for cn, df in zip(cycle_numbers, half_cycle_dfs):
            try:
                result = self.assess(df)
                row: dict = {
                    "cycle_number": cn,
                    "soh_pred":     result["soh_pred"],
                    "explanation":  result["explanation"],
                    **{f"phys_{k}": v for k, v in result["physics"].items()},
                    **{f"feat_{k}": v for k, v in result["features"].items()},
                }
            except Exception as exc:
                row = {
                    "cycle_number": cn,
                    "soh_pred":     float("nan"),
                    "explanation":  f"Extraction failed: {exc}",
                }
            rows.append(row)

        return pd.DataFrame(rows)
