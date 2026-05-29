"""
battery_health_report.py
Unified battery health assessment combining ML SOH predictions with
physics-based ICA degradation features.

Typical usage
-------------
    from step4_soh_model import SOHModelTrainer
    from step4_interpretation import build_physics_features
    from battery_health_report import build_report, save_report

    model_results = SOHModelTrainer().run("features.csv", "models/")
    phys_df       = build_physics_features(df_ica, voltage_grid)
    report        = build_report(
                        predictions       = model_results.predictions,
                        overall_metrics   = model_results.overall_metrics,
                        per_cell_metrics  = model_results.per_cell_metrics,
                        feature_importance= model_results.feature_importance,
                        phys_df           = phys_df,
                    )
    save_report(report, "reports/run_01/")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORT_VERSION = "1.0"

# Physics columns expected from step4_interpretation.build_physics_features().
PHYSICS_COLS: List[str] = [
    "LLI_mean_shift",
    "LLI_n_matched",
    "LLI_n_disappeared",
    "LLI_n_appeared",
    "LLI_confidence",
    "LLI_disappearance_penalty",
    "LAM_loss",
    "Resistance_growth",
    "mean_ref_width_mV",
    "mean_curr_width_mV",
    "grid_spacing_mV",
]

# Signals used for per-cell linear trend analysis.
TREND_SIGNALS: List[str] = [
    "soh_pred",
    "LLI_mean_shift",
    "LAM_loss",
    "Resistance_growth",
]

# Reference scale for each mechanism — "one unit of significant change per cycle".
# Used to normalize slopes before picking the dominant mechanism.
_MECHANISM_SCALES: Dict[str, Dict] = {
    "LLI":               {"col": "LLI_mean_shift",   "scale": 1e-4},  # 100 µV/cycle
    "LAM":               {"col": "LAM_loss",          "scale": 5e-4},  # 0.05%/cycle
    "resistance_growth": {"col": "Resistance_growth", "scale": 1e-3},  # 0.1%/cycle
}


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class BatteryHealthReport:
    """Complete battery health assessment.

    Fields
    ------
    cycle_df
        One row per (cell_id, cycle_number).  Contains soh_true, soh_pred,
        soh_residual, soh_abs_residual, all physics features, and optionally
        the raw dqv_* statistical features.
    cell_summary
        One row per cell.  Aggregates SOH endpoints, prediction metrics,
        mechanism trends, dominant mechanism, and overall confidence.
    trend_df
        One row per (cell_id, signal).  Linear fit (slope, intercept, r2)
        for each TREND_SIGNALS over the cycle history.
    model_metrics
        {"overall": {rmse, mae, r2}, "per_cell": {cell_id: {rmse, mae, r2}}}.
    feature_importance
        DataFrame with feature, coefficient, abs_coefficient.
    mechanism_evidence
        Per-cell dict of mechanism evidence dicts ready for JSON export.
    report_version
        Schema version; bump when adding/removing top-level report fields.
    created_at
        ISO 8601 UTC timestamp of report creation.
    """

    cycle_df:           pd.DataFrame
    cell_summary:       pd.DataFrame
    trend_df:           pd.DataFrame
    model_metrics:      dict
    feature_importance: pd.DataFrame
    mechanism_evidence: dict
    report_version:     str = field(default=REPORT_VERSION)
    created_at:         str = field(default="")

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public constructor
# ---------------------------------------------------------------------------

def build_report(
    *,
    predictions:        pd.DataFrame,
    overall_metrics:    dict,
    per_cell_metrics:   dict,
    feature_importance: pd.DataFrame,
    phys_df:            pd.DataFrame,
    feature_df:         Optional[pd.DataFrame] = None,
) -> BatteryHealthReport:
    """Build a BatteryHealthReport from ML + physics outputs.

    Parameters
    ----------
    predictions
        DataFrame with columns: cell_id, cycle_number, y_true, y_pred.
        Typically ModelResults.predictions from SOHModelTrainer.
    overall_metrics
        {"rmse": float, "mae": float, "r2": float}.
    per_cell_metrics
        {cell_id: {"rmse": float, "mae": float, "r2": float}}.
    feature_importance
        DataFrame with columns: feature, coefficient, abs_coefficient.
    phys_df
        Output of step4_interpretation.build_physics_features().
        Required columns: cell_id, cycle_number + PHYSICS_COLS.
    feature_df
        Optional.  Raw feature matrix from step3 FeatureMatrixBuilder.build().
        When supplied, dqv_* columns are joined into cycle_df.
    """
    _validate_inputs(predictions, per_cell_metrics, phys_df, feature_importance)

    cycle_df    = _merge_all(predictions, phys_df, feature_df)
    trend_df    = _compute_trends(cycle_df)
    cell_summary = _compute_cell_summary(cycle_df, trend_df, per_cell_metrics)
    mech_ev     = _mechanism_evidence_dict(cell_summary, trend_df)

    return BatteryHealthReport(
        cycle_df          = cycle_df,
        cell_summary      = cell_summary,
        trend_df          = trend_df,
        model_metrics     = {"overall": overall_metrics, "per_cell": per_cell_metrics},
        feature_importance= feature_importance.copy(),
        mechanism_evidence= mech_ev,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    predictions:        pd.DataFrame,
    per_cell_metrics:   dict,
    phys_df:            pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    _check_cols(predictions, ["cell_id", "cycle_number", "y_true", "y_pred"],
                "predictions")
    _check_cols(phys_df,     ["cell_id", "cycle_number"],
                "phys_df")
    _check_cols(feature_importance, ["feature", "coefficient"],
                "feature_importance")

    _check_no_duplicate_keys(predictions, "predictions")
    _check_no_duplicate_keys(phys_df,     "phys_df")

    # per_cell_metrics keys must be a superset of prediction cells
    pred_cells  = set(predictions["cell_id"].unique())
    metric_keys = set(per_cell_metrics.keys())
    missing     = pred_cells - metric_keys
    if missing:
        raise ValueError(
            f"per_cell_metrics is missing entries for cell(s): {sorted(missing)}"
        )

    # Check for physics columns — warn but do not fail (phys_df may be partial)
    missing_phys = [c for c in PHYSICS_COLS if c not in phys_df.columns]
    if missing_phys:
        import warnings
        warnings.warn(
            f"phys_df is missing {len(missing_phys)} physics column(s): "
            f"{missing_phys}.  Affected cycle_df cells will contain NaN.",
            stacklevel=3,
        )


def _check_cols(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {missing}")


def _check_no_duplicate_keys(df: pd.DataFrame, name: str) -> None:
    dupes = df.duplicated(subset=["cell_id", "cycle_number"])
    if dupes.any():
        bad = df.loc[dupes, ["cell_id", "cycle_number"]].to_dict("records")
        raise ValueError(f"Duplicate (cell_id, cycle_number) in {name}: {bad}")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _merge_all(
    predictions: pd.DataFrame,
    phys_df:     pd.DataFrame,
    feature_df:  Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build the per-(cell, cycle) table from all input sources."""
    df = predictions.rename(columns={"y_true": "soh_true", "y_pred": "soh_pred"}).copy()
    df["soh_residual"]     = df["soh_pred"] - df["soh_true"]
    df["soh_abs_residual"] = df["soh_residual"].abs()

    # Join physics features (left: keep all prediction rows, NaN where phys absent)
    phys_cols = ["cell_id", "cycle_number"] + [
        c for c in PHYSICS_COLS if c in phys_df.columns
    ]
    df = df.merge(phys_df[phys_cols], on=["cell_id", "cycle_number"], how="left")

    # Optionally join raw dqv_* statistical features
    if feature_df is not None:
        from feature_schema import FEATURE_COLS
        feat_cols = ["cell_id", "cycle_number"] + [
            c for c in FEATURE_COLS if c in feature_df.columns
        ]
        df = df.merge(feature_df[feat_cols], on=["cell_id", "cycle_number"], how="left")

    return df.sort_values(["cell_id", "cycle_number"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

def _fit_trend(x: np.ndarray, y: np.ndarray) -> Dict:
    """OLS linear fit; returns slope, intercept, r2, n_cycles."""
    mask = np.isfinite(y)
    n    = int(mask.sum())
    if n < 2:
        return {"slope": float("nan"), "intercept": float("nan"),
                "r2": float("nan"), "n_cycles": n}
    xm, ym = x[mask], y[mask]
    slope, intercept = np.polyfit(xm, ym, 1)
    y_hat  = slope * xm + intercept
    ss_res = float(np.sum((ym - y_hat) ** 2))
    ss_tot = float(np.sum((ym - ym.mean()) ** 2))
    r2     = 1.0 - ss_res / (ss_tot + 1e-12)
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": float(r2), "n_cycles": n}


def _compute_trends(cycle_df: pd.DataFrame) -> pd.DataFrame:
    """Fit linear trends for each (cell, signal) pair.

    Returns a DataFrame with columns:
        cell_id, signal, slope, intercept, r2, n_cycles
    """
    rows = []
    signals = [s for s in TREND_SIGNALS if s in cycle_df.columns]

    for cell_id, group in cycle_df.groupby("cell_id"):
        x = group["cycle_number"].to_numpy(dtype=float)
        for sig in signals:
            y    = group[sig].to_numpy(dtype=float)
            fit  = _fit_trend(x, y)
            rows.append({"cell_id": cell_id, "signal": sig, **fit})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cell summary
# ---------------------------------------------------------------------------

def _dominant_mechanism(row: pd.Series) -> str:
    """Return the mechanism name with the largest normalised absolute trend slope."""
    scores: Dict[str, float] = {}
    for mech, spec in _MECHANISM_SCALES.items():
        slope_col = f"{spec['col']}_slope"
        slope     = row.get(slope_col, float("nan"))
        if np.isfinite(slope):
            scores[mech] = abs(float(slope)) / spec["scale"]
    if not scores or max(scores.values()) < 0.1:
        return "indeterminate"
    return max(scores, key=scores.__getitem__)


def _compute_cell_summary(
    cycle_df:        pd.DataFrame,
    trend_df:        pd.DataFrame,
    per_cell_metrics: dict,
) -> pd.DataFrame:
    """Build one-row-per-cell summary DataFrame."""
    rows = []

    # Pivot trend_df for easy column access
    trend_pivot = trend_df.pivot(index="cell_id", columns="signal",
                                 values=["slope", "intercept", "r2", "n_cycles"])
    trend_pivot.columns = [f"{val}_{sig}" for val, sig in trend_pivot.columns]
    trend_pivot = trend_pivot.reset_index()

    for cell_id, group in cycle_df.groupby("cell_id"):
        group   = group.sort_values("cycle_number")
        metrics = per_cell_metrics.get(str(cell_id), {})

        row: Dict = {
            "cell_id":               cell_id,
            "n_cycles":              len(group),
            "first_cycle":           int(group["cycle_number"].iloc[0]),
            "last_cycle":            int(group["cycle_number"].iloc[-1]),
            "soh_pred_first":        float(group["soh_pred"].iloc[0]),
            "soh_pred_last":         float(group["soh_pred"].iloc[-1]),
            "soh_pred_total_change": float(group["soh_pred"].iloc[-1]
                                           - group["soh_pred"].iloc[0]),
            "pred_rmse":             float(metrics.get("rmse", float("nan"))),
            "pred_mae":              float(metrics.get("mae",  float("nan"))),
            "pred_r2":               float(metrics.get("r2",   float("nan"))),
            "soh_abs_residual_mean": float(group["soh_abs_residual"].mean()),
        }

        if "soh_true" in group.columns:
            row["soh_true_first"] = float(group["soh_true"].iloc[0])
            row["soh_true_last"]  = float(group["soh_true"].iloc[-1])

        if "LLI_confidence" in group.columns:
            row["LLI_confidence_mean"] = float(group["LLI_confidence"].mean(skipna=True))

        if "LLI_disappearance_penalty" in group.columns:
            row["LLI_disappearance_penalty_mean"] = float(
                group["LLI_disappearance_penalty"].mean(skipna=True)
            )

        rows.append(row)

    summary = pd.DataFrame(rows)

    # Merge trend slopes and R² values
    summary = summary.merge(trend_pivot, on="cell_id", how="left")

    # Derive dominant mechanism from pivoted slope columns
    summary["dominant_mechanism"] = summary.apply(_dominant_mechanism, axis=1)

    # Overall confidence: mean of LLI_confidence weighted with prediction quality
    if "LLI_confidence_mean" in summary.columns:
        pred_conf  = (summary["pred_r2"].clip(0, 1) + 1) / 2  # map [-∞,1] → [0,1]
        lli_conf   = summary["LLI_confidence_mean"].fillna(0.5)
        summary["overall_confidence"] = ((pred_conf + lli_conf) / 2).clip(0.0, 1.0)
    else:
        summary["overall_confidence"] = ((summary["pred_r2"].clip(0, 1) + 1) / 2).clip(0, 1)

    return summary.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mechanism evidence (for JSON export)
# ---------------------------------------------------------------------------

def _mechanism_evidence_dict(
    cell_summary: pd.DataFrame,
    trend_df:     pd.DataFrame,
) -> dict:
    """Build a per-cell dict of mechanism evidence for JSON serialisation."""
    evidence: dict = {}

    for _, row in cell_summary.iterrows():
        cell_id = str(row["cell_id"])
        cell_ev: dict = {}

        for mech, spec in _MECHANISM_SCALES.items():
            col       = spec["col"]
            slope_col = f"{col}_slope"
            r2_col    = f"{col}_r2"
            cell_ev[mech] = {
                "signal":          col,
                "slope_per_cycle": _safe_float(row.get(slope_col)),
                "r2":              _safe_float(row.get(r2_col)),
                "scale_reference": spec["scale"],
                "normalised_rate": _safe_float(
                    abs(row.get(slope_col, float("nan"))) / spec["scale"]
                    if np.isfinite(row.get(slope_col, float("nan"))) else float("nan")
                ),
            }

        if "LLI_confidence_mean" in row.index:
            cell_ev["LLI"]["peak_match_confidence"] = _safe_float(
                row.get("LLI_confidence_mean")
            )
        if "LLI_disappearance_penalty_mean" in row.index:
            cell_ev["LLI"]["disappearance_penalty"] = _safe_float(
                row.get("LLI_disappearance_penalty_mean")
            )

        cell_ev["dominant_mechanism"] = str(row.get("dominant_mechanism", "indeterminate"))
        evidence[cell_id] = cell_ev

    return evidence


def _safe_float(val) -> Optional[float]:
    """Convert to Python float; return None for NaN/None so JSON stays clean."""
    try:
        f = float(val)
        return None if not np.isfinite(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def save_report(
    report:     BatteryHealthReport,
    output_dir: str | Path,
) -> Dict[str, Path]:
    """Write all report artifacts to output_dir.

    Written files
    -------------
    cycle_data.csv          — per-(cell, cycle) merged table
    cell_summary.csv        — per-cell aggregates and mechanism labels
    trend_analysis.csv      — per-(cell, signal) linear fit results
    feature_importance.csv  — ElasticNet coefficients
    summary.json            — model metrics + mechanism evidence + metadata
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    paths["cycle_data"]        = out / "cycle_data.csv"
    paths["cell_summary"]      = out / "cell_summary.csv"
    paths["trend_analysis"]    = out / "trend_analysis.csv"
    paths["feature_importance"]= out / "feature_importance.csv"
    paths["summary_json"]      = out / "summary.json"

    report.cycle_df.to_csv(paths["cycle_data"], index=False)
    report.cell_summary.to_csv(paths["cell_summary"], index=False)
    report.trend_df.to_csv(paths["trend_analysis"], index=False)
    report.feature_importance.to_csv(paths["feature_importance"], index=False)

    summary = {
        "report_version":    report.report_version,
        "created_at":        report.created_at,
        "model_metrics":     _json_safe(report.model_metrics),
        "mechanism_evidence":report.mechanism_evidence,
        "cells":             sorted(report.cycle_df["cell_id"].unique().tolist()),
        "n_cells":           int(report.cycle_df["cell_id"].nunique()),
        "n_cycles_total":    int(len(report.cycle_df)),
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2, default=str))

    return paths


def _json_safe(obj):
    """Recursively convert numpy scalars to Python native types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    return obj


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _require_matplotlib(fn_name: str):
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            f"{fn_name} requires matplotlib.  Install with: pip install matplotlib"
        ) from exc


def plot_soh_predictions(
    report:  BatteryHealthReport,
    cell_id: Optional[str] = None,
    ax=None,
):
    """True vs predicted SOH over cycles for one or all cells.

    Returns a dict mapping cell_id → matplotlib Axes,
    or a single Axes if cell_id is specified.
    """
    plt = _require_matplotlib("plot_soh_predictions")

    cells   = [cell_id] if cell_id else sorted(report.cycle_df["cell_id"].unique())
    n_cells = len(cells)

    if ax is None:
        ncols = min(n_cells, 3)
        nrows = (n_cells + ncols - 1) // ncols
        _, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                               squeeze=False)
        flat_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    else:
        flat_axes = [ax] * n_cells

    result = {}
    for i, cid in enumerate(cells):
        a   = flat_axes[i]
        sub = report.cycle_df[report.cycle_df["cell_id"] == cid].sort_values("cycle_number")

        if "soh_true" in sub.columns:
            a.plot(sub["cycle_number"], sub["soh_true"], "k--",
                   linewidth=1.2, label="True SOH")
        a.plot(sub["cycle_number"], sub["soh_pred"], "b-",
               linewidth=1.5, label="Predicted SOH")
        a.fill_between(
            sub["cycle_number"],
            sub["soh_pred"] - sub["soh_abs_residual"],
            sub["soh_pred"] + sub["soh_abs_residual"],
            alpha=0.2, color="blue", label="±|residual|",
        )
        a.set_title(str(cid))
        a.set_xlabel("Cycle")
        a.set_ylabel("SOH")
        a.legend(fontsize="x-small")
        a.set_ylim(bottom=0)
        result[str(cid)] = a

    # Hide unused subplots
    if ax is None:
        for j in range(n_cells, len(flat_axes)):
            flat_axes[j].set_visible(False)

    return result if cell_id is None else result[str(cell_id)]


def plot_mechanism_trends(
    report:  BatteryHealthReport,
    cell_id: Optional[str] = None,
    axes=None,
):
    """LLI / LAM / resistance_growth_frac over cycles for one or all cells.

    Returns dict: signal_name → Axes.
    """
    plt = _require_matplotlib("plot_mechanism_trends")

    cells   = [cell_id] if cell_id else sorted(report.cycle_df["cell_id"].unique())
    signals = [s for s in ["LLI_mean_shift", "LAM_loss", "Resistance_growth"]
               if s in report.cycle_df.columns]

    if not signals:
        raise ValueError("cycle_df contains no physics mechanism columns.")

    if axes is None:
        _, ax_arr = plt.subplots(1, len(signals),
                                 figsize=(5 * len(signals), 4), squeeze=False)
        axes_list = list(ax_arr[0])
    else:
        axes_list = list(axes) if hasattr(axes, "__iter__") else [axes]

    ylabels = {
        "LLI_mean_shift":   "LLI shift (V)",
        "LAM_loss":          "LAM loss (frac.)",
        "Resistance_growth": "Resistance growth (frac.)",
    }

    result = {}
    for ax_i, sig in zip(axes_list, signals):
        for cid in cells:
            sub = report.cycle_df[report.cycle_df["cell_id"] == cid].sort_values("cycle_number")
            if sig in sub.columns:
                ax_i.plot(sub["cycle_number"], sub[sig], marker=".", markersize=4,
                          label=str(cid))
        ax_i.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax_i.set_xlabel("Cycle")
        ax_i.set_ylabel(ylabels.get(sig, sig))
        ax_i.set_title(sig.replace("_", " "))
        ax_i.legend(fontsize="x-small")
        result[sig] = ax_i

    return result


def plot_feature_importance(
    report: BatteryHealthReport,
    top_n:  int = 10,
    ax=None,
):
    """Horizontal bar chart of ElasticNet feature coefficients.

    Returns matplotlib Axes.
    """
    plt = _require_matplotlib("plot_feature_importance")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(3, top_n * 0.4)))

    fi = (report.feature_importance
          .nlargest(top_n, "abs_coefficient")
          .sort_values("abs_coefficient"))

    colors = ["steelblue" if c >= 0 else "tomato" for c in fi["coefficient"]]
    ax.barh(fi["feature"], fi["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ElasticNet coefficient")
    ax.set_title(f"Top {top_n} features by |coefficient|")
    return ax


def plot_dashboard(
    report:  BatteryHealthReport,
    cell_id: Optional[str] = None,
):
    """Four-panel dashboard: SOH predictions, mechanism trends, feature importance.

    Returns dict of panel name → Axes.
    """
    plt = _require_matplotlib("plot_dashboard")

    fig  = plt.figure(figsize=(16, 10))
    spec = fig.add_gridspec(2, 3)

    ax_soh  = fig.add_subplot(spec[0, :2])   # top-left wide
    ax_fi   = fig.add_subplot(spec[0, 2])    # top-right
    ax_mech = [fig.add_subplot(spec[1, i]) for i in range(3)]

    result: dict = {}

    # SOH panel
    cells = [cell_id] if cell_id else sorted(report.cycle_df["cell_id"].unique())
    for cid in cells:
        sub = report.cycle_df[report.cycle_df["cell_id"] == cid].sort_values("cycle_number")
        if "soh_true" in sub.columns:
            ax_soh.plot(sub["cycle_number"], sub["soh_true"],  "k--", linewidth=1.0)
        ax_soh.plot(sub["cycle_number"], sub["soh_pred"], linewidth=1.5, label=str(cid))
    ax_soh.set_xlabel("Cycle")
    ax_soh.set_ylabel("SOH")
    ax_soh.set_title("SOH: true (dashed) vs predicted")
    ax_soh.legend(fontsize="x-small")
    result["soh"] = ax_soh

    # Feature importance panel
    result["feature_importance"] = plot_feature_importance(report, top_n=8, ax=ax_fi)

    # Mechanism panels
    sigs = [s for s in ["LLI_mean_shift", "LAM_loss", "Resistance_growth"]
            if s in report.cycle_df.columns]
    mech_axes = plot_mechanism_trends(report, cell_id=cell_id, axes=ax_mech[:len(sigs)])
    result.update(mech_axes)

    plt.tight_layout()
    result["figure"] = fig
    return result
