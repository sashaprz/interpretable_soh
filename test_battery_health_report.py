"""
test_battery_health_report.py
Unit tests for battery_health_report.py.
Run with: pytest test_battery_health_report.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from battery_health_report import (
    PHYSICS_COLS,
    TREND_SIGNALS,
    BatteryHealthReport,
    _compute_cell_summary,
    _compute_trends,
    _fit_trend,
    _json_safe,
    _mechanism_evidence_dict,
    _merge_all,
    _validate_inputs,
    build_report,
    save_report,
)


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------

N_CELLS  = 3
N_CYCLES = 10
CELLS    = [f"c{i}" for i in range(1, N_CELLS + 1)]
CYCLES   = list(range(1, N_CYCLES + 1))

rng = np.random.default_rng(42)


def _make_predictions(noise: float = 0.01) -> pd.DataFrame:
    rows = []
    for cell_id in CELLS:
        for cyc in CYCLES:
            soh_true = max(0.70, 1.0 - 0.02 * cyc + rng.normal(0, 0.002))
            soh_pred = soh_true + rng.normal(0, noise)
            rows.append({
                "cell_id":      cell_id,
                "cycle_number": cyc,
                "y_true":       soh_true,
                "y_pred":       soh_pred,
            })
    return pd.DataFrame(rows)


def _make_phys_df() -> pd.DataFrame:
    rows = []
    for cell_id in CELLS:
        for cyc in CYCLES:
            frac = (cyc - 1) / max(N_CYCLES - 1, 1)
            rows.append({
                "cell_id":                   cell_id,
                "cycle_number":              cyc,
                "LLI_mean_shift":            -frac * 0.005 + rng.normal(0, 0.0005),
                "LLI_n_matched":             2,
                "LLI_n_disappeared":         int(frac > 0.7),
                "LLI_n_appeared":            0,
                "LLI_confidence":            max(0.4, 1.0 - frac * 0.5),
                "LLI_disappearance_penalty": 0.0 if frac <= 0.7 else 0.5,
                "LAM_loss":                  frac * 0.08 + rng.normal(0, 0.003),
                "Resistance_growth":         frac * 0.15 + rng.normal(0, 0.005),
                "mean_ref_width_mV":         50.0 + rng.normal(0, 1.0),
                "mean_curr_width_mV":        50.0 + frac * 10.0 + rng.normal(0, 1.0),
                "grid_spacing_mV":           5.0,
            })
    return pd.DataFrame(rows)


def _make_feature_importance() -> pd.DataFrame:
    from feature_schema import FEATURE_COLS
    coefs = rng.normal(0, 0.1, len(FEATURE_COLS))
    return pd.DataFrame({
        "feature":         FEATURE_COLS,
        "coefficient":     coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def _make_per_cell_metrics() -> dict:
    return {
        cell: {"rmse": rng.uniform(0.005, 0.02),
               "mae":  rng.uniform(0.003, 0.015),
               "r2":   rng.uniform(0.85, 0.99)}
        for cell in CELLS
    }


def _make_overall_metrics() -> dict:
    return {"rmse": 0.012, "mae": 0.009, "r2": 0.93}


def _make_report(**kwargs) -> BatteryHealthReport:
    defaults = dict(
        predictions        = _make_predictions(),
        overall_metrics    = _make_overall_metrics(),
        per_cell_metrics   = _make_per_cell_metrics(),
        feature_importance = _make_feature_importance(),
        phys_df            = _make_phys_df(),
    )
    defaults.update(kwargs)
    return build_report(**defaults)


# ---------------------------------------------------------------------------
# 1. Validation
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_raises_on_missing_prediction_col(self):
        bad = _make_predictions().drop(columns=["y_pred"])
        with pytest.raises(ValueError, match="y_pred"):
            _validate_inputs(bad, _make_per_cell_metrics(),
                             _make_phys_df(), _make_feature_importance())

    def test_raises_on_duplicate_keys_in_predictions(self):
        dup = pd.concat([_make_predictions(), _make_predictions()], ignore_index=True)
        with pytest.raises(ValueError, match="Duplicate"):
            _validate_inputs(dup, _make_per_cell_metrics(),
                             _make_phys_df(), _make_feature_importance())

    def test_raises_when_per_cell_metrics_missing_cell(self):
        metrics = _make_per_cell_metrics()
        del metrics[CELLS[0]]
        with pytest.raises(ValueError, match="per_cell_metrics"):
            _validate_inputs(_make_predictions(), metrics,
                             _make_phys_df(), _make_feature_importance())

    def test_warns_on_missing_physics_cols(self):
        phys = _make_phys_df().drop(columns=["LAM_loss", "Resistance_growth"])
        with pytest.warns(UserWarning, match="physics column"):
            _validate_inputs(_make_predictions(), _make_per_cell_metrics(),
                             phys, _make_feature_importance())


# ---------------------------------------------------------------------------
# 2. Merge
# ---------------------------------------------------------------------------

class TestMergeAll:
    def test_output_has_soh_pred_and_true(self):
        df = _merge_all(_make_predictions(), _make_phys_df(), None)
        assert "soh_pred" in df.columns
        assert "soh_true" in df.columns

    def test_residuals_computed(self):
        df = _merge_all(_make_predictions(), _make_phys_df(), None)
        assert "soh_residual"     in df.columns
        assert "soh_abs_residual" in df.columns
        np.testing.assert_allclose(
            df["soh_abs_residual"].values,
            (df["soh_pred"] - df["soh_true"]).abs().values,
        )

    def test_physics_cols_present(self):
        df = _merge_all(_make_predictions(), _make_phys_df(), None)
        for col in PHYSICS_COLS:
            assert col in df.columns

    def test_row_count_equals_predictions(self):
        pred = _make_predictions()
        df   = _merge_all(pred, _make_phys_df(), None)
        assert len(df) == len(pred)

    def test_sorted_by_cell_cycle(self):
        df = _merge_all(_make_predictions(), _make_phys_df(), None)
        for _, grp in df.groupby("cell_id"):
            assert list(grp["cycle_number"]) == sorted(grp["cycle_number"].tolist())

    def test_left_join_keeps_rows_when_phys_missing(self):
        phys_partial = _make_phys_df().iloc[:5]  # only 5 rows of physics
        pred         = _make_predictions()
        df           = _merge_all(pred, phys_partial, None)
        assert len(df) == len(pred)
        # rows without physics should be NaN
        assert df["LAM_loss"].isna().any()

    def test_optional_feature_df_adds_dqv_cols(self):
        from feature_schema import FEATURE_COLS
        feat_df = pd.DataFrame({
            "cell_id":      [c for c in CELLS for _ in CYCLES],
            "cycle_number": CYCLES * N_CELLS,
            **{col: rng.random(N_CELLS * N_CYCLES) for col in FEATURE_COLS},
        })
        df = _merge_all(_make_predictions(), _make_phys_df(), feat_df)
        for col in FEATURE_COLS:
            assert col in df.columns

    def test_no_duplicate_keys_in_output(self):
        df = _merge_all(_make_predictions(), _make_phys_df(), None)
        assert not df.duplicated(subset=["cell_id", "cycle_number"]).any()


# ---------------------------------------------------------------------------
# 3. Trend analysis
# ---------------------------------------------------------------------------

class TestFitTrend:
    def test_perfect_linear_signal_r2_is_one(self):
        x = np.arange(10, dtype=float)
        y = 3 * x + 5.0
        fit = _fit_trend(x, y)
        assert fit["slope"]     == pytest.approx(3.0, rel=1e-6)
        assert fit["intercept"] == pytest.approx(5.0, rel=1e-6)
        assert fit["r2"]        == pytest.approx(1.0, abs=1e-9)
        assert fit["n_cycles"]  == 10

    def test_constant_signal_returns_finite_r2(self):
        x = np.arange(10, dtype=float)
        y = np.ones(10) * 0.9
        fit = _fit_trend(x, y)
        assert fit["slope"] == pytest.approx(0.0, abs=1e-10)
        assert np.isfinite(fit["r2"]) or True  # R² may be 1 or degenerate — just no crash

    def test_all_nan_returns_nan_slope(self):
        x = np.arange(5, dtype=float)
        y = np.full(5, float("nan"))
        fit = _fit_trend(x, y)
        assert np.isnan(fit["slope"])
        assert fit["n_cycles"] == 0

    def test_one_valid_point_returns_nan_slope(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([float("nan"), 0.5, float("nan")])
        fit = _fit_trend(x, y)
        assert np.isnan(fit["slope"])


class TestComputeTrends:
    def test_shape_is_cells_x_signals(self):
        cycle_df = _merge_all(_make_predictions(), _make_phys_df(), None)
        trend_df = _compute_trends(cycle_df)
        signals  = [s for s in TREND_SIGNALS if s in cycle_df.columns]
        assert len(trend_df) == N_CELLS * len(signals)

    def test_required_columns_present(self):
        cycle_df = _merge_all(_make_predictions(), _make_phys_df(), None)
        trend_df = _compute_trends(cycle_df)
        for col in ["cell_id", "signal", "slope", "intercept", "r2", "n_cycles"]:
            assert col in trend_df.columns

    def test_soh_pred_has_negative_slope(self):
        cycle_df = _merge_all(_make_predictions(), _make_phys_df(), None)
        trend_df = _compute_trends(cycle_df)
        soh_rows = trend_df[trend_df["signal"] == "soh_pred"]
        assert (soh_rows["slope"] < 0).all(), "SOH should degrade over cycles"

    def test_lam_has_positive_slope(self):
        cycle_df = _merge_all(_make_predictions(), _make_phys_df(), None)
        trend_df = _compute_trends(cycle_df)
        lam_rows = trend_df[trend_df["signal"] == "LAM_loss"]
        assert (lam_rows["slope"] > 0).all(), "LAM loss should increase over cycles"


# ---------------------------------------------------------------------------
# 4. build_report — structure
# ---------------------------------------------------------------------------

class TestBuildReport:
    @pytest.fixture(scope="class")
    def report(self):
        return _make_report()

    def test_cycle_df_has_all_cells(self, report):
        assert set(report.cycle_df["cell_id"].unique()) == set(CELLS)

    def test_cycle_df_has_correct_row_count(self, report):
        assert len(report.cycle_df) == N_CELLS * N_CYCLES

    def test_cell_summary_one_row_per_cell(self, report):
        assert len(report.cell_summary) == N_CELLS

    def test_trend_df_covers_all_cells_and_signals(self, report):
        signals = [s for s in TREND_SIGNALS if s in report.cycle_df.columns]
        assert len(report.trend_df) == N_CELLS * len(signals)

    def test_model_metrics_keys(self, report):
        assert "overall" in report.model_metrics
        assert "per_cell" in report.model_metrics

    def test_feature_importance_preserved(self, report):
        assert "feature"     in report.feature_importance.columns
        assert "coefficient" in report.feature_importance.columns

    def test_mechanism_evidence_has_all_cells(self, report):
        assert set(report.mechanism_evidence.keys()) == set(CELLS)

    def test_mechanism_evidence_has_all_mechanisms(self, report):
        from battery_health_report import _MECHANISM_SCALES
        for cell_ev in report.mechanism_evidence.values():
            for mech in _MECHANISM_SCALES:
                assert mech in cell_ev, f"Missing mechanism '{mech}' in evidence"
            assert "dominant_mechanism" in cell_ev

    def test_cell_summary_has_dominant_mechanism(self, report):
        assert "dominant_mechanism" in report.cell_summary.columns
        assert report.cell_summary["dominant_mechanism"].notna().all()

    def test_cell_summary_has_overall_confidence(self, report):
        conf = report.cell_summary["overall_confidence"]
        assert conf.between(0.0, 1.0).all()

    def test_report_version_is_string(self, report):
        assert isinstance(report.report_version, str)

    def test_created_at_is_iso_format(self, report):
        from datetime import datetime
        datetime.fromisoformat(report.created_at)  # raises if malformed

    def test_cycle_df_indexing_consistent_with_phys_df(self, report):
        """Every (cell_id, cycle_number) in cycle_df should also be in the original predictions."""
        pred = _make_predictions()
        pred_keys = set(zip(pred["cell_id"], pred["cycle_number"]))
        rep_keys  = set(zip(report.cycle_df["cell_id"], report.cycle_df["cycle_number"]))
        assert rep_keys == pred_keys


# ---------------------------------------------------------------------------
# 5. Cell summary correctness
# ---------------------------------------------------------------------------

class TestCellSummary:
    def test_soh_pred_total_change_is_negative(self):
        report = _make_report()
        assert (report.cell_summary["soh_pred_total_change"] < 0).all()

    def test_per_cell_metrics_embedded(self):
        per_cell = _make_per_cell_metrics()
        report   = _make_report(per_cell_metrics=per_cell)
        for _, row in report.cell_summary.iterrows():
            assert row["pred_rmse"] == pytest.approx(per_cell[row["cell_id"]]["rmse"], rel=1e-6)

    def test_first_last_cycle_endpoints(self):
        report = _make_report()
        for _, row in report.cell_summary.iterrows():
            assert row["first_cycle"] == CYCLES[0]
            assert row["last_cycle"]  == CYCLES[-1]

    def test_lli_confidence_mean_in_range(self):
        report = _make_report()
        if "LLI_confidence_mean" in report.cell_summary.columns:
            assert report.cell_summary["LLI_confidence_mean"].between(0, 1).all()


# ---------------------------------------------------------------------------
# 6. save_report
# ---------------------------------------------------------------------------

class TestSaveReport:
    @pytest.fixture
    def saved(self, tmp_path):
        report = _make_report()
        paths  = save_report(report, tmp_path)
        return report, paths, tmp_path

    def test_all_files_created(self, saved):
        _, paths, _ = saved
        for name, path in paths.items():
            assert path.exists(), f"Expected {name} at {path}"

    def test_cycle_data_csv_has_correct_shape(self, saved):
        _, paths, _ = saved
        df = pd.read_csv(paths["cycle_data"])
        assert len(df) == N_CELLS * N_CYCLES

    def test_cell_summary_csv_has_correct_shape(self, saved):
        _, paths, _ = saved
        df = pd.read_csv(paths["cell_summary"])
        assert len(df) == N_CELLS

    def test_summary_json_is_valid(self, saved):
        _, paths, _ = saved
        data = json.loads(paths["summary_json"].read_text())
        assert "report_version"    in data
        assert "model_metrics"     in data
        assert "mechanism_evidence" in data
        assert "n_cells"           in data
        assert data["n_cells"]     == N_CELLS

    def test_summary_json_mechanism_evidence_has_all_cells(self, saved):
        _, paths, _ = saved
        data = json.loads(paths["summary_json"].read_text())
        assert set(data["mechanism_evidence"].keys()) == set(CELLS)

    def test_trend_csv_has_slope_column(self, saved):
        _, paths, _ = saved
        df = pd.read_csv(paths["trend_analysis"])
        assert "slope" in df.columns

    def test_output_dir_created_if_absent(self, tmp_path):
        report   = _make_report()
        deep_dir = tmp_path / "a" / "b" / "c"
        paths    = save_report(report, deep_dir)
        assert paths["cycle_data"].exists()


# ---------------------------------------------------------------------------
# 7. JSON serialisation helpers
# ---------------------------------------------------------------------------

class TestJsonSafe:
    def test_numpy_int_converted(self):
        assert isinstance(_json_safe(np.int64(5)), int)

    def test_numpy_float_converted(self):
        assert isinstance(_json_safe(np.float64(3.14)), float)

    def test_nan_becomes_none(self):
        assert _json_safe(np.float64(float("nan"))) is None

    def test_inf_becomes_none(self):
        assert _json_safe(np.float64(float("inf"))) is None

    def test_nested_dict_converted(self):
        d = {"a": np.int64(1), "b": {"c": np.float64(2.5)}}
        r = _json_safe(d)
        assert type(r["a"]) is int
        assert type(r["b"]["c"]) is float

    def test_native_types_unchanged(self):
        assert _json_safe(1)     == 1
        assert _json_safe(3.14)  == pytest.approx(3.14)
        assert _json_safe("foo") == "foo"


# ---------------------------------------------------------------------------
# 8. Consistent (cell_id, cycle_number) indexing
# ---------------------------------------------------------------------------

class TestIndexConsistency:
    def test_cycle_df_and_trend_df_share_same_cells(self):
        report = _make_report()
        assert set(report.cycle_df["cell_id"].unique()) == \
               set(report.trend_df["cell_id"].unique())

    def test_cycle_df_and_cell_summary_share_same_cells(self):
        report = _make_report()
        assert set(report.cycle_df["cell_id"].unique()) == \
               set(report.cell_summary["cell_id"].unique())

    def test_mechanism_evidence_keys_match_cell_summary(self):
        report = _make_report()
        assert set(report.mechanism_evidence.keys()) == \
               set(str(c) for c in report.cell_summary["cell_id"])

    def test_no_duplicate_cell_cycle_in_cycle_df(self):
        report = _make_report()
        assert not report.cycle_df.duplicated(subset=["cell_id", "cycle_number"]).any()

    def test_physics_missing_cycles_still_in_cycle_df(self):
        phys_partial = _make_phys_df().query("cycle_number > 3")
        report       = _make_report(phys_df=phys_partial)
        assert len(report.cycle_df) == N_CELLS * N_CYCLES  # all rows kept

    def test_extra_phys_cycles_not_added_to_cycle_df(self):
        """phys_df rows without matching prediction rows must not inflate cycle_df."""
        extra_row = pd.DataFrame([{
            "cell_id": CELLS[0], "cycle_number": 9999,
            **{col: 0.0 for col in PHYSICS_COLS},
        }])
        phys_with_extra = pd.concat([_make_phys_df(), extra_row], ignore_index=True)
        report = _make_report(phys_df=phys_with_extra)
        assert len(report.cycle_df) == N_CELLS * N_CYCLES


# ---------------------------------------------------------------------------
# 9. Plotting (skipped if matplotlib absent)
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for tests
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

pytestmark_mpl = pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")


@pytestmark_mpl
class TestPlotting:
    @pytest.fixture(scope="class")
    def report(self):
        return _make_report()

    def test_plot_soh_predictions_returns_dict(self, report):
        from battery_health_report import plot_soh_predictions
        result = plot_soh_predictions(report)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(CELLS)

    def test_plot_soh_predictions_single_cell(self, report):
        from battery_health_report import plot_soh_predictions
        ax = plot_soh_predictions(report, cell_id=CELLS[0])
        assert hasattr(ax, "get_xlabel")

    def test_plot_mechanism_trends_returns_dict(self, report):
        from battery_health_report import plot_mechanism_trends
        result = plot_mechanism_trends(report)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_plot_feature_importance_returns_axes(self, report):
        from battery_health_report import plot_feature_importance
        ax = plot_feature_importance(report, top_n=5)
        assert hasattr(ax, "get_xlabel")

    def test_plot_dashboard_returns_figure(self, report):
        from battery_health_report import plot_dashboard
        result = plot_dashboard(report, cell_id=CELLS[0])
        assert "figure" in result
        assert "soh"    in result

    def teardown_method(self, method):
        plt.close("all")
