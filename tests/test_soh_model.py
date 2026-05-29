"""
Tests for SOHModelTrainer and ModelResults in step4_soh_model.py.
Uses a synthetic two-cell dataset so tests run without any real data files.
"""

import json
import numpy as np
import pandas as pd
import pytest

from step4_soh_model import ModelResults, SOHModelTrainer
from feature_schema import FEATURE_COLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _make_df(n_cells: int = 3, cycles_per_cell: int = 20) -> pd.DataFrame:
    """Synthetic feature DataFrame with a linear SOH trend."""
    rows = []
    for cell_idx in range(n_cells):
        for cyc in range(cycles_per_cell):
            soh = 1.0 - 0.003 * cyc + RNG.normal(0, 0.002)
            feats = {col: RNG.normal() for col in FEATURE_COLS}
            rows.append({"cell_id": f"cell_{cell_idx}", "cycle_number": cyc, "soh": soh, **feats})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_df():
    return _make_df()


@pytest.fixture(scope="module")
def trained(synthetic_df, tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    trainer = SOHModelTrainer()
    results = trainer.fit(synthetic_df)
    trainer.save(results, out)
    return trainer, results, out


# ---------------------------------------------------------------------------
# ModelResults structure
# ---------------------------------------------------------------------------

def test_results_predictions_shape(trained, synthetic_df):
    _, results, _ = trained
    assert len(results.predictions) == len(synthetic_df)


def test_results_predictions_columns(trained):
    _, results, _ = trained
    assert set(results.predictions.columns) >= {"cell_id", "cycle_number", "y_true", "y_pred"}


def test_results_overall_metrics_keys(trained):
    _, results, _ = trained
    assert set(results.overall_metrics) == {"rmse", "mae", "r2"}


def test_results_per_cell_keys(trained, synthetic_df):
    _, results, _ = trained
    expected_cells = set(synthetic_df["cell_id"].unique())
    assert set(results.per_cell_metrics) == expected_cells


def test_results_feature_importance_columns(trained):
    _, results, _ = trained
    assert set(results.feature_importance.columns) >= {"feature", "coefficient", "abs_coefficient"}


def test_results_feature_importance_all_features(trained):
    _, results, _ = trained
    assert set(results.feature_importance["feature"]) == set(FEATURE_COLS)


def test_results_feature_importance_sorted(trained):
    _, results, _ = trained
    vals = results.feature_importance["abs_coefficient"].values
    assert list(vals) == sorted(vals, reverse=True)


# ---------------------------------------------------------------------------
# Metrics sanity
# ---------------------------------------------------------------------------

def test_overall_metrics_are_finite(trained):
    _, results, _ = trained
    for v in results.overall_metrics.values():
        assert np.isfinite(v)


def test_rmse_non_negative(trained):
    _, results, _ = trained
    assert results.overall_metrics["rmse"] >= 0


def test_mae_non_negative(trained):
    _, results, _ = trained
    assert results.overall_metrics["mae"] >= 0


def test_per_cell_metrics_all_finite(trained):
    _, results, _ = trained
    for cell, m in results.per_cell_metrics.items():
        for k, v in m.items():
            assert np.isfinite(v), f"Cell {cell} metric {k} is not finite"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_model_file_exists(trained):
    _, results, _ = trained
    assert results.model_path.exists()


def test_schema_file_exists(trained):
    _, results, _ = trained
    assert results.schema_path.exists()


def test_metrics_json_exists(trained):
    _, _, out = trained
    assert (out / "metrics.json").exists()


def test_metrics_json_structure(trained):
    _, _, out = trained
    data = json.loads((out / "metrics.json").read_text())
    assert "overall" in data and "per_cell" in data
    assert set(data["overall"]) == {"rmse", "mae", "r2"}


def test_predictions_csv_exists(trained):
    _, _, out = trained
    assert (out / "predictions.csv").exists()


def test_feature_importance_csv_exists(trained):
    _, _, out = trained
    assert (out / "feature_importance.csv").exists()


def test_saved_model_is_loadable(trained):
    import joblib
    _, results, _ = trained
    pipeline = joblib.load(results.model_path)
    assert hasattr(pipeline, "predict")


# ---------------------------------------------------------------------------
# LOCO correctness — no data leakage
# ---------------------------------------------------------------------------

def test_loco_no_train_cell_in_test(synthetic_df):
    trainer = SOHModelTrainer()
    for train_df, test_df, cell_id in trainer._loco_splits(synthetic_df):
        assert cell_id not in train_df["cell_id"].values


def test_loco_covers_all_cells(synthetic_df):
    trainer = SOHModelTrainer()
    seen = {cell_id for _, _, cell_id in trainer._loco_splits(synthetic_df)}
    assert seen == set(synthetic_df["cell_id"].unique())


# ---------------------------------------------------------------------------
# load_features validation
# ---------------------------------------------------------------------------

def test_load_features_missing_column_raises(tmp_path, synthetic_df):
    broken = synthetic_df.drop(columns=["soh"])
    p = tmp_path / "bad.csv"
    broken.to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing"):
        SOHModelTrainer().load_features(p)


def test_load_features_applies_column_aliases(tmp_path, synthetic_df):
    aliased = synthetic_df.rename(columns={"dqv_variance": "var_deltaQ"})
    p = tmp_path / "aliased.csv"
    aliased.to_csv(p, index=False)
    df = SOHModelTrainer().load_features(p)
    assert "dqv_variance" in df.columns


# ---------------------------------------------------------------------------
# save() guard
# ---------------------------------------------------------------------------

def test_save_before_fit_raises():
    trainer = SOHModelTrainer()
    dummy = ModelResults(
        predictions=pd.DataFrame(),
        overall_metrics={},
        per_cell_metrics={},
        feature_importance=pd.DataFrame(),
    )
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        trainer.save(dummy, "/tmp/unused")
