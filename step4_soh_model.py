"""
Step 4: SOH Prediction from ICA-derived ΔQ(V) features

Model:      ElasticNet Regression
Validation: Leave-One-Cell-Out Cross Validation (no leakage)

Usage:
    python step4_soh_model.py --features-path features.csv --output-dir models/
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_schema import (
    COLUMN_ALIASES,
    FEATURE_COLS,
    save_schema_json,
    validate_feature_columns,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ModelResults:
    """Structured output of SOHModelTrainer.fit() / .run()."""

    predictions: pd.DataFrame
    """cell_id, cycle_number, y_true, y_pred — one row per held-out observation."""

    overall_metrics: dict
    """{'rmse': float, 'mae': float, 'r2': float} aggregated across all LOCO folds."""

    per_cell_metrics: dict
    """{'<cell_id>': {'rmse': float, 'mae': float, 'r2': float}, ...}"""

    feature_importance: pd.DataFrame
    """feature, coefficient, abs_coefficient — sorted by abs_coefficient desc."""

    model_path: Path | None = field(default=None, repr=False)
    """Path to the serialized final-model joblib artifact, set after save()."""

    schema_path: Path | None = field(default=None, repr=False)
    """Path to the feature_schema.json artifact, set after save()."""


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SOHModelTrainer:
    """Trains an ElasticNet SOH model using Leave-One-Cell-Out CV.

    Typical usage::

        trainer = SOHModelTrainer()
        results = trainer.run("features.csv", "models/")
        print(results.overall_metrics)
    """

    def __init__(
        self,
        alpha: float = 1e-3,
        l1_ratio: float = 0.5,
        max_iter: int = 20_000,
        random_state: int = 42,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state
        self._final_pipeline: Pipeline | None = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_features(self, path: str | Path) -> pd.DataFrame:
        """Read the step3 CSV, apply column aliases, and validate schema."""
        df = pd.read_csv(Path(path))
        df = df.rename(columns=COLUMN_ALIASES)
        validate_feature_columns(df)
        return df

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )),
        ])

    def _loco_splits(
        self, df: pd.DataFrame
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, str]]:
        """Yield (train_df, test_df, cell_id) for each unique cell."""
        for cell in np.unique(df["cell_id"].values):
            yield df[df["cell_id"] != cell], df[df["cell_id"] == cell], str(cell)

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae":  float(mean_absolute_error(y_true, y_pred)),
            "r2":   float(r2_score(y_true, y_pred)),
        }

    def _extract_feature_importance(self, pipeline: Pipeline) -> pd.DataFrame:
        coefs = pipeline.named_steps["enet"].coef_
        return (
            pd.DataFrame({
                "feature":         FEATURE_COLS,
                "coefficient":     coefs,
                "abs_coefficient": np.abs(coefs),
            })
            .sort_values("abs_coefficient", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> ModelResults:
        """Run LOCO CV over *df* and return a :class:`ModelResults`.

        A second pass trains a final model on all data for serialization
        and feature importance; LOCO predictions are used for all metrics.
        """
        fold_dfs: list[pd.DataFrame] = []
        per_cell_metrics: dict = {}

        for train_df, test_df, cell_id in self._loco_splits(df):
            X_train = train_df[FEATURE_COLS].values
            y_train = train_df["soh"].values
            X_test  = test_df[FEATURE_COLS].values
            y_test  = test_df["soh"].values

            pipeline = self._build_pipeline()
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            per_cell_metrics[cell_id] = self._compute_metrics(y_test, y_pred)

            fold_df = test_df[["cell_id", "cycle_number"]].copy()
            fold_df["y_true"] = y_test
            fold_df["y_pred"] = y_pred
            fold_dfs.append(fold_df)

        predictions = pd.concat(fold_dfs, ignore_index=True)
        overall_metrics = self._compute_metrics(
            predictions["y_true"].values,
            predictions["y_pred"].values,
        )

        # Final model on all data — serialized to disk and used for feature importance.
        # LOCO predictions above are the only source of reported metrics.
        self._final_pipeline = self._build_pipeline()
        self._final_pipeline.fit(df[FEATURE_COLS].values, df["soh"].values)

        return ModelResults(
            predictions=predictions,
            overall_metrics=overall_metrics,
            per_cell_metrics=per_cell_metrics,
            feature_importance=self._extract_feature_importance(self._final_pipeline),
        )

    def save(self, results: ModelResults, output_dir: str | Path) -> ModelResults:
        """Persist all artifacts to *output_dir*; mutates and returns *results*.

        Written files:
            elasticnet_soh.joblib   — trained pipeline (StandardScaler + ElasticNet)
            feature_schema.json     — column contract from feature_schema.py
            metrics.json            — overall + per-cell RMSE / MAE / R²
            predictions.csv         — LOCO held-out predictions
            feature_importance.csv  — ElasticNet coefficients
        """
        if self._final_pipeline is None:
            raise RuntimeError("Call fit() before save().")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        model_path = out / "elasticnet_soh.joblib"
        joblib.dump(self._final_pipeline, model_path)

        schema_path = save_schema_json(out / "feature_schema.json")

        metrics = {
            "overall":  results.overall_metrics,
            "per_cell": results.per_cell_metrics,
        }
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

        results.predictions.to_csv(out / "predictions.csv", index=False)
        results.feature_importance.to_csv(out / "feature_importance.csv", index=False)

        results.model_path = model_path
        results.schema_path = schema_path
        return results

    def run(self, features_path: str | Path, output_dir: str | Path) -> ModelResults:
        """Load → fit → save in one call.  Equivalent to the three-step API."""
        df = self.load_features(features_path)
        results = self.fit(df)
        return self.save(results, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ElasticNet SOH model with Leave-One-Cell-Out CV",
    )
    p.add_argument(
        "--features-path", required=True, metavar="PATH",
        help="CSV produced by step3 FeatureMatrixBuilder",
    )
    p.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Directory for model artifacts (created if absent)",
    )
    p.add_argument(
        "--alpha", type=float, default=1e-3,
        help="ElasticNet regularisation strength (default: 1e-3)",
    )
    p.add_argument(
        "--l1-ratio", type=float, default=0.5,
        help="ElasticNet L1/L2 mix: 0=Ridge, 1=Lasso (default: 0.5)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trainer = SOHModelTrainer(alpha=args.alpha, l1_ratio=args.l1_ratio)
    results = trainer.run(args.features_path, args.output_dir)

    m = results.overall_metrics
    print(f"\nLOCO CV  |  RMSE {m['rmse']:.5f}  MAE {m['mae']:.5f}  R² {m['r2']:.5f}")
    print(f"Model    →  {results.model_path}")
    print(f"Metrics  →  {results.model_path.parent / 'metrics.json'}")
    print(f"\nTop features (by |coeff|):")
    print(results.feature_importance.head(5).to_string(index=False))
