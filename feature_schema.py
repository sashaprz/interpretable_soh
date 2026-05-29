"""
feature_schema.py
Single source of truth for the delta Q(V) feature pipeline contract.

Import FEATURE_COLS, validate_feature_columns, and save_schema_json in both
step3 and step4.  Changing a column name here is the only change required to
keep the pipeline in sync.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

SCHEMA_VERSION = "1.0"

# Canonical feature column names produced by step3 and consumed by step4.
FEATURE_COLS: List[str] = [
    "dqv_variance",
    "dqv_log_variance",
    "dqv_skewness",
    "dqv_kurtosis",
    "dqv_integral_abs",
    "dqv_max_deviation",
    "dqv_min",
    "dqv_max",
    "dqv_mean",
    "dqv_rms",
]

# Non-feature columns always present in the feature DataFrame.
METADATA_COLS: List[str] = [
    "cell_id",
    "cycle_number",
    "soh",
    "temperature",
    "protocol",
    "ref_cycle_index",
    "config_hash",
    "extraction_ok",
]

# Legacy name → canonical name.  Apply with df.rename(columns=COLUMN_ALIASES).
COLUMN_ALIASES: Dict[str, str] = {
    "log_var_deltaQ":      "dqv_log_variance",
    "var_deltaQ":          "dqv_variance",
    "skew_deltaQ":         "dqv_skewness",
    "kurtosis_deltaQ":     "dqv_kurtosis",
    "abs_integral_deltaQ": "dqv_integral_abs",
    "max_dev_deltaQ":      "dqv_max_deviation",
    "cycle_index":         "cycle_number",
}


def validate_feature_columns(
    df,
    extra_cols: Optional[List[str]] = None,
) -> None:
    """Raise ValueError listing every missing required column.

    Checks FEATURE_COLS + ['cell_id', 'cycle_number', 'soh'] plus any
    caller-supplied extras.  Always call this before model training/inference.
    """
    required = set(FEATURE_COLS) | {"cell_id", "cycle_number", "soh"}
    if extra_cols:
        required |= set(extra_cols)

    missing = sorted(required - set(df.columns))
    if not missing:
        return

    alias_hits = [c for c in df.columns if c in COLUMN_ALIASES]
    hint = ""
    if alias_hits:
        canonical = [COLUMN_ALIASES[c] for c in alias_hits]
        hint = (
            f"  Hint: legacy column(s) {alias_hits} detected "
            f"(canonical: {canonical}); "
            "call df.rename(columns=COLUMN_ALIASES) to fix."
        )
    raise ValueError(
        f"Feature DataFrame is missing {len(missing)} required column(s): "
        f"{missing}.\n{hint}".rstrip()
    )


def schema_as_dict() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_cols": FEATURE_COLS,
        "metadata_cols": METADATA_COLS,
        "column_aliases": COLUMN_ALIASES,
    }


def save_schema_json(path) -> Path:
    """Write feature schema JSON alongside model artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema_as_dict(), indent=2))
    return path
