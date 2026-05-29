"""
test_feature_alignment.py
Unit tests verifying exact column alignment between step3 and step4.
Run with: pytest test_feature_alignment.py -v
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_schema import (
    COLUMN_ALIASES,
    FEATURE_COLS,
    METADATA_COLS,
    SCHEMA_VERSION,
    save_schema_json,
    schema_as_dict,
    validate_feature_columns,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STEP3_PATH = Path(__file__).parent / "step3_deltaQ(V)_feature_extraction.py"


def _import_step3():
    import sys
    spec = importlib.util.spec_from_file_location("step3", _STEP3_PATH)
    mod  = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve its own module namespace.
    sys.modules["step3"] = mod
    spec.loader.exec_module(mod)
    return mod


def _full_valid_df(**overrides) -> pd.DataFrame:
    cols = FEATURE_COLS + ["cell_id", "cycle_number", "soh"]
    data = {c: [0.0] for c in cols}
    data["cell_id"]      = ["cell_01"]
    data["cycle_number"] = [1]
    data.update(overrides)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 1. Schema self-consistency
# ---------------------------------------------------------------------------

class TestSchemaInternals:
    def test_feature_cols_are_unique(self):
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))

    def test_metadata_cols_are_unique(self):
        assert len(METADATA_COLS) == len(set(METADATA_COLS))

    def test_no_overlap_between_feature_and_metadata(self):
        overlap = set(FEATURE_COLS) & set(METADATA_COLS)
        assert not overlap, f"Columns appear in both lists: {overlap}"

    def test_schema_version_is_string(self):
        assert isinstance(SCHEMA_VERSION, str) and SCHEMA_VERSION

    def test_schema_dict_keys(self):
        d = schema_as_dict()
        assert set(d.keys()) == {"schema_version", "feature_cols",
                                 "metadata_cols", "column_aliases"}

    def test_schema_dict_feature_cols_match(self):
        assert schema_as_dict()["feature_cols"] == FEATURE_COLS


# ---------------------------------------------------------------------------
# 2. Step3 / schema alignment
# ---------------------------------------------------------------------------

class TestStep3SchemaAlignment:
    def test_core_features_keys_exactly_match_schema(self):
        """_CORE_FEATURES in step3 must be identical (order included) to FEATURE_COLS."""
        step3 = _import_step3()
        assert list(step3._CORE_FEATURES.keys()) == FEATURE_COLS, (
            f"step3._CORE_FEATURES keys:\n  {list(step3._CORE_FEATURES.keys())}\n"
            f"schema FEATURE_COLS:\n  {FEATURE_COLS}"
        )

    def test_step3_feature_extractor_names_match_schema(self):
        step3 = _import_step3()
        cfg   = step3.FeatureConfig(chemistry="generic")
        ext   = step3.FeatureExtractor(cfg)
        assert ext.feature_names == FEATURE_COLS

    def test_feature_matrix_builder_columns_include_schema_features(self):
        step3 = _import_step3()
        rng   = np.random.default_rng(0)
        cfg   = step3.FeatureConfig(v_min=2.5, v_max=4.2, dv=0.005,
                                    chemistry="generic")

        def _half(n=400):
            t = np.linspace(0.0, 3600.0, n)
            v = np.linspace(4.15, 2.55, n) + rng.normal(0, 0.003, n)
            i = np.full(n, -1.0) + rng.normal(0, 0.005, n)
            return pd.DataFrame({"time": t, "voltage": v, "current": i})

        records = [
            step3.CycleRecord("c1", cyc, _half(), soh=1.0 - 0.02 * cyc)
            for cyc in range(1, 6)
        ]
        builder = step3.FeatureMatrixBuilder(cfg, nominal_capacity_ah=10.0)
        feat_df = builder.build(records)

        for col in FEATURE_COLS:
            assert col in feat_df.columns, f"Column {col!r} missing from builder output"

    def test_feature_matrix_uses_cycle_number_not_cycle_index(self):
        step3 = _import_step3()
        rng   = np.random.default_rng(1)
        cfg   = step3.FeatureConfig()

        def _half(n=400):
            t = np.linspace(0.0, 3600.0, n)
            v = np.linspace(4.15, 2.55, n) + rng.normal(0, 0.003, n)
            i = np.full(n, -1.0)
            return pd.DataFrame({"time": t, "voltage": v, "current": i})

        records = [step3.CycleRecord("c1", c, _half(), soh=0.9) for c in range(1, 4)]
        builder = step3.FeatureMatrixBuilder(cfg, nominal_capacity_ah=10.0)
        feat_df = builder.build(records)

        assert "cycle_number" in feat_df.columns
        assert "cycle_index" not in feat_df.columns


# ---------------------------------------------------------------------------
# 3. validate_feature_columns
# ---------------------------------------------------------------------------

class TestValidateFeatureColumns:
    def test_passes_on_complete_df(self):
        validate_feature_columns(_full_valid_df())  # must not raise

    def test_raises_on_empty_df(self):
        with pytest.raises(ValueError, match="missing.*required column"):
            validate_feature_columns(pd.DataFrame())

    def test_raises_listing_each_missing_column(self):
        df = pd.DataFrame(columns=["cell_id", "cycle_number", "soh"])
        with pytest.raises(ValueError) as exc_info:
            validate_feature_columns(df)
        msg = str(exc_info.value)
        for col in FEATURE_COLS:
            assert col in msg

    def test_raises_on_missing_soh(self):
        df = _full_valid_df()
        df = df.drop(columns=["soh"])
        with pytest.raises(ValueError, match="soh"):
            validate_feature_columns(df)

    def test_raises_on_missing_cell_id(self):
        df = _full_valid_df()
        df = df.drop(columns=["cell_id"])
        with pytest.raises(ValueError, match="cell_id"):
            validate_feature_columns(df)

    def test_hint_shown_when_legacy_columns_present(self):
        legacy_cols = list(COLUMN_ALIASES.keys()) + ["cell_id", "soh"]
        df = pd.DataFrame(columns=legacy_cols)
        with pytest.raises(ValueError, match="legacy column"):
            validate_feature_columns(df)

    def test_extra_cols_checked(self):
        df = _full_valid_df()
        with pytest.raises(ValueError, match="extraction_ok"):
            validate_feature_columns(df, extra_cols=["extraction_ok"])

    def test_extra_cols_pass_when_present(self):
        df = _full_valid_df(extraction_ok=[True])
        validate_feature_columns(df, extra_cols=["extraction_ok"])  # no raise


# ---------------------------------------------------------------------------
# 4. Column aliases
# ---------------------------------------------------------------------------

class TestColumnAliases:
    def test_all_old_step4_names_in_aliases(self):
        old_names = [
            "log_var_deltaQ", "var_deltaQ", "skew_deltaQ",
            "kurtosis_deltaQ", "abs_integral_deltaQ", "max_dev_deltaQ",
        ]
        for name in old_names:
            assert name in COLUMN_ALIASES, f"{name!r} not in COLUMN_ALIASES"

    def test_cycle_index_alias_present(self):
        assert "cycle_index" in COLUMN_ALIASES
        assert COLUMN_ALIASES["cycle_index"] == "cycle_number"

    def test_alias_targets_are_canonical(self):
        all_canonical = set(FEATURE_COLS) | set(METADATA_COLS)
        for old, new in COLUMN_ALIASES.items():
            assert new in all_canonical, (
                f"Alias {old!r} → {new!r} but {new!r} is not in FEATURE_COLS or METADATA_COLS"
            )

    def test_rename_with_aliases_produces_valid_df(self):
        all_canonical = set(FEATURE_COLS) | {"cell_id", "cycle_number", "soh"}
        old_cols = list(COLUMN_ALIASES.keys())
        # Build a df that has both old aliases and the canonical cols not covered by aliases
        canonical_only = sorted(all_canonical - set(COLUMN_ALIASES.values()))
        df = pd.DataFrame(columns=old_cols + canonical_only)
        df = df.rename(columns=COLUMN_ALIASES)
        validate_feature_columns(df)  # must not raise


# ---------------------------------------------------------------------------
# 5. Schema JSON persistence
# ---------------------------------------------------------------------------

class TestSchemaJson:
    def test_save_creates_file(self, tmp_path):
        p = save_schema_json(tmp_path / "schema.json")
        assert p.exists()

    def test_json_round_trip(self, tmp_path):
        p = save_schema_json(tmp_path / "schema.json")
        loaded = json.loads(p.read_text())
        assert loaded["schema_version"] == SCHEMA_VERSION
        assert loaded["feature_cols"]   == FEATURE_COLS
        assert loaded["metadata_cols"]  == METADATA_COLS
        assert loaded["column_aliases"] == COLUMN_ALIASES

    def test_creates_parent_dirs(self, tmp_path):
        p = save_schema_json(tmp_path / "a" / "b" / "schema.json")
        assert p.exists()
