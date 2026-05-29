"""
test_ica_adapter.py
Unit tests for ica_curve_adapter.
Run with: pytest test_ica_adapter.py -v
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ica_curve_adapter import (
    ALL_ICA_COLS,
    ICACurve,
    _ARRAY_COLS,
    _SCALAR_COLS,
    dataframe_to_ica_curves,
    ica_curve_to_dataframe,
    load_ica_parquet,
    load_ica_pickle,
    save_ica_parquet,
    save_ica_pickle,
    validate_ica_dataframe,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 50  # grid length used throughout tests


def _grid(n: int = N) -> np.ndarray:
    return np.linspace(2.5, 4.2, n)


def _dqdv(n: int = N, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(n)


def _curve(cell_id="c1", cycle_number=1, *, n=N, seed=0, **kw) -> ICACurve:
    return ICACurve(
        cell_id=cell_id,
        cycle_number=cycle_number,
        voltage_grid=_grid(n),
        dqdv=_dqdv(n, seed=seed),
        **kw,
    )


def _basic_df(n_cycles: int = 3) -> pd.DataFrame:
    curves = [_curve(cycle_number=i, seed=i) for i in range(1, n_cycles + 1)]
    return ica_curve_to_dataframe(curves)


# ---------------------------------------------------------------------------
# 1. ICACurve dataclass
# ---------------------------------------------------------------------------

class TestICACurve:
    def test_arrays_coerced_to_float64(self):
        c = ICACurve("c1", 1, np.ones(5, dtype=np.float32), np.zeros(5, dtype=np.int32))
        assert c.voltage_grid.dtype == np.float64
        assert c.dqdv.dtype == np.float64

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="length"):
            ICACurve("c1", 1, np.ones(5), np.ones(6))

    def test_rejects_empty_arrays(self):
        with pytest.raises(ValueError, match="empty"):
            ICACurve("c1", 1, np.array([]), np.array([]))

    def test_rejects_2d_arrays(self):
        with pytest.raises(ValueError, match="1-D"):
            ICACurve("c1", 1, np.ones((5, 2)), np.ones((5, 2)))

    def test_optional_fields_default_to_none(self):
        c = _curve()
        assert c.capacity_ah is None
        assert c.ref_cycle_number is None
        assert c.temperature is None
        assert c.protocol is None

    def test_is_reference_defaults_to_false(self):
        assert _curve().is_reference is False


# ---------------------------------------------------------------------------
# 2. ica_curve_to_dataframe
# ---------------------------------------------------------------------------

class TestIcaCurveToDataframe:
    def test_one_row_per_cycle(self):
        df = _basic_df(4)
        assert len(df) == 4

    def test_all_columns_present(self):
        df = _basic_df()
        for col in ALL_ICA_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_voltage_grid_is_float64(self):
        df = _basic_df()
        for arr in df["voltage_grid"]:
            assert arr.dtype == np.float64

    def test_dqdv_is_float64(self):
        df = _basic_df()
        for arr in df["dqdv"]:
            assert arr.dtype == np.float64

    def test_metadata_columns_correct_types(self):
        df = ica_curve_to_dataframe([
            _curve(capacity_ah=2.5, is_reference=True,
                   ref_cycle_number=1, temperature=25.0, protocol="C/5"),
        ])
        assert df["capacity_ah"].dtype == np.float64
        assert df["temperature"].dtype == np.float64
        assert df["is_reference"].dtype == bool

    def test_ref_cycle_number_nullable_int(self):
        curves = [
            _curve("c1", 1, ref_cycle_number=1),
            _curve("c1", 2, ref_cycle_number=None),
        ]
        df = ica_curve_to_dataframe(curves)
        assert df["ref_cycle_number"].dtype.name == "Int64"
        assert pd.isna(df.loc[1, "ref_cycle_number"])

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            ica_curve_to_dataframe([])

    def test_rejects_mismatched_grid_lengths(self):
        curves = [_curve(n=50), _curve(cycle_number=2, n=60)]
        with pytest.raises(ValueError, match="same length"):
            ica_curve_to_dataframe(curves)

    def test_reference_cycle_linkage_preserved(self):
        curves = [
            _curve("c1", 1, is_reference=True,  ref_cycle_number=1),
            _curve("c1", 2, is_reference=False, ref_cycle_number=1),
        ]
        df = ica_curve_to_dataframe(curves)
        assert bool(df.loc[0, "is_reference"]) is True
        assert int(df.loc[1, "ref_cycle_number"]) == 1

    def test_multiple_cells(self):
        curves = [_curve("c1", 1), _curve("c2", 1, seed=5)]
        df = ica_curve_to_dataframe(curves)
        assert set(df["cell_id"]) == {"c1", "c2"}

    def test_arrays_are_copies(self):
        c = _curve()
        df = ica_curve_to_dataframe([c])
        df["dqdv"].iloc[0][0] = 999.0
        assert c.dqdv[0] != 999.0


# ---------------------------------------------------------------------------
# 3. dataframe_to_ica_curves (reverse adapter)
# ---------------------------------------------------------------------------

class TestDataframeToIcaCurves:
    def test_round_trip_count(self):
        original = [_curve("c1", i, seed=i) for i in range(1, 5)]
        df     = ica_curve_to_dataframe(original)
        result = dataframe_to_ica_curves(df)
        assert len(result) == len(original)

    def test_round_trip_array_values(self):
        c   = _curve(seed=7)
        df  = ica_curve_to_dataframe([c])
        out = dataframe_to_ica_curves(df)[0]
        np.testing.assert_array_equal(c.voltage_grid, out.voltage_grid)
        np.testing.assert_array_equal(c.dqdv, out.dqdv)

    def test_round_trip_scalar_metadata(self):
        c   = _curve(capacity_ah=3.1, is_reference=True, ref_cycle_number=1,
                     temperature=30.0, protocol="C/10")
        df  = ica_curve_to_dataframe([c])
        out = dataframe_to_ica_curves(df)[0]
        assert out.capacity_ah == pytest.approx(3.1)
        assert out.is_reference is True
        assert out.ref_cycle_number == 1
        assert out.temperature == pytest.approx(30.0)
        assert out.protocol == "C/10"

    def test_none_optional_fields_survive_round_trip(self):
        c   = _curve()  # all optionals are None
        df  = ica_curve_to_dataframe([c])
        out = dataframe_to_ica_curves(df)[0]
        assert out.capacity_ah is None
        assert out.ref_cycle_number is None
        assert out.temperature is None
        assert out.protocol is None

    def test_arrays_are_float64_after_round_trip(self):
        df  = _basic_df()
        out = dataframe_to_ica_curves(df)
        for c in out:
            assert c.voltage_grid.dtype == np.float64
            assert c.dqdv.dtype == np.float64


# ---------------------------------------------------------------------------
# 4. validate_ica_dataframe
# ---------------------------------------------------------------------------

class TestValidateIcaDataframe:
    def test_passes_on_valid_df(self):
        validate_ica_dataframe(_basic_df())  # no raise

    def test_raises_on_missing_columns(self):
        df = _basic_df().drop(columns=["dqdv"])
        with pytest.raises(ValueError, match="dqdv"):
            validate_ica_dataframe(df)

    def test_raises_on_duplicate_cell_cycle(self):
        df = pd.concat([_basic_df(2), _basic_df(2)], ignore_index=True)
        with pytest.raises(ValueError, match="Duplicate"):
            validate_ica_dataframe(df)

    def test_raises_on_non_uniform_grid_lengths(self):
        df       = _basic_df(2)
        bad_arr  = np.ones(N + 5)
        df.at[1, "voltage_grid"] = bad_arr
        df.at[1, "dqdv"]         = bad_arr
        with pytest.raises(ValueError, match="uniform"):
            validate_ica_dataframe(df)

    def test_raises_on_dqdv_length_mismatch(self):
        df = _basic_df(2)
        df.at[0, "dqdv"] = np.ones(N + 1)
        with pytest.raises(ValueError, match="dqdv length"):
            validate_ica_dataframe(df)

    def test_raises_on_nan_in_arrays(self):
        df = _basic_df(1)
        arr = df.at[0, "dqdv"].copy()
        arr[5] = np.nan
        df.at[0, "dqdv"] = arr
        with pytest.raises(ValueError, match="NaN"):
            validate_ica_dataframe(df)

    def test_raises_on_inf_in_arrays(self):
        df = _basic_df(1)
        arr = df.at[0, "voltage_grid"].copy()
        arr[0] = np.inf
        df.at[0, "voltage_grid"] = arr
        with pytest.raises(ValueError, match="NaN"):
            validate_ica_dataframe(df)


# ---------------------------------------------------------------------------
# 5. Pickle serialization
# ---------------------------------------------------------------------------

class TestPickleSerialization:
    def test_round_trip(self, tmp_path):
        df  = _basic_df(3)
        p   = save_ica_pickle(df, tmp_path / "ica.pkl")
        df2 = load_ica_pickle(p)
        assert df.shape == df2.shape
        np.testing.assert_array_equal(df["dqdv"].iloc[0], df2["dqdv"].iloc[0])

    def test_creates_parent_dirs(self, tmp_path):
        df = _basic_df()
        p  = save_ica_pickle(df, tmp_path / "a" / "b" / "ica.pkl")
        assert p.exists()

    def test_validates_on_save(self, tmp_path):
        bad = _basic_df().drop(columns=["cell_id"])
        with pytest.raises(ValueError):
            save_ica_pickle(bad, tmp_path / "bad.pkl")

    def test_validates_on_load(self, tmp_path):
        # Write a corrupt pickle directly (bypassing save_ica_pickle validation)
        bad = _basic_df().drop(columns=["cell_id"])
        p   = tmp_path / "bad.pkl"
        with open(p, "wb") as fh:
            pickle.dump(bad, fh)
        with pytest.raises(ValueError):
            load_ica_pickle(p)


# ---------------------------------------------------------------------------
# 6. Parquet serialization
# ---------------------------------------------------------------------------

class TestParquetSerialization:
    def test_round_trip_shape(self, tmp_path):
        df  = _basic_df(3)
        p   = save_ica_parquet(df, tmp_path / "ica.parquet")
        df2 = load_ica_parquet(p)
        assert df.shape == df2.shape

    def test_round_trip_array_values(self, tmp_path):
        df  = _basic_df(2)
        p   = save_ica_parquet(df, tmp_path / "ica.parquet")
        df2 = load_ica_parquet(p)
        for i in range(len(df)):
            np.testing.assert_array_almost_equal(
                df["voltage_grid"].iloc[i], df2["voltage_grid"].iloc[i]
            )
            np.testing.assert_array_almost_equal(
                df["dqdv"].iloc[i], df2["dqdv"].iloc[i]
            )

    def test_round_trip_scalar_metadata(self, tmp_path):
        curves = [_curve(capacity_ah=2.5, temperature=25.0, protocol="C/5")]
        df  = ica_curve_to_dataframe(curves)
        p   = save_ica_parquet(df, tmp_path / "ica.parquet")
        df2 = load_ica_parquet(p)
        assert df2["capacity_ah"].iloc[0] == pytest.approx(2.5)
        assert df2["temperature"].iloc[0] == pytest.approx(25.0)
        assert df2["protocol"].iloc[0] == "C/5"

    def test_no_byte_blob_columns_after_load(self, tmp_path):
        df  = _basic_df()
        p   = save_ica_parquet(df, tmp_path / "ica.parquet")
        df2 = load_ica_parquet(p)
        assert "_voltage_grid_bytes" not in df2.columns
        assert "_dqdv_bytes"         not in df2.columns

    def test_arrays_are_float64_after_load(self, tmp_path):
        df  = _basic_df()
        p   = save_ica_parquet(df, tmp_path / "ica.parquet")
        df2 = load_ica_parquet(p)
        assert df2["voltage_grid"].iloc[0].dtype == np.float64
        assert df2["dqdv"].iloc[0].dtype          == np.float64

    def test_full_round_trip_via_ica_curves(self, tmp_path):
        """Parquet → DataFrame → ICACurves must recover original arrays."""
        original = [
            _curve("c1", i, seed=i, capacity_ah=float(i),
                   is_reference=(i == 1), ref_cycle_number=1)
            for i in range(1, 4)
        ]
        df_in  = ica_curve_to_dataframe(original)
        p      = save_ica_parquet(df_in, tmp_path / "ica.parquet")
        df_out = load_ica_parquet(p)
        curves = dataframe_to_ica_curves(df_out)

        for orig, loaded in zip(original, curves):
            np.testing.assert_array_equal(orig.voltage_grid, loaded.voltage_grid)
            np.testing.assert_array_equal(orig.dqdv, loaded.dqdv)
            assert orig.is_reference     == loaded.is_reference
            assert orig.ref_cycle_number == loaded.ref_cycle_number
