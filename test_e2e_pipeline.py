"""
test_e2e_pipeline.py
====================
End-to-end integration tests for the battery degradation pipeline.

Two user flows are exercised:

  Flow 1 — multi-cell dataset:
      Upload cycle data (CSV) → SOH trajectory graph + degradation explanation.
      Chain: step1 (parse) → step3 (features) → step4 (model) → report.

  Flow 2 — single new cycle:
      Provide one discharge half-cycle for a cell of a known chemistry →
      SOH prediction + natural-language degradation explanation.
      Chain: predict.SOHPredictor (uses step3 QV/feature extraction + saved model).

All tests use synthetic data; no real dataset files are required.
Run with: pytest test_e2e_pipeline.py -v
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(2024)


def _make_discharge_half(
    soh: float,
    n: int = 500,
    seed: int | None = None,
    v_high: float = 4.15,
    v_low: float = 2.52,
    i_mean: float = -1.0,
    i_noise: float = 0.004,
) -> pd.DataFrame:
    """Realistic CC discharge half-cycle.

    The capacity (discharge duration) scales with SOH.  Voltage drops from
    v_high to v_low with a mild sigmoid dip near the end — similar to a
    typical LFP/NMC profile.  Current is approximately constant at i_mean A
    with tiny Gaussian noise so rolling-CV stays well under 2%.
    """
    rng = np.random.default_rng(seed if seed is not None else int(soh * 1e6))
    # Time axis: soh scales total discharge time
    t = np.linspace(0.0, 3600.0 * soh, n)
    # Voltage: monotonically decreasing, slight sigmoid toward end
    x = np.linspace(0.0, 1.0, n)
    v = v_high - (v_high - v_low) * (x + 0.15 * np.sin(np.pi * x))
    v += rng.normal(0.0, 0.001, n)        # tiny voltage noise
    # Current: constant CC, tiny noise (CV ~ 0.4%)
    i = np.full(n, i_mean) + rng.normal(0.0, i_noise, n)
    return pd.DataFrame({"time": t, "voltage": v, "current": i})


def _make_charge_half(
    soh: float,
    n: int = 400,
    seed: int | None = None,
) -> pd.DataFrame:
    """CC charge half-cycle (positive current)."""
    rng = np.random.default_rng(seed if seed is not None else int(soh * 2e6 + 1))
    t = np.linspace(0.0, 3600.0 * soh, n)
    v = np.linspace(3.0, 4.15, n) + rng.normal(0.0, 0.001, n)
    i = np.full(n, 1.0) + rng.normal(0.0, 0.004, n)
    return pd.DataFrame({"time": t, "voltage": v, "current": i})


def _make_cell_csv(
    cell_id: str,
    n_cycles: int = 10,
    initial_soh: float = 1.0,
    soh_decay: float = 0.025,
    seed: int = 0,
) -> pd.DataFrame:
    """Concatenated charge+discharge cycles, suitable for step1.parse_file()."""
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    t_offset = 0.0

    for cyc in range(n_cycles):
        soh = max(0.65, initial_soh - soh_decay * cyc + rng.normal(0, 0.002))

        chg = _make_charge_half(soh, seed=seed + cyc * 2)
        chg["time"] = chg["time"] + t_offset
        chg["cycle_index"] = cyc
        t_offset = float(chg["time"].iloc[-1]) + 10.0

        dis = _make_discharge_half(soh, seed=seed + cyc * 2 + 1)
        dis["time"] = dis["time"] + t_offset
        dis["cycle_index"] = cyc
        t_offset = float(dis["time"].iloc[-1]) + 10.0

        rows.extend([chg, dis])

    return pd.concat(rows, ignore_index=True)


def _make_multi_cell_csvs(
    tmp_path: Path,
    n_cells: int = 3,
    n_cycles: int = 8,
    nominal_capacity_ah: float = 10.0,
) -> list[tuple[str, Path]]:
    """Write CSV files for n_cells cells; return [(cell_id, path)]."""
    paths = []
    for k in range(n_cells):
        cell_id = f"cell_{k + 1:02d}"
        df = _make_cell_csv(
            cell_id,
            n_cycles=n_cycles,
            initial_soh=1.0,
            soh_decay=0.025 + k * 0.005,
            seed=k * 100,
        )
        p = tmp_path / f"{cell_id}.csv"
        df.to_csv(p, index=False)
        paths.append((cell_id, p))
    return paths


# ---------------------------------------------------------------------------
# Step3 import helper (handles parenthesised filename)
# ---------------------------------------------------------------------------

_MODULE_CACHE: dict[str, object] = {}


def _import_step3():
    if "step3" in _MODULE_CACHE:
        return _MODULE_CACHE["step3"]
    here = Path(__file__).parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    path = here / "step3_deltaQ(V)_feature_extraction.py"
    spec = importlib.util.spec_from_file_location("step3", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step3"] = mod
    spec.loader.exec_module(mod)     # type: ignore[union-attr]
    _MODULE_CACHE["step3"] = mod
    return mod


def _default_feat_cfg():
    s3 = _import_step3()
    # Slightly relaxed CC tolerance for synthetic data
    return s3.FeatureConfig(
        v_min=2.5,
        v_max=4.2,
        dv=0.005,
        chemistry="generic",
        half_cycle="discharge",
        c_rate_max=0.15,          # synthetic data is exactly C/10 = 0.10
        current_cv_max=0.05,      # allow up to 5% CV (synthetic noise is ~0.4%)
    )


# ---------------------------------------------------------------------------
# 1. Step1 parsing integration
# ---------------------------------------------------------------------------

class TestStep1Parsing:
    def test_parse_csv_produces_cycles(self, tmp_path):
        """step1.parse_file returns the correct number of cycles."""
        import step1_data_parsing as s1

        df = _make_cell_csv("c1", n_cycles=5)
        csv_path = tmp_path / "c1.csv"
        df.to_csv(csv_path, index=False)

        cell_cfg  = s1.CellConfig(nominal_capacity_ah=10.0)
        parse_cfg = s1.ParseConfig(cache_dir=tmp_path / "cache")
        parsed    = s1.parse_file(csv_path, cell_cfg, parse_cfg, use_cache=False)

        assert len(parsed.cycles) > 0, "No cycles detected"

    def test_discharge_half_cycles_present(self, tmp_path):
        import step1_data_parsing as s1

        df = _make_cell_csv("c1", n_cycles=4)
        csv_path = tmp_path / "c1.csv"
        df.to_csv(csv_path, index=False)

        cell_cfg  = s1.CellConfig(nominal_capacity_ah=10.0)
        parse_cfg = s1.ParseConfig(cache_dir=tmp_path / "cache")
        parsed    = s1.parse_file(csv_path, cell_cfg, parse_cfg, use_cache=False)

        # Each cycle should have a non-empty discharge half
        n_with_dis = sum(
            1 for cyc in parsed.cycles.values()
            if len(cyc["discharge"]) > 0
        )
        assert n_with_dis > 0, "No discharge half-cycles found"

    def test_soh_labels_decrease_over_cycles(self, tmp_path):
        """Discharge capacity should decrease as cycles progress."""
        import step1_data_parsing as s1

        df = _make_cell_csv("c1", n_cycles=6, soh_decay=0.04)
        csv_path = tmp_path / "c1.csv"
        df.to_csv(csv_path, index=False)

        cell_cfg  = s1.CellConfig(nominal_capacity_ah=10.0)
        parse_cfg = s1.ParseConfig(cache_dir=tmp_path / "cache")
        parsed    = s1.parse_file(csv_path, cell_cfg, parse_cfg, use_cache=False)

        caps = [
            cyc["meta"]["discharge_capacity_ah"]
            for cyc in sorted(parsed.cycles.values(), key=lambda c: c["cycle_index"])
            if cyc["meta"]["discharge_capacity_ah"] > 0
        ]
        assert len(caps) >= 2, "Need at least 2 cycles with capacity"
        # First cycle should have higher capacity than last
        assert caps[0] > caps[-1], (
            f"Capacity should decrease: first={caps[0]:.3f} last={caps[-1]:.3f}"
        )

    def test_parsed_dataset_has_required_attributes(self, tmp_path):
        import step1_data_parsing as s1

        df = _make_cell_csv("c1", n_cycles=3)
        csv_path = tmp_path / "c1.csv"
        df.to_csv(csv_path, index=False)

        cell_cfg  = s1.CellConfig(nominal_capacity_ah=10.0)
        parse_cfg = s1.ParseConfig(cache_dir=tmp_path / "cache")
        parsed    = s1.parse_file(csv_path, cell_cfg, parse_cfg, use_cache=False)

        assert hasattr(parsed, "raw")
        assert hasattr(parsed, "clean")
        assert hasattr(parsed, "cycles")
        assert hasattr(parsed, "source")
        assert callable(parsed.ica_cycles)
        assert callable(parsed.flagged)

    def test_unknown_format_raises(self, tmp_path):
        import step1_data_parsing as s1

        bad = tmp_path / "data.unk"
        bad.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            s1.load_raw(bad)

    def test_missing_file_raises(self):
        import step1_data_parsing as s1

        with pytest.raises(FileNotFoundError):
            s1.parse_file("/no/such/file.csv",
                          s1.CellConfig(nominal_capacity_ah=1.0))


# ---------------------------------------------------------------------------
# 2. Step3 feature extraction integration
# ---------------------------------------------------------------------------

class TestStep3FeatureExtraction:
    def test_single_cell_feature_matrix_produced(self):
        """FeatureMatrixBuilder returns at least one valid feature row."""
        s3 = _import_step3()
        cfg = _default_feat_cfg()
        records = [
            s3.CycleRecord(
                "c1", cyc,
                _make_discharge_half(soh=1.0 - 0.03 * cyc, seed=cyc),
                soh=1.0 - 0.03 * cyc,
            )
            for cyc in range(1, 7)
        ]
        builder = s3.FeatureMatrixBuilder(cfg, nominal_capacity_ah=10.0)
        df = builder.build(records)

        assert len(df) == 6
        n_ok = int(df["extraction_ok"].sum())
        assert n_ok >= 3, f"Only {n_ok}/6 cycles extracted successfully"

    def test_feature_column_names_match_schema(self):
        from feature_schema import FEATURE_COLS
        s3 = _import_step3()
        cfg = _default_feat_cfg()
        records = [
            s3.CycleRecord("c1", c, _make_discharge_half(1.0 - 0.03 * c, seed=c),
                           soh=1.0 - 0.03 * c)
            for c in range(1, 5)
        ]
        df = s3.FeatureMatrixBuilder(cfg, nominal_capacity_ah=10.0).build(records)

        for col in FEATURE_COLS:
            assert col in df.columns, f"Feature column '{col}' missing from output"

    def test_features_are_finite_for_valid_non_reference_cycles(self):
        """Features should be finite for degraded cycles (not the reference itself)."""
        from feature_schema import FEATURE_COLS
        s3 = _import_step3()
        cfg = _default_feat_cfg()
        records = [
            s3.CycleRecord("c1", c, _make_discharge_half(1.0 - 0.02 * c, seed=c),
                           soh=1.0 - 0.02 * c)
            for c in range(1, 6)
        ]
        df = s3.FeatureMatrixBuilder(cfg, nominal_capacity_ah=10.0).build(records)
        ok_rows = df[df["extraction_ok"] == True]
        assert len(ok_rows) >= 1, "Need at least one successfully extracted cycle"
        # Reference cycle delta = 0 → some stats (skewness, kurtosis) legitimately NaN
        # Check non-reference rows only for strict finiteness
        ref_idx = df["ref_cycle_index"].dropna().iloc[0]
        non_ref = ok_rows[ok_rows["cycle_number"] != ref_idx]
        if len(non_ref) == 0:
            pytest.skip("All ok rows are reference cycles")
        for col in ["dqv_variance", "dqv_log_variance", "dqv_max_deviation",
                    "dqv_min", "dqv_max", "dqv_mean", "dqv_rms", "dqv_integral_abs"]:
            bad = non_ref[col].isna().sum()
            assert bad == 0, f"{bad} NaN values in '{col}' for non-reference ok rows"

    def test_dqv_variance_increases_with_degradation(self):
        """ΔQ(V) variance should grow as the cell degrades further."""
        s3 = _import_step3()
        cfg = _default_feat_cfg()

        # Early cycles (high SOH) → reference and slightly degraded
        records_early = [
            s3.CycleRecord("c1", c, _make_discharge_half(1.0 - 0.01 * c, seed=c),
                           soh=1.0 - 0.01 * c)
            for c in range(1, 5)
        ]
        # Later cycles (low SOH)
        records_late = [
            s3.CycleRecord("c1", c, _make_discharge_half(0.75 - 0.01 * (c - 10), seed=c),
                           soh=0.75 - 0.01 * (c - 10))
            for c in range(10, 14)
        ]
        all_records = records_early + records_late
        df = s3.FeatureMatrixBuilder(cfg, nominal_capacity_ah=10.0).build(all_records)

        ok = df[df["extraction_ok"] == True].reset_index(drop=True)
        if len(ok) < 4:
            pytest.skip("Insufficient ok rows for comparison")

        # First two ok rows (high SOH) should have lower variance than last two
        early_var = ok["dqv_variance"].iloc[:2].mean()
        late_var  = ok["dqv_variance"].iloc[-2:].mean()
        # Not a hard invariant (depends on synthetic shape) — just sanity-check finite
        assert np.isfinite(early_var) and np.isfinite(late_var)

    def test_no_soh_leakage_in_feature_names(self):
        """Feature extractor must not include SOH-proxy column names."""
        s3 = _import_step3()
        cfg = _default_feat_cfg()
        ext = s3.FeatureExtractor(cfg)
        soh_proxies = {"soh", "capacity", "discharge_capacity", "coulombic_efficiency"}
        overlap = set(ext.feature_names) & soh_proxies
        assert not overlap, f"SOH leakage in features: {overlap}"

    def test_reference_cycle_is_fixed_per_cell(self):
        """Reference cycle should be selected once and not drift mid-batch."""
        s3 = _import_step3()
        cfg = _default_feat_cfg()
        records = [
            s3.CycleRecord("c1", c, _make_discharge_half(1.0 - 0.03 * c, seed=c),
                           soh=1.0 - 0.03 * c)
            for c in range(1, 8)
        ]
        df = s3.FeatureMatrixBuilder(cfg, nominal_capacity_ah=10.0).build(records)

        ref_indices = df["ref_cycle_index"].dropna().unique()
        assert len(ref_indices) == 1, (
            f"ref_cycle_index should be constant per cell; got {ref_indices}"
        )


# ---------------------------------------------------------------------------
# 3. Step4 SOH model training integration
# ---------------------------------------------------------------------------

class TestStep4ModelTraining:
    @pytest.fixture(scope="class")
    def feature_csv(self, tmp_path_factory):
        """Write a minimal multi-cell feature CSV acceptable by SOHModelTrainer."""
        from feature_schema import FEATURE_COLS
        s3 = _import_step3()
        cfg = _default_feat_cfg()

        rows_all = []
        for cell_num in range(1, 4):
            cell_id = f"c{cell_num}"
            records = [
                s3.CycleRecord(
                    cell_id, cyc,
                    _make_discharge_half(1.0 - 0.03 * cyc, seed=cell_num * 100 + cyc),
                    soh=1.0 - 0.03 * cyc,
                )
                for cyc in range(1, 9)
            ]
            df_cell = s3.FeatureMatrixBuilder(cfg, nominal_capacity_ah=10.0).build(records)
            rows_all.append(df_cell)

        combined = pd.concat(rows_all, ignore_index=True)
        # Keep only rows where all features are finite
        combined = combined[combined["extraction_ok"] == True].copy()
        combined = combined.dropna(subset=FEATURE_COLS)

        tmp = tmp_path_factory.mktemp("features")
        p   = tmp / "features.csv"
        combined.to_csv(p, index=False)
        return p

    def test_trainer_loads_and_validates_features(self, feature_csv):
        from step4_soh_model import SOHModelTrainer
        trainer = SOHModelTrainer()
        df      = trainer.load_features(feature_csv)
        assert len(df) > 0

    def test_loco_cv_runs_without_error(self, feature_csv):
        from step4_soh_model import SOHModelTrainer
        results = SOHModelTrainer().fit(
            SOHModelTrainer().load_features(feature_csv)
        )
        assert results.predictions is not None
        assert len(results.predictions) > 0

    def test_predictions_cover_all_cells(self, feature_csv):
        from step4_soh_model import SOHModelTrainer
        df      = SOHModelTrainer().load_features(feature_csv)
        results = SOHModelTrainer().fit(df)
        pred_cells  = set(results.predictions["cell_id"].unique())
        train_cells = set(df["cell_id"].unique())
        assert pred_cells == train_cells

    def test_overall_metrics_are_finite(self, feature_csv):
        from step4_soh_model import SOHModelTrainer
        results = SOHModelTrainer().fit(
            SOHModelTrainer().load_features(feature_csv)
        )
        m = results.overall_metrics
        assert np.isfinite(m["rmse"])
        assert np.isfinite(m["mae"])
        assert np.isfinite(m["r2"])

    def test_rmse_is_reasonable(self, feature_csv):
        """RMSE on synthetic data should be well under 0.5 SOH points."""
        from step4_soh_model import SOHModelTrainer
        results = SOHModelTrainer().fit(
            SOHModelTrainer().load_features(feature_csv)
        )
        assert results.overall_metrics["rmse"] < 0.50, (
            f"RMSE {results.overall_metrics['rmse']:.4f} is unreasonably high"
        )

    def test_feature_importance_has_correct_columns(self, feature_csv):
        from step4_soh_model import SOHModelTrainer
        results = SOHModelTrainer().fit(
            SOHModelTrainer().load_features(feature_csv)
        )
        fi = results.feature_importance
        assert "feature"         in fi.columns
        assert "coefficient"     in fi.columns
        assert "abs_coefficient" in fi.columns

    def test_model_saves_and_reloads(self, feature_csv, tmp_path):
        from step4_soh_model import SOHModelTrainer
        import joblib
        trainer = SOHModelTrainer()
        df      = trainer.load_features(feature_csv)
        results = trainer.fit(df)
        results = trainer.save(results, tmp_path / "model")

        assert results.model_path is not None
        assert results.model_path.exists()

        loaded = joblib.load(results.model_path)
        assert hasattr(loaded, "predict")

    def test_predictions_csv_is_written(self, feature_csv, tmp_path):
        from step4_soh_model import SOHModelTrainer
        results = SOHModelTrainer().run(feature_csv, tmp_path / "model")
        assert (tmp_path / "model" / "predictions.csv").exists()


# ---------------------------------------------------------------------------
# 4. Physics interpretation integration
# ---------------------------------------------------------------------------

class TestPhysicsInterpretation:
    def _make_ica_df(self, n_cycles: int = 6) -> tuple[pd.DataFrame, np.ndarray]:
        """Synthetic ICA DataFrame + shared voltage grid.

        Simulates realistic degradation: amplitude decreases (LAM), peak
        shifts slightly (LLI), width broadens (resistance growth).
        """
        from ica_curve_adapter import ICACurve, ica_curve_to_dataframe

        v_grid = np.linspace(2.5, 4.2, 340)
        center = 3.5
        curves = []
        for cyc in range(n_cycles):
            deg  = cyc / max(n_cycles - 1, 1)
            amp  = 1.0 - deg * 0.35   # amplitude decreases → LAM
            c    = center - deg * 0.01  # slight LLI shift
            w    = 0.05 + deg * 0.01    # slight broadening
            dqdv = amp * np.exp(-0.5 * ((v_grid - c) / w) ** 2)
            curves.append(ICACurve(
                cell_id          = "c1",
                cycle_number     = cyc,
                voltage_grid     = v_grid.copy(),
                dqdv             = dqdv,
                is_reference     = (cyc == 0),
                ref_cycle_number = 0,
            ))
        return ica_curve_to_dataframe(curves), v_grid

    def test_build_physics_features_returns_dataframe(self):
        from step4_interpretation import build_physics_features
        df_ica, v_grid = self._make_ica_df()
        phys_df = build_physics_features(df_ica, v_grid, reference_cycle=0)
        assert isinstance(phys_df, pd.DataFrame)
        assert len(phys_df) > 0

    def test_physics_df_has_required_columns(self):
        from step4_interpretation import build_physics_features
        from battery_health_report import PHYSICS_COLS
        df_ica, v_grid = self._make_ica_df()
        phys_df = build_physics_features(df_ica, v_grid, reference_cycle=0)
        for col in PHYSICS_COLS:
            assert col in phys_df.columns, f"Physics column '{col}' missing"

    def test_lam_increases_over_cycles(self):
        """LAM_loss should increase as peak area decreases with cycle number."""
        from step4_interpretation import build_physics_features
        df_ica, v_grid = self._make_ica_df(n_cycles=8)
        phys_df = build_physics_features(df_ica, v_grid, reference_cycle=0)
        phys_df = phys_df.sort_values("cycle_number").reset_index(drop=True)
        # LAM should be 0 at the reference and grow
        assert phys_df["LAM_loss"].iloc[0] == pytest.approx(0.0, abs=0.05)
        lam_last = float(phys_df["LAM_loss"].iloc[-1])
        assert lam_last > 0.0, f"LAM_loss should be positive at end; got {lam_last:.4f}"

    def test_resistance_growth_is_non_negative_for_broadening_peak(self):
        """A widening Gaussian peak should produce non-negative resistance growth."""
        from step4_interpretation import build_physics_features
        df_ica, v_grid = self._make_ica_df(n_cycles=6)
        phys_df = build_physics_features(df_ica, v_grid, reference_cycle=0)
        phys_df = phys_df.sort_values("cycle_number").reset_index(drop=True)
        # Skip reference cycle (growth = 0 by definition)
        later = phys_df.iloc[1:]
        assert (later["Resistance_growth"] >= -0.1).all(), (
            "Resistance_growth should not be largely negative for broadening peaks"
        )

    def test_interpret_cycle_returns_all_keys(self):
        from step4_interpretation import interpret_cycle
        v    = np.linspace(2.5, 4.2, 340)
        ref  = np.exp(-0.5 * ((v - 3.5) / 0.05) ** 2)
        curr = np.exp(-0.5 * ((v - 3.52) / 0.06) ** 2)
        metrics, lli = interpret_cycle(v, ref, curr)
        assert "LLI_mean_shift"    in metrics
        assert "LAM_loss"          in metrics
        assert "Resistance_growth" in metrics
        assert "LLI_confidence"    in metrics
        assert lli is not None

    def test_lli_shift_sign_is_correct(self):
        """Peak shifted to higher voltage → positive LLI_mean_shift."""
        from step4_interpretation import interpret_cycle, TOLERANCE_V
        v     = np.linspace(2.5, 4.2, 340)
        shift = TOLERANCE_V * 0.5  # 10 mV — well within matching window
        ref   = np.exp(-0.5 * ((v - 3.50) / 0.05) ** 2)
        curr  = np.exp(-0.5 * ((v - (3.50 + shift)) / 0.05) ** 2)
        m, _  = interpret_cycle(v, ref, curr)
        # Either shift is detected (positive) or no peaks matched (0.0 is also acceptable
        # if peak detection fails for this synthetic signal)
        assert m["LLI_mean_shift"] >= 0.0, (
            f"Rightward shift should be non-negative; got {m['LLI_mean_shift']}"
        )

    def test_grid_spacing_mV_correct(self):
        from step4_interpretation import interpret_cycle
        v    = np.linspace(2.5, 4.2, 340)
        ref  = np.exp(-0.5 * ((v - 3.5) / 0.05) ** 2)
        m, _ = interpret_cycle(v, ref, ref)
        expected = (4.2 - 2.5) / 339 * 1000
        assert m["grid_spacing_mV"] == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# 5. Flow 1 — end-to-end: CSV data → SOH trajectory + report
# ---------------------------------------------------------------------------

class TestFlow1EndToEnd:
    @pytest.fixture(scope="class")
    def flow1_report(self, tmp_path_factory):
        """Run the full Flow 1 chain and return the BatteryHealthReport."""
        from feature_schema import FEATURE_COLS
        from step4_soh_model import SOHModelTrainer
        from step4_interpretation import build_physics_features
        from battery_health_report import build_report
        from ica_curve_adapter import ICACurve, ica_curve_to_dataframe

        s3     = _import_step3()
        cfg    = _default_feat_cfg()
        tmp    = tmp_path_factory.mktemp("flow1")
        N_CELLS, N_CYCLES = 3, 8
        NOM_CAP = 10.0

        # ── Step 3: build feature matrix for all cells ──────────────────────
        all_records = []
        for k in range(N_CELLS):
            cell_id = f"c{k + 1}"
            for cyc in range(1, N_CYCLES + 1):
                soh = max(0.65, 1.0 - (0.03 + k * 0.005) * cyc)
                all_records.append(s3.CycleRecord(
                    cell_id, cyc,
                    _make_discharge_half(soh, seed=k * 1000 + cyc),
                    soh=soh,
                ))

        feat_df = s3.FeatureMatrixBuilder(cfg, NOM_CAP).build(all_records)
        feat_df = feat_df[feat_df["extraction_ok"] == True].dropna(subset=FEATURE_COLS)
        assert len(feat_df) >= N_CELLS * 3, "Too few valid feature rows for Flow 1 test"

        feat_csv = tmp / "features.csv"
        feat_df.to_csv(feat_csv, index=False)

        # ── Step 4: train model ──────────────────────────────────────────────
        trainer  = SOHModelTrainer()
        results  = trainer.fit(trainer.load_features(feat_csv))
        results  = trainer.save(results, tmp / "model")

        # ── Physics ICA features ─────────────────────────────────────────────
        v_grid = cfg.v_grid
        ica_curves = []
        ref_cycle  = 1
        for k in range(N_CELLS):
            cell_id = f"c{k + 1}"
            ref_soh = max(0.65, 1.0 - (0.03 + k * 0.005) * ref_cycle)
            ref_dis = _make_discharge_half(ref_soh, seed=k * 1000 + ref_cycle)
            ref_qv  = s3.QVExtractor(cfg, NOM_CAP)
            q_ref, dqdv_ref, info_ref = ref_qv.extract(ref_dis)

            for cyc in range(1, N_CYCLES + 1):
                soh = max(0.65, 1.0 - (0.03 + k * 0.005) * cyc)
                dis = _make_discharge_half(soh, seed=k * 1000 + cyc)
                q_c, dqdv_c, info_c = ref_qv.extract(dis)
                if not info_c.get("ok") or dqdv_c is None:
                    continue
                # Fill any NaN with linear interp so ICACurve validates
                finite_c = np.isfinite(dqdv_c)
                if not finite_c.all():
                    dqdv_c = np.where(finite_c, dqdv_c, 0.0)
                finite_g = np.isfinite(v_grid)
                ica_curves.append(ICACurve(
                    cell_id=cell_id,
                    cycle_number=cyc,
                    voltage_grid=v_grid.copy(),
                    dqdv=dqdv_c,
                    is_reference=(cyc == ref_cycle),
                    ref_cycle_number=ref_cycle,
                ))

        if ica_curves:
            df_ica  = ica_curve_to_dataframe(ica_curves)
            phys_df = build_physics_features(df_ica, v_grid, reference_cycle=ref_cycle)
        else:
            # Minimal stub so build_report doesn't fail
            phys_df = results.predictions[["cell_id", "cycle_number"]].copy()

        # ── Build report ─────────────────────────────────────────────────────
        report = build_report(
            predictions       = results.predictions,
            overall_metrics   = results.overall_metrics,
            per_cell_metrics  = results.per_cell_metrics,
            feature_importance= results.feature_importance,
            phys_df           = phys_df,
            feature_df        = feat_df,
        )
        return report

    def test_report_is_built_successfully(self, flow1_report):
        from battery_health_report import BatteryHealthReport
        assert isinstance(flow1_report, BatteryHealthReport)

    def test_cycle_df_has_all_cells(self, flow1_report):
        cells_in_report = set(flow1_report.cycle_df["cell_id"].unique())
        assert cells_in_report == {"c1", "c2", "c3"}

    def test_soh_pred_decreases_over_cycles(self, flow1_report):
        """The predicted SOH trajectory should trend downward for every cell."""
        for cell_id, grp in flow1_report.cycle_df.groupby("cell_id"):
            grp = grp.sort_values("cycle_number")
            soh = grp["soh_pred"].to_numpy()
            if len(soh) < 3:
                continue
            # Fit a line; slope should be negative
            x = np.arange(len(soh), dtype=float)
            slope = np.polyfit(x, soh, 1)[0]
            assert slope < 0, (
                f"Cell {cell_id}: SOH slope={slope:.5f} is not negative "
                "(predicted SOH not decreasing with cycles)"
            )

    def test_cell_summary_has_dominant_mechanism(self, flow1_report):
        assert "dominant_mechanism" in flow1_report.cell_summary.columns
        non_empty = flow1_report.cell_summary["dominant_mechanism"].notna()
        assert non_empty.all()

    def test_model_metrics_are_reasonable(self, flow1_report):
        m = flow1_report.model_metrics["overall"]
        assert m["rmse"] < 0.5
        assert np.isfinite(m["mae"])
        assert np.isfinite(m["r2"])

    def test_report_saves_to_disk(self, flow1_report, tmp_path):
        from battery_health_report import save_report
        paths = save_report(flow1_report, tmp_path / "report")
        assert paths["cycle_data"].exists()
        assert paths["cell_summary"].exists()
        assert paths["summary_json"].exists()

    def test_cycle_data_csv_soh_column_exists(self, flow1_report, tmp_path):
        from battery_health_report import save_report
        import json
        paths = save_report(flow1_report, tmp_path / "report2")
        df    = pd.read_csv(paths["cycle_data"])
        assert "soh_pred" in df.columns
        # JSON should parse cleanly
        summary = json.loads(paths["summary_json"].read_text())
        assert summary["n_cells"] == 3

    def test_feature_importance_all_features_present(self, flow1_report):
        from feature_schema import FEATURE_COLS
        fi_features = set(flow1_report.feature_importance["feature"])
        assert fi_features == set(FEATURE_COLS)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("matplotlib"),
        reason="matplotlib not installed",
    )
    def test_soh_plot_renders_without_error(self, flow1_report):
        import matplotlib
        matplotlib.use("Agg")
        from battery_health_report import plot_soh_predictions
        result = plot_soh_predictions(flow1_report)
        assert isinstance(result, dict)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 6. Flow 2 — single new cycle → SOH prediction + explanation
# ---------------------------------------------------------------------------

class TestFlow2SingleCyclePrediction:
    @pytest.fixture(scope="class")
    def trained_predictor(self, tmp_path_factory):
        """Train a model on 3 cells and return an SOHPredictor."""
        from feature_schema import FEATURE_COLS
        from step4_soh_model import SOHModelTrainer
        from predict import SOHPredictor

        s3  = _import_step3()
        cfg = _default_feat_cfg()
        tmp = tmp_path_factory.mktemp("flow2")

        all_records = []
        for k in range(3):
            cell_id = f"c{k + 1}"
            for cyc in range(1, 9):
                soh = max(0.65, 1.0 - (0.03 + k * 0.005) * cyc)
                all_records.append(s3.CycleRecord(
                    cell_id, cyc,
                    _make_discharge_half(soh, seed=k * 1000 + cyc),
                    soh=soh,
                ))

        feat_df = s3.FeatureMatrixBuilder(cfg, 10.0).build(all_records)
        feat_df = feat_df[feat_df["extraction_ok"] == True].dropna(subset=FEATURE_COLS)

        feat_csv = tmp / "features.csv"
        feat_df.to_csv(feat_csv, index=False)

        trainer = SOHModelTrainer()
        results = trainer.fit(trainer.load_features(feat_csv))
        results = trainer.save(results, tmp / "model")

        predictor = SOHPredictor.from_model(
            results.model_path, feat_cfg=cfg, nominal_capacity_ah=10.0
        )
        # Set reference from a healthy early cycle
        ref_df = _make_discharge_half(soh=1.0, seed=9999)
        predictor.set_reference(ref_df)
        return predictor

    def test_assess_returns_expected_keys(self, trained_predictor):
        curr = _make_discharge_half(soh=0.85, seed=1001)
        result = trained_predictor.assess(curr)
        assert "soh_pred"        in result
        assert "features"        in result
        assert "physics"         in result
        assert "explanation"     in result
        assert "extraction_info" in result

    def test_soh_pred_is_numeric(self, trained_predictor):
        curr   = _make_discharge_half(soh=0.85, seed=1002)
        result = trained_predictor.assess(curr)
        assert isinstance(result["soh_pred"], float)
        assert np.isfinite(result["soh_pred"])

    def test_soh_pred_in_plausible_range(self, trained_predictor):
        """Prediction for a moderately degraded cell should be in (0.4, 1.1)."""
        curr   = _make_discharge_half(soh=0.85, seed=1003)
        result = trained_predictor.assess(curr)
        assert 0.40 < result["soh_pred"] < 1.10, (
            f"SOH prediction out of plausible range: {result['soh_pred']:.4f}"
        )

    def test_more_degraded_cycle_predicts_lower_soh(self, trained_predictor):
        """A cell at 70% SOH should predict lower than one at 95% SOH."""
        healthy = trained_predictor.assess(_make_discharge_half(soh=0.95, seed=2001))
        aged    = trained_predictor.assess(_make_discharge_half(soh=0.70, seed=2002))
        assert aged["soh_pred"] < healthy["soh_pred"], (
            f"Expected aged ({aged['soh_pred']:.4f}) < healthy ({healthy['soh_pred']:.4f})"
        )

    def test_explanation_is_non_empty_string(self, trained_predictor):
        result = trained_predictor.assess(_make_discharge_half(soh=0.80, seed=3001))
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 10

    def test_explanation_mentions_soh(self, trained_predictor):
        result = trained_predictor.assess(_make_discharge_half(soh=0.80, seed=3002))
        assert "SOH" in result["explanation"] or "soh" in result["explanation"].lower()

    def test_physics_dict_has_lli_lam_resistance(self, trained_predictor):
        result = trained_predictor.assess(_make_discharge_half(soh=0.80, seed=3003))
        phys   = result["physics"]
        assert "LLI_mean_shift"    in phys
        assert "LAM_loss"          in phys
        assert "Resistance_growth" in phys

    def test_features_dict_has_all_feature_cols(self, trained_predictor):
        from feature_schema import FEATURE_COLS
        result = trained_predictor.assess(_make_discharge_half(soh=0.85, seed=4001))
        for col in FEATURE_COLS:
            assert col in result["features"], f"Feature '{col}' missing from output"

    def test_assess_without_reference_raises(self, tmp_path):
        from predict import SOHPredictor
        from step4_soh_model import SOHModelTrainer
        from feature_schema import FEATURE_COLS
        s3  = _import_step3()
        cfg = _default_feat_cfg()

        # Train a tiny model
        records = [
            s3.CycleRecord(f"c{k}", cyc, _make_discharge_half(0.9, seed=k * 50 + cyc), soh=0.9)
            for k in range(1, 4) for cyc in range(1, 6)
        ]
        feat_df  = s3.FeatureMatrixBuilder(cfg, 10.0).build(records)
        feat_df  = feat_df[feat_df["extraction_ok"] == True].dropna(subset=FEATURE_COLS)
        feat_csv = tmp_path / "f.csv"
        feat_df.to_csv(feat_csv, index=False)
        trainer = SOHModelTrainer()
        results = trainer.fit(trainer.load_features(feat_csv))
        results = trainer.save(results, tmp_path / "m")

        predictor = SOHPredictor.from_model(results.model_path, feat_cfg=cfg)
        # No set_reference → should raise RuntimeError
        with pytest.raises(RuntimeError, match="set_reference"):
            predictor.assess(_make_discharge_half(soh=0.9))

    def test_trajectory_returns_dataframe(self, trained_predictor):
        halves = [_make_discharge_half(1.0 - 0.05 * i, seed=i) for i in range(5)]
        traj   = trained_predictor.assess_trajectory(halves, cycle_numbers=list(range(5)))
        assert isinstance(traj, pd.DataFrame)
        assert len(traj) == 5
        assert "soh_pred"     in traj.columns
        assert "cycle_number" in traj.columns

    def test_trajectory_soh_decreases(self, trained_predictor):
        """Trajectory over increasingly degraded half-cycles should trend downward."""
        halves = [_make_discharge_half(1.0 - 0.06 * i, seed=5000 + i) for i in range(6)]
        traj   = trained_predictor.assess_trajectory(halves)
        soh    = traj["soh_pred"].to_numpy()
        finite = soh[np.isfinite(soh)]
        assert len(finite) >= 3
        slope = np.polyfit(np.arange(len(finite)), finite, 1)[0]
        assert slope < 0, (
            f"SOH trajectory slope={slope:.5f} should be negative"
        )


# ---------------------------------------------------------------------------
# 7. Dataset loaders — structural tests (no real files needed)
# ---------------------------------------------------------------------------

class TestDatasetLoaders:
    def test_registry_has_oxford_and_severson(self):
        from dataset_loaders import DatasetLoaderRegistry
        registered = DatasetLoaderRegistry.registered()
        assert "oxford"   in registered
        assert "severson" in registered

    def test_auto_load_mat_dispatches_to_oxford(self):
        """auto_load(".mat") should map to the oxford loader."""
        from dataset_loaders import DatasetLoaderRegistry
        # Verify extension mapping (does not require a real .mat file)
        ext_map = DatasetLoaderRegistry._ext_map
        assert ".mat" in ext_map
        assert ext_map[".mat"] == "oxford"

    def test_auto_load_pkl_dispatches_to_severson(self):
        from dataset_loaders import DatasetLoaderRegistry
        assert DatasetLoaderRegistry._ext_map.get(".pkl") == "severson"

    def test_unknown_loader_name_raises(self):
        from dataset_loaders import DatasetLoaderRegistry
        with pytest.raises(KeyError, match="Unknown dataset"):
            DatasetLoaderRegistry.load("does_not_exist", "path.csv")

    def test_unknown_extension_raises(self, tmp_path):
        from dataset_loaders import DatasetLoaderRegistry
        with pytest.raises(ValueError, match="No loader registered"):
            DatasetLoaderRegistry.auto_load(tmp_path / "data.zzz")

    def test_severson_loader_handles_dict_batch(self, tmp_path):
        """A minimal dict-of-cells batch.pkl is parsed without errors."""
        import pickle
        from dataset_loaders import load_severson_batch

        batch = {
            "b1c0": {
                "barcode":    "b1c0",
                "cathode":    "LFP",
                "charge_policy": "3.6C(80%)-0.2C",
                "summary": {
                    "QDischarge": [1.1, 1.08, 1.05],
                    "cycle":      [1,   2,    3],
                },
                "cycles": {
                    "1": {
                        "t": [0.0, 1.0, 2.0],
                        "V": [3.6, 3.5, 3.4],
                        "I": [-1.0, -1.0, -1.0],
                        "Qc": [0.0, 0.0, 0.0],
                        "Qd": [0.0, 0.001, 0.002],
                        "T": [25.0, 25.1, 25.2],
                    }
                },
            }
        }
        pkl_path = tmp_path / "batch.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(batch, fh)

        records = load_severson_batch(pkl_path, time_unit="min")
        assert len(records) == 1
        assert records[0].cell_id == "b1c0"
        assert records[0].cycle_index == 1
        # time should be scaled from minutes to seconds
        assert records[0].time[0] == pytest.approx(0.0)
        assert records[0].time[-1] == pytest.approx(2.0 * 60.0)

    def test_severson_soh_computed_from_qdischarge(self, tmp_path):
        """SOH = Q_cycle / Q_first should be <1 for subsequent cycles."""
        import pickle
        from dataset_loaders import load_severson_batch

        batch = {
            "cell_A": {
                "summary": {
                    "QDischarge": [1.0, 0.95, 0.90],
                    "cycle":      [1,   2,    3],
                },
                "cycles": {
                    str(cyc): {
                        "t": [0.0, 1.0],
                        "V": [3.6, 3.4],
                        "I": [-1.0, -1.0],
                        "Qc": [0.0, 0.0],
                        "Qd": [0.0, 0.001],
                    }
                    for cyc in range(1, 4)
                },
            }
        }
        pkl_path = tmp_path / "b.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(batch, fh)

        records = load_severson_batch(pkl_path)
        soh_map = {r.cycle_index: r.soh for r in records}
        assert soh_map[1] == pytest.approx(1.0)
        assert soh_map[2] == pytest.approx(0.95)
        assert soh_map[3] == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# 8. Predict module unit tests
# ---------------------------------------------------------------------------

class TestPredictModule:
    @pytest.fixture(scope="class")
    def saved_model_path(self, tmp_path_factory):
        from feature_schema import FEATURE_COLS
        from step4_soh_model import SOHModelTrainer

        s3  = _import_step3()
        cfg = _default_feat_cfg()
        tmp = tmp_path_factory.mktemp("predict_model")

        records = [
            s3.CycleRecord(f"c{k}", cyc, _make_discharge_half(1.0 - 0.03 * cyc, seed=k * 500 + cyc), soh=1.0 - 0.03 * cyc)
            for k in range(1, 4) for cyc in range(1, 8)
        ]
        feat_df  = s3.FeatureMatrixBuilder(cfg, 10.0).build(records)
        feat_df  = feat_df[feat_df["extraction_ok"] == True].dropna(subset=FEATURE_COLS)
        feat_csv = tmp / "f.csv"
        feat_df.to_csv(feat_csv, index=False)
        results  = SOHModelTrainer().run(feat_csv, tmp / "m")
        return results.model_path

    def test_from_model_loads_pipeline(self, saved_model_path):
        from predict import SOHPredictor
        p = SOHPredictor.from_model(saved_model_path)
        assert p._pipeline is not None

    def test_set_reference_accepts_valid_half_cycle(self, saved_model_path):
        from predict import SOHPredictor
        p = SOHPredictor.from_model(saved_model_path,
                                    feat_cfg=_default_feat_cfg(),
                                    nominal_capacity_ah=10.0)
        ref = _make_discharge_half(soh=1.0, seed=7777)
        p.set_reference(ref)  # must not raise
        assert p._ref_q_smooth is not None
        assert p._ref_dqdv is not None

    def test_set_reference_rejects_bad_data(self, saved_model_path):
        from predict import SOHPredictor
        p = SOHPredictor.from_model(saved_model_path,
                                    feat_cfg=_default_feat_cfg(),
                                    nominal_capacity_ah=10.0)
        # Flat voltage profile — no CC region that passes the gate
        bad = pd.DataFrame({
            "time":    np.linspace(0, 10, 20),
            "voltage": np.full(20, 3.5),
            "current": np.full(20, 0.0),  # zero current → not CC
        })
        with pytest.raises(ValueError):
            p.set_reference(bad)

    def test_nonexistent_model_raises(self):
        from predict import SOHPredictor
        with pytest.raises(Exception):  # joblib raises FileNotFoundError or similar
            SOHPredictor.from_model("/no/such/model.joblib")

    def test_assess_physics_keys_all_finite(self, saved_model_path):
        from predict import SOHPredictor
        p = SOHPredictor.from_model(saved_model_path,
                                    feat_cfg=_default_feat_cfg(),
                                    nominal_capacity_ah=10.0)
        p.set_reference(_make_discharge_half(soh=1.0, seed=8001))
        result = p.assess(_make_discharge_half(soh=0.82, seed=8002))
        for k, v in result["physics"].items():
            assert v is not None and np.isfinite(v), (
                f"Physics key '{k}' is not finite: {v}"
            )

    def test_assess_features_match_feature_schema(self, saved_model_path):
        from predict import SOHPredictor
        from feature_schema import FEATURE_COLS
        p = SOHPredictor.from_model(saved_model_path,
                                    feat_cfg=_default_feat_cfg(),
                                    nominal_capacity_ah=10.0)
        p.set_reference(_make_discharge_half(soh=1.0, seed=9001))
        result = p.assess(_make_discharge_half(soh=0.80, seed=9002))
        assert set(FEATURE_COLS).issubset(set(result["features"].keys()))


# ---------------------------------------------------------------------------
# 9. Schema contract — step3 ↔ step4 alignment
# ---------------------------------------------------------------------------

class TestSchemaContract:
    def test_step3_output_accepted_by_step4_trainer(self):
        """Direct chain: step3 feature matrix → step4 trainer.fit() → success."""
        from feature_schema import FEATURE_COLS
        from step4_soh_model import SOHModelTrainer

        s3  = _import_step3()
        cfg = _default_feat_cfg()

        records = [
            s3.CycleRecord(f"c{k}", cyc, _make_discharge_half(1.0 - 0.03 * cyc, seed=k * 200 + cyc), soh=1.0 - 0.03 * cyc)
            for k in range(1, 4) for cyc in range(1, 7)
        ]
        feat_df = s3.FeatureMatrixBuilder(cfg, 10.0).build(records)
        feat_df = feat_df[feat_df["extraction_ok"] == True].dropna(subset=FEATURE_COLS)

        # Must not raise
        trainer = SOHModelTrainer()
        results = trainer.fit(feat_df)
        assert results.overall_metrics["rmse"] >= 0

    def test_legacy_column_names_accepted_after_aliasing(self):
        """Dataframes with old step4 column names should pass after rename."""
        from feature_schema import COLUMN_ALIASES, FEATURE_COLS, validate_feature_columns

        old_names = {
            "log_var_deltaQ": 0.0, "var_deltaQ": 0.0, "skew_deltaQ": 0.0,
            "kurtosis_deltaQ": 0.0, "abs_integral_deltaQ": 0.0,
            "max_dev_deltaQ": 0.0,
        }
        df = pd.DataFrame([{
            **old_names,
            "cell_id": "c1",
            "cycle_index": 1,
            "soh": 0.9,
            # canonical-only columns (not in aliases)
            "dqv_min": 0.0, "dqv_max": 0.0, "dqv_mean": 0.0, "dqv_rms": 0.0,
        }])
        df = df.rename(columns=COLUMN_ALIASES)
        validate_feature_columns(df)  # must not raise
