"""
test_resistance_growth.py
Tests for the dimensional fix in step4_interpretation.py.

Core invariant: Resistance_growth must be identical (within numerical noise)
for the same physical scenario measured on grids with different spacing.
"""
from __future__ import annotations

import numpy as np
import pytest

from step4_interpretation import (
    PEAK_WIDTH_MIN_MV,
    compute_resistance_growth,
    extract_peaks,
    grid_spacing_mv,
    interpret_cycle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_dqdv(voltage_v: np.ndarray, center_v: float, width_v: float) -> np.ndarray:
    """Gaussian ICA peak centred at center_v with physical half-width width_v (V)."""
    return np.exp(-0.5 * ((voltage_v - center_v) / width_v) ** 2)


def _make_scenario(
    dv_v: float,
    ref_width_v: float = 0.050,
    curr_width_v: float = 0.070,
    center_v: float = 3.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ref + current Gaussian curves on a grid with step dv_v (Volts)."""
    v        = np.arange(2.5, 4.2 + dv_v / 2, dv_v)
    ref_dqdv = _gaussian_dqdv(v, center_v, ref_width_v)
    cur_dqdv = _gaussian_dqdv(v, center_v, curr_width_v)
    return v, ref_dqdv, cur_dqdv


# ---------------------------------------------------------------------------
# 1. grid_spacing_mv public helper
# ---------------------------------------------------------------------------

class TestGridSpacingMv:
    def test_correct_value_5mv_grid(self):
        v = np.arange(2.5, 4.2 + 0.005 / 2, 0.005)
        assert grid_spacing_mv(v) == pytest.approx(5.0, rel=1e-4)

    def test_correct_value_2p5mv_grid(self):
        v = np.arange(2.5, 4.2 + 0.0025 / 2, 0.0025)
        assert grid_spacing_mv(v) == pytest.approx(2.5, rel=1e-4)

    def test_raises_on_too_short_array(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            grid_spacing_mv(np.array([3.5]))

    def test_raises_on_non_monotonic_grid(self):
        v = np.array([2.5, 2.6, 2.55, 2.7])
        with pytest.raises(ValueError):
            grid_spacing_mv(v)

    def test_raises_on_non_uniform_grid(self):
        v = np.array([2.5, 2.505, 2.520, 2.540])  # irregular steps
        with pytest.raises(ValueError):
            grid_spacing_mv(v)

    def test_passes_nearly_uniform_grid(self):
        v      = np.linspace(2.5, 4.2, 340)
        result = grid_spacing_mv(v)
        assert result == pytest.approx((4.2 - 2.5) / 339 * 1000, rel=1e-4)


# ---------------------------------------------------------------------------
# 2. extract_peaks — widths are in mV
# ---------------------------------------------------------------------------

class TestExtractPeaksUnits:
    def test_widths_mV_are_non_negative(self):
        v, ref, _ = _make_scenario(0.005)
        info = extract_peaks(v, ref)
        assert np.all(info["widths_mV"] >= 0)

    def test_grid_spacing_mV_key_present(self):
        v, ref, _ = _make_scenario(0.005)
        info = extract_peaks(v, ref)
        assert "widths_mV"       in info
        assert "grid_spacing_mV" in info
        assert "peak_voltage"    in info

    def test_grid_spacing_mV_matches_helper(self):
        v, ref, _ = _make_scenario(0.005)
        info = extract_peaks(v, ref)
        assert info["grid_spacing_mV"] == pytest.approx(grid_spacing_mv(v), rel=1e-6)

    def test_widths_mV_scale_with_physical_peak_width(self):
        """A wider Gaussian → wider peak in mV."""
        v      = np.arange(2.5, 4.2 + 0.005 / 2, 0.005)
        narrow = extract_peaks(v, _gaussian_dqdv(v, 3.5, 0.040))
        wide   = extract_peaks(v, _gaussian_dqdv(v, 3.5, 0.080))
        if len(narrow["widths_mV"]) > 0 and len(wide["widths_mV"]) > 0:
            assert wide["widths_mV"][0] > narrow["widths_mV"][0]

    def test_peak_width_filter_is_grid_invariant(self):
        """Minimum-width filter should detect the same peak count on both grids."""
        v5,  ref5,  _ = _make_scenario(0.005)
        v25, ref25, _ = _make_scenario(0.0025)
        info5  = extract_peaks(v5,  ref5)
        info25 = extract_peaks(v25, ref25)
        assert len(info5["peaks"]) == len(info25["peaks"]), (
            f"Peak count differs: 5mV grid={len(info5['peaks'])}, "
            f"2.5mV grid={len(info25['peaks'])}"
        )


# ---------------------------------------------------------------------------
# 3. compute_resistance_growth — dimensional assertions
# ---------------------------------------------------------------------------

class TestComputeResistanceGrowth:
    def test_zero_when_widths_unchanged(self):
        w = np.array([50.0, 60.0])
        assert compute_resistance_growth(w, w) == pytest.approx(0.0, abs=1e-9)

    def test_positive_when_peaks_broader(self):
        assert compute_resistance_growth(np.array([50.0]), np.array([60.0])) > 0.0

    def test_negative_when_peaks_narrower(self):
        assert compute_resistance_growth(np.array([60.0]), np.array([50.0])) < 0.0

    def test_returns_zero_on_empty_ref(self):
        assert compute_resistance_growth(np.array([]), np.array([50.0])) == 0.0

    def test_returns_zero_on_empty_curr(self):
        assert compute_resistance_growth(np.array([50.0]), np.array([])) == 0.0

    def test_assertion_fires_on_non_positive_width(self):
        """Negative or zero widths mean raw sample indices were passed instead of mV."""
        with pytest.raises(AssertionError, match="non-positive"):
            compute_resistance_growth(np.array([-5.0]), np.array([10.0]))

    def test_correct_fractional_growth(self):
        ref  = np.array([40.0])   # 40 mV ref peak
        curr = np.array([60.0])   # 60 mV → 50% growth
        assert compute_resistance_growth(ref, curr) == pytest.approx(0.5, rel=1e-4)


# ---------------------------------------------------------------------------
# 4. Grid-density invariance  (the primary regression test)
# ---------------------------------------------------------------------------

class TestGridDensityInvariance:
    """Resistance_growth must be grid-density-invariant."""

    def _growth(self, dv_v: float, ref_w: float, curr_w: float) -> float:
        v, ref, curr = _make_scenario(dv_v, ref_width_v=ref_w, curr_width_v=curr_w)
        metrics, _   = interpret_cycle(v, ref, curr)
        return metrics["Resistance_growth"]

    def test_same_growth_on_5mv_and_2p5mv_grids(self):
        g5   = self._growth(0.005,  0.050, 0.070)
        g2p5 = self._growth(0.0025, 0.050, 0.070)
        assert abs(g5 - g2p5) < 0.05, (
            f"Resistance_growth differs by grid density: "
            f"5mV={g5:.4f}, 2.5mV={g2p5:.4f}"
        )

    def test_same_growth_on_10mv_and_5mv_grids(self):
        g10 = self._growth(0.010, 0.060, 0.090)
        g5  = self._growth(0.005, 0.060, 0.090)
        assert abs(g10 - g5) < 0.05, (
            f"Resistance_growth differs by grid density: "
            f"10mV={g10:.4f}, 5mV={g5:.4f}"
        )

    def test_zero_growth_invariant_across_grids(self):
        """Identical ref and current → growth ≈ 0 regardless of grid."""
        v5,  r5,  _ = _make_scenario(0.005)
        v25, r25, _ = _make_scenario(0.0025)
        m5,  _ = interpret_cycle(v5,  r5,  r5)
        m25, _ = interpret_cycle(v25, r25, r25)
        assert abs(m5["Resistance_growth"])  < 0.02
        assert abs(m25["Resistance_growth"]) < 0.02

    def test_old_index_units_would_fail_invariance(self):
        """Confirm raw sample widths differ ~2× across grids (the historical bug).

        On a 5 mV grid one physical peak = N samples.
        On a 2.5 mV grid the same peak = ~2N samples.
        After the fix, widths_mV are equal; sample counts are not.
        """
        v5,  ref5,  _ = _make_scenario(0.005,  0.050, 0.070)
        v25, ref25, _ = _make_scenario(0.0025, 0.050, 0.070)

        # Recover raw sample counts by dividing mV widths by their grid spacing
        w5_samples  = extract_peaks(v5,  ref5)["widths_mV"] / grid_spacing_mv(v5)
        w25_samples = extract_peaks(v25, ref25)["widths_mV"] / grid_spacing_mv(v25)

        if len(w5_samples) > 0 and len(w25_samples) > 0:
            assert w25_samples[0] == pytest.approx(w5_samples[0] * 2, rel=0.15), (
                "Expected 2× sample count for half the grid spacing"
            )


# ---------------------------------------------------------------------------
# 5. interpret_cycle output keys
# ---------------------------------------------------------------------------

class TestInterpretCycleOutputKeys:
    def test_required_keys_present(self):
        v, ref, curr  = _make_scenario(0.005)
        metrics, _lli = interpret_cycle(v, ref, curr)
        assert "Resistance_growth" in metrics
        assert "LLI_mean_shift"    in metrics
        assert "LAM_loss"          in metrics
        assert "widths_mV" not in metrics  # arrays stay in extract_peaks, not here

    def test_resistance_growth_is_dimensionless(self):
        """Growth fraction for a ~40% width increase should be in a sane range."""
        v, ref, curr = _make_scenario(0.005, ref_width_v=0.050, curr_width_v=0.070)
        metrics, _   = interpret_cycle(v, ref, curr)
        frac         = metrics["Resistance_growth"]
        assert -1.0 < frac < 5.0, (
            f"Resistance_growth={frac:.3f} is unreasonably large; "
            "check that widths_mV (not sample indices) are used"
        )

    def test_grid_spacing_mV_passed_through(self):
        """grid_spacing_mV should appear in interpret_cycle metrics for provenance."""
        v, ref, curr = _make_scenario(0.005)
        metrics, _   = interpret_cycle(v, ref, curr)
        assert "grid_spacing_mV" in metrics
        assert metrics["grid_spacing_mV"] == pytest.approx(5.0, rel=1e-3)

    def test_mean_widths_mV_tracked(self):
        """Mean peak width (mV) of both ref and curr should be in metrics."""
        v, ref, curr = _make_scenario(0.005)
        metrics, _   = interpret_cycle(v, ref, curr)
        assert "mean_ref_width_mV"  in metrics
        assert "mean_curr_width_mV" in metrics


# ---------------------------------------------------------------------------
# 6. PEAK_WIDTH_MIN_MV constant
# ---------------------------------------------------------------------------

class TestPeakWidthMinMv:
    def test_constant_is_positive(self):
        assert PEAK_WIDTH_MIN_MV > 0

    def test_width_min_in_samples_scales_with_grid(self):
        """Minimum-samples threshold should double when grid spacing halves."""
        min_5mv   = max(1, round(PEAK_WIDTH_MIN_MV / 5.0))
        min_2p5mv = max(1, round(PEAK_WIDTH_MIN_MV / 2.5))
        assert min_2p5mv == pytest.approx(min_5mv * 2, abs=1)
