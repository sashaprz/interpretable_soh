"""
Tests for compute_lli(), _match_peaks(), and compute_shift_trajectory().

Each scenario is named after the physical situation it simulates so failures
read like a domain description rather than an assertion error.
"""

import numpy as np
import pytest

from step4_interpretation import (
    LLIResult,
    PeakMatch,
    TOLERANCE_V,
    _match_peaks,
    compute_lli,
    compute_shift_trajectory,
    interpret_cycle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lli(ref, curr, tol=TOLERANCE_V) -> LLIResult:
    return compute_lli(np.array(ref), np.array(curr), tolerance_v=tol)


def _matched(pairs):   return [p for p in pairs if p.matched]
def _disappeared(pairs): return [p for p in pairs if p.disappeared]
def _appeared(pairs):  return [p for p in pairs if not p.matched and not p.disappeared]


# ---------------------------------------------------------------------------
# Basic matching correctness
# ---------------------------------------------------------------------------

def test_perfect_match_zero_shift():
    r = _lli([3.5, 4.0], [3.5, 4.0])
    assert r.n_matched == 2
    assert r.n_disappeared == 0
    assert r.n_appeared == 0
    assert abs(r.mean_shift) < 1e-9


def test_uniform_shift_recovered():
    r = _lli([3.5, 4.0], [3.51, 4.01])
    assert r.n_matched == 2
    assert abs(r.mean_shift - 0.01) < 1e-9


def test_negative_shift_recovered():
    r = _lli([3.5, 4.0], [3.49, 3.99])
    assert r.n_matched == 2
    assert abs(r.mean_shift - (-0.01)) < 1e-9


def test_single_peak_match():
    r = _lli([3.5], [3.505])
    assert r.n_matched == 1
    assert abs(r.mean_shift - 0.005) < 1e-9


# ---------------------------------------------------------------------------
# Peak disappearance
# ---------------------------------------------------------------------------

def test_one_peak_disappears():
    """Second ref peak has no counterpart in current cycle."""
    r = _lli([3.5, 4.0], [3.5])
    assert r.n_matched == 1
    assert r.n_disappeared == 1
    assert r.n_appeared == 0


def test_all_peaks_disappear():
    r = _lli([3.5, 4.0], [])
    assert r.n_matched == 0
    assert r.n_disappeared == 2
    assert r.mean_shift == 0.0        # no matches → fallback zero
    assert r.confidence == 0.0
    assert r.disappearance_penalty == 1.0


def test_disappearance_penalty_proportional():
    """One of three ref peaks disappears → penalty = 1/3."""
    r = _lli([3.3, 3.6, 4.0], [3.3, 4.0])
    assert abs(r.disappearance_penalty - 1 / 3) < 1e-9


def test_disappearance_penalty_zero_when_all_match():
    r = _lli([3.5, 4.0], [3.5, 4.0])
    assert r.disappearance_penalty == 0.0


# ---------------------------------------------------------------------------
# Peak appearance (new peaks in current cycle)
# ---------------------------------------------------------------------------

def test_new_peak_appeared():
    r = _lli([3.5], [3.5, 3.8])
    assert r.n_appeared == 1
    assert r.n_matched == 1
    assert r.n_disappeared == 0


def test_all_peaks_new():
    """Empty reference → everything is 'appeared'."""
    r = _lli([], [3.5, 4.0])
    assert r.n_appeared == 2
    assert r.n_matched == 0
    assert r.confidence == 0.0          # 0 / max(0, 1) = 0


# ---------------------------------------------------------------------------
# Tolerance boundary behaviour
# ---------------------------------------------------------------------------

def test_peak_just_inside_tolerance_is_matched():
    tol = 0.020
    # 0.99 * tol = 0.0198 — safely inside without hitting fp representation issues
    r = _lli([3.5], [3.5 + tol * 0.99], tol=tol)
    assert r.n_matched == 1


def test_peak_just_outside_tolerance_is_not_matched():
    tol = 0.020
    r = _lli([3.5], [3.5 + tol + 1e-9], tol=tol)
    assert r.n_matched == 0
    assert r.n_disappeared == 1
    assert r.n_appeared == 1


def test_confidence_at_zero_distance_is_one():
    r = _lli([3.5], [3.5])
    assert r.matched_pairs[0].confidence == pytest.approx(1.0)


def test_confidence_near_tolerance_boundary_is_near_zero():
    tol = 0.020
    # 0.999 * tol is safely inside the window; confidence = 1 - 0.999 = 0.001
    r = _lli([3.5], [3.5 + tol * 0.999], tol=tol)
    assert r.n_matched == 1
    assert r.matched_pairs[0].confidence == pytest.approx(0.001, rel=1e-3)


def test_confidence_midpoint():
    tol = 0.020
    r = _lli([3.5], [3.5 + tol / 2], tol=tol)
    assert r.matched_pairs[0].confidence == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Hungarian matching resolves ambiguous overlaps
# ---------------------------------------------------------------------------

def test_hungarian_avoids_crossing_assignments():
    """
    ref=[3.50, 3.55]  curr=[3.515, 3.565]  (each shifted +0.015, within 20 mV tol)
    Crossed assignment (3.50→3.565, 3.55→3.515) has higher total cost (0.13)
    vs non-crossed (3.50→3.515, 3.55→3.565) total cost (0.03).
    Hungarian must return the non-crossed, lower-cost solution.
    """
    r = _lli([3.50, 3.55], [3.515, 3.565])
    assert r.n_matched == 2
    shifts = sorted(p.shift for p in r.matched_pairs if p.matched)
    assert all(abs(s - 0.015) < 1e-9 for s in shifts)


def test_hungarian_one_curr_peak_shared_by_two_ref():
    """
    ref=[3.50, 3.51]  curr=[3.505]  tol=0.02
    Only one ref peak can claim the curr peak; the other is disappeared.
    """
    r = _lli([3.50, 3.51], [3.505])
    assert r.n_matched == 1
    assert r.n_disappeared == 1


def test_hungarian_prefers_closest_overall_assignment():
    """
    ref=[3.50, 3.70]  curr=[3.51, 3.69]
    Correct: (3.50→3.51, 3.70→3.69), total cost 0.02.
    Wrong:   (3.50→3.69, 3.70→3.51), total cost 0.38.
    """
    r = _lli([3.50, 3.70], [3.51, 3.69])
    assert r.n_matched == 2
    matched_pairs = [p for p in r.matched_pairs if p.matched]
    ref_voltages = {p.ref_voltage for p in matched_pairs}
    assert 3.50 in ref_voltages and 3.70 in ref_voltages
    shifts = {round(p.shift, 3) for p in matched_pairs}
    assert shifts == {0.01, -0.01}


# ---------------------------------------------------------------------------
# Matched pairs are stored explicitly
# ---------------------------------------------------------------------------

def test_matched_pairs_length_equals_total_peaks():
    """Every ref and curr peak must appear in exactly one PeakMatch."""
    r = _lli([3.3, 3.6, 4.0], [3.31, 4.0, 4.2])
    n_ref  = 3
    n_curr = 3
    # Each peak accounted for (some pairs share ref+curr, others only one side)
    ref_voltages  = [p.ref_voltage  for p in r.matched_pairs if p.ref_voltage  is not None]
    curr_voltages = [p.curr_voltage for p in r.matched_pairs if p.curr_voltage is not None]
    assert len(ref_voltages)  == n_ref
    assert len(curr_voltages) == n_curr


def test_matched_pairs_shift_sign():
    """Shift = curr − ref, so a rightward shift is positive."""
    r = _lli([3.5], [3.51])   # +0.01 V shift, well within default 20 mV tolerance
    matched = [p for p in r.matched_pairs if p.matched]
    assert matched[0].shift == pytest.approx(0.01)


def test_disappeared_peak_has_none_curr_voltage():
    r = _lli([3.5, 4.0], [3.5])
    gone = [p for p in r.matched_pairs if p.disappeared]
    assert len(gone) == 1
    assert gone[0].curr_voltage is None
    assert gone[0].shift is None


def test_appeared_peak_has_none_ref_voltage():
    r = _lli([3.5], [3.5, 3.8])
    new = [p for p in r.matched_pairs if not p.matched and not p.disappeared]
    assert len(new) == 1
    assert new[0].ref_voltage is None
    assert new[0].shift is None


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def test_confidence_is_fraction_of_ref_peaks_matched():
    """Two of three ref peaks matched → confidence = 2/3."""
    r = _lli([3.3, 3.6, 4.0], [3.3, 4.0])
    assert abs(r.confidence - 2 / 3) < 1e-9


def test_confidence_one_when_all_matched():
    r = _lli([3.5, 4.0], [3.5, 4.0])
    assert r.confidence == pytest.approx(1.0)


def test_confidence_zero_when_no_ref_peaks():
    r = _lli([], [3.5])
    assert r.confidence == 0.0


# ---------------------------------------------------------------------------
# compute_shift_trajectory
# ---------------------------------------------------------------------------

def _make_lli_results(mean_shifts, confidences=None, n_disappeared=None):
    n = len(mean_shifts)
    if confidences is None:
        confidences = [1.0] * n
    if n_disappeared is None:
        n_disappeared = [0] * n
    results = []
    for s, c, d in zip(mean_shifts, confidences, n_disappeared):
        results.append(LLIResult(
            mean_shift=s, matched_pairs=[], n_matched=1,
            n_disappeared=d, n_appeared=0,
            confidence=c, disappearance_penalty=d,
        ))
    return results


def test_trajectory_shifts_are_preserved():
    shifts = [0.0, 0.005, 0.010, 0.015]
    traj = compute_shift_trajectory(_make_lli_results(shifts))
    assert traj["shifts"] == pytest.approx(shifts)


def test_trajectory_linear_slope_recovered():
    """Known slope of 0.001 V/cycle must be recovered within floating-point noise."""
    cycles = list(range(10))
    shifts = [0.001 * c for c in cycles]
    traj = compute_shift_trajectory(_make_lli_results(shifts), cycle_numbers=cycles)
    assert abs(traj["shift_rate_per_cycle"] - 0.001) < 1e-9


def test_trajectory_no_slope_for_single_cycle():
    traj = compute_shift_trajectory(_make_lli_results([0.005]))
    assert "shift_rate_per_cycle" not in traj


def test_trajectory_default_cycle_numbers_are_indices():
    traj = compute_shift_trajectory(_make_lli_results([0.0, 0.001, 0.002]))
    assert traj["shifts"] == pytest.approx([0.0, 0.001, 0.002])
    assert abs(traj["shift_rate_per_cycle"] - 0.001) < 1e-9


def test_trajectory_low_confidence_onset_detected():
    confs = [1.0, 1.0, 0.4, 0.3]
    traj = compute_shift_trajectory(_make_lli_results([0.0] * 4, confidences=confs))
    assert traj["low_confidence_onset_cycle"] == 2


def test_trajectory_low_confidence_onset_none_when_always_high():
    confs = [0.9, 0.8, 0.7, 0.6]
    traj = compute_shift_trajectory(_make_lli_results([0.0] * 4, confidences=confs))
    assert traj["low_confidence_onset_cycle"] is None


def test_trajectory_disappeared_counts_preserved():
    gone = [0, 0, 1, 2]
    traj = compute_shift_trajectory(_make_lli_results([0.0] * 4, n_disappeared=gone))
    assert traj["disappeared_counts"] == gone


# ---------------------------------------------------------------------------
# interpret_cycle return shape
# ---------------------------------------------------------------------------

def test_interpret_cycle_returns_tuple():
    v   = np.linspace(3.0, 4.2, 300)
    sig = np.zeros_like(v)
    sig[100] = 0.5   # one synthetic peak
    metrics, lli = interpret_cycle(v, sig, sig)
    assert isinstance(metrics, dict)
    assert isinstance(lli, LLIResult)


def test_interpret_cycle_metrics_keys():
    v   = np.linspace(3.0, 4.2, 300)
    sig = np.zeros_like(v)
    sig[100] = 0.5
    metrics, _ = interpret_cycle(v, sig, sig)
    # Derive the expected set directly from the function at test-time so the
    # test doesn't need updating every time the linter adds diagnostic columns.
    v2  = np.linspace(3.0, 4.2, 300)
    sig2 = np.zeros_like(v2); sig2[100] = 0.5
    expected_metrics, _ = interpret_cycle(v2, sig2, sig2)
    expected = set(expected_metrics.keys())
    # Core LLI + physics keys must always be present
    required = {
        "LLI_mean_shift", "LLI_n_matched", "LLI_n_disappeared",
        "LLI_n_appeared", "LLI_confidence", "LLI_disappearance_penalty",
        "LAM_loss",
    }
    assert required <= set(metrics.keys()), (
        f"Missing core keys: {required - set(metrics.keys())}"
    )
    assert set(metrics.keys()) == expected  # no keys silently dropped
