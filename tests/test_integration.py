"""
Regression tests for numerical integration in step4_interpretation.py.
These fix against known analytical values so any future scipy API change
or sign error in compute_lam is caught immediately.
"""

import numpy as np
import pytest
from scipy.integrate import simpson

from step4_interpretation import compute_lam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine_curve(n=200):
    """sin(x) over [0, pi] — analytical integral = 2.0."""
    x = np.linspace(0, np.pi, n)
    return x, np.sin(x)


def _flat_curve(n=200):
    """f(x)=1 over [0, 1] — analytical integral = 1.0."""
    x = np.linspace(0, 1, n)
    return x, np.ones(n)


# ---------------------------------------------------------------------------
# scipy.integrate.simpson baseline
# ---------------------------------------------------------------------------

def test_simpson_sine_analytical():
    x, y = _sine_curve()
    result = simpson(y, x=x)
    assert abs(result - 2.0) < 1e-4, f"Expected ≈2.0, got {result}"


def test_simpson_flat_analytical():
    x, y = _flat_curve()
    result = simpson(y, x=x)
    assert abs(result - 1.0) < 1e-4, f"Expected ≈1.0, got {result}"


def test_simpson_keyword_x_arg():
    """x= keyword form must be identical to positional call."""
    x, y = _sine_curve()
    assert simpson(y, x=x) == simpson(y, x)


# ---------------------------------------------------------------------------
# compute_lam regression
# ---------------------------------------------------------------------------

def test_compute_lam_identical_signals_is_zero():
    x, y = _sine_curve()
    lam = compute_lam(x, y, y)
    assert abs(lam) < 1e-6, f"Identical signals → LAM should be 0, got {lam}"


def test_compute_lam_half_signal():
    """curr = 0.5 * ref → LAM ≈ 0.5 (50 % active material loss)."""
    x, y = _sine_curve()
    lam = compute_lam(x, y, 0.5 * y)
    assert abs(lam - 0.5) < 1e-4, f"Expected LAM ≈ 0.5, got {lam}"


def test_compute_lam_zero_signal():
    """Fully degraded cell (curr ≈ 0) → LAM ≈ 1.0."""
    x, y = _sine_curve()
    lam = compute_lam(x, y, np.zeros_like(y))
    assert abs(lam - 1.0) < 1e-4, f"Expected LAM ≈ 1.0, got {lam}"


def test_compute_lam_sine_reference_value():
    """Pin the exact LAM value for the 200-point sine reference curve."""
    x, y = _sine_curve(n=200)
    lam = compute_lam(x, y, 0.75 * y)
    assert abs(lam - 0.25) < 1e-4, f"Expected LAM ≈ 0.25, got {lam}"


def test_compute_lam_monotone_voltage():
    """Non-uniform voltage spacing is handled correctly via x= kwarg."""
    voltage = np.array([3.0, 3.2, 3.5, 3.9, 4.1, 4.2])
    ref = np.array([0.1, 0.3, 0.5, 0.4, 0.2, 0.1])
    curr = ref * 0.6
    lam = compute_lam(voltage, ref, curr)
    assert abs(lam - 0.4) < 1e-4, f"Expected LAM ≈ 0.4, got {lam}"
