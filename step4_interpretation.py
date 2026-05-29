"""
Step 5: Physics-based interpretation of battery degradation
Input:
    - ICA curves: dQ/dV vs V for each cycle
    - reference cycle ICA (early life baseline)

Output:
    - LLI score (peak shift, disappearance penalty, confidence)
    - LAM score (peak area loss)
    - Resistance growth score (peak broadening)
    - combined degradation signature per cycle

Assumptions:
    - ICA peaks correspond to phase transitions in electrode materials
    - Peaks are matched across cycles by nearest voltage within TOLERANCE_V
    - Hungarian algorithm resolves ambiguous many-to-many overlaps

Dimensional note on peak widths
--------------------------------
find_peaks() returns widths in *sample index* units, not physical voltage units.
All width values exposed by this module are in **millivolts** (mV).  Conversion:

    width_mV = width_samples * grid_spacing_mV

where grid_spacing_mV comes from ICAConfig.dv_mv (stored in
ICACurve.diagnostics["grid_spacing_mV"]).  This makes width-based metrics
grid-spacing-invariant: a 10 mV peak is 10 mV regardless of whether the grid
was built at 1 mV or 5 mV resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

# =========================================================
# 1. CORE CONFIG
# =========================================================

PEAK_PROMINENCE   = 0.02
PEAK_WIDTH_MIN_MV = 5.0    # minimum peak half-width at half-prominence (mV)
TOLERANCE_V       = 0.020  # 20 mV matching window for peak correspondence

# =========================================================
# 2. UTILITY FUNCTIONS
# =========================================================

def smooth_signal(y, window=11):
    """Simple smoothing fallback if SG already applied."""
    return np.convolve(y, np.ones(window) / window, mode="same")


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)


def grid_spacing_mv(voltage_v: np.ndarray, rel_tol: float = 1e-3) -> float:
    """Return mean grid spacing in mV; raise ValueError if grid is not uniform.

    Public wrapper around _assert_uniform_grid with ValueError semantics.
    rel_tol is the maximum allowed relative variation in spacing (default 0.1%).
    """
    if len(voltage_v) < 2:
        raise ValueError("voltage_v must have at least 2 points.")
    try:
        return _assert_uniform_grid(voltage_v, tol_frac=rel_tol)
    except AssertionError as exc:
        raise ValueError(str(exc)) from exc


def _assert_uniform_grid(voltage: np.ndarray, tol_frac: float = 0.02) -> float:
    """Assert voltage is a uniform grid; return spacing in mV.

    Parameters
    ----------
    voltage  : strictly increasing voltage array (V)
    tol_frac : max allowed fractional deviation of any step from the mean step

    Returns
    -------
    spacing_mV : mean grid spacing in millivolts

    Raises
    ------
    AssertionError if grid spacing deviates beyond tol_frac or is non-positive.
    """
    diffs_mV = np.diff(voltage) * 1000.0
    spacing_mV = float(np.mean(diffs_mV))
    assert spacing_mV > 0, (
        f"Voltage grid has non-positive mean spacing ({spacing_mV:.4f} mV). "
        "Grid must be strictly increasing."
    )
    max_dev_frac = float(np.max(np.abs(diffs_mV - spacing_mV))) / spacing_mV
    assert max_dev_frac < tol_frac, (
        f"Voltage grid is not uniform: mean {spacing_mV:.4f} mV, "
        f"max deviation {100 * max_dev_frac:.1f}% > {100 * tol_frac:.0f}% tolerance. "
        "Width measurements in mV require a uniform grid."
    )
    return spacing_mV

# =========================================================
# 3. PEAK DETECTION
# =========================================================

def extract_peaks(
    voltage: np.ndarray,
    dqdv: np.ndarray,
    grid_spacing_mV: float | None = None,
) -> dict:
    """Detect ICA peaks and return widths in **millivolts**.

    Parameters
    ----------
    voltage         : uniform voltage grid (V); used to convert index units -> mV
    dqdv            : dQ/dV on that grid
    grid_spacing_mV : explicit grid step (mV); inferred from `voltage` when None

    Returns
    -------
    dict with keys:
        peaks           : integer indices of detected peaks
        peak_voltage    : voltages of peaks (V)
        peak_height     : dQ/dV values at peaks (Ah/V)
        widths_mV       : peak widths at half-prominence height, in **mV**
        grid_spacing_mV : spacing used for conversion (mV)

    Width invariance
    ----------------
    widths_mV = raw_sample_widths * grid_spacing_mV
    A physical peak of width W mV produces the same widths_mV regardless of
    whether the ICA grid was built at 1 mV, 2 mV, or 5 mV resolution.
    """
    voltage = np.asarray(voltage, dtype=float)
    dqdv    = np.asarray(dqdv,    dtype=float)

    if grid_spacing_mV is None:
        grid_spacing_mV = _assert_uniform_grid(voltage)
    assert grid_spacing_mV > 0, (
        f"grid_spacing_mV={grid_spacing_mV:.4f} is non-positive"
    )

    # Convert mV threshold -> sample indices for find_peaks
    width_min_pts = max(1, round(PEAK_WIDTH_MIN_MV / grid_spacing_mV))

    peaks, props = find_peaks(
        dqdv,
        prominence=PEAK_PROMINENCE,
        width=width_min_pts,
    )

    # ── Dimensional conversion: samples -> mV ──────────────────────────────
    raw_widths_samples = props.get("widths", np.array([]))
    widths_mV = raw_widths_samples * grid_spacing_mV  # [samples] * [mV/sample] = mV

    return {
        "peaks":           peaks,
        "peak_voltage":    voltage[peaks] if len(peaks) else np.array([]),
        "peak_height":     dqdv[peaks]    if len(peaks) else np.array([]),
        "widths_mV":       widths_mV,        # physically meaningful: mV
        "grid_spacing_mV": grid_spacing_mV,  # pass-through for downstream validation
    }

# =========================================================
# 4. LLI DATA STRUCTURES
# =========================================================

@dataclass
class PeakMatch:
    """Correspondence between one reference peak and one current-cycle peak."""

    ref_voltage:  float | None  # voltage of the reference peak; None = appeared
    curr_voltage: float | None  # voltage of the current peak; None = disappeared
    shift:        float | None  # curr_voltage - ref_voltage; None if unmatched
    matched:      bool          # True when both sides present within TOLERANCE_V
    disappeared:  bool          # True when ref peak has no current counterpart
    confidence:   float         # 1.0 at zero distance, 0.0 at tolerance boundary


@dataclass
class LLIResult:
    """Full output of compute_lli() for one cycle pair."""

    mean_shift:            float            # mean shift of matched peaks (V)
    matched_pairs:         list[PeakMatch]  # every match, disappearance, appearance
    n_matched:             int
    n_disappeared:         int              # ref peaks absent from current cycle
    n_appeared:            int              # current peaks absent from reference
    confidence:            float            # n_matched / max(n_ref, 1)
    disappearance_penalty: float            # n_disappeared / max(n_ref, 1)

# =========================================================
# 5. FEATURE: LLI (PEAK MATCHING + SHIFT)
# =========================================================

def _match_peaks(
    ref_v: np.ndarray,
    curr_v: np.ndarray,
    tolerance_v: float,
) -> list[PeakMatch]:
    """
    Hungarian-optimal peak matching within tolerance_v.

    Unmatched ref peaks  -> disappeared (PeakMatch.disappeared = True).
    Unmatched curr peaks -> appeared   (matched = False, disappeared = False).
    """
    ref_v  = np.asarray(ref_v,  dtype=float)
    curr_v = np.asarray(curr_v, dtype=float)

    if len(ref_v) == 0 and len(curr_v) == 0:
        return []
    if len(ref_v) == 0:
        return [PeakMatch(None, float(v), None, False, False, 0.0) for v in curr_v]
    if len(curr_v) == 0:
        return [PeakMatch(float(v), None, None, False, True, 0.0) for v in ref_v]

    # Build absolute-distance cost matrix and solve with Hungarian algorithm.
    cost = np.abs(ref_v[:, None] - curr_v[None, :])   # shape (n_ref, n_curr)
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_ref:  set[int] = set()
    matched_curr: set[int] = set()
    pairs: list[PeakMatch] = []

    for r, c in zip(row_ind, col_ind):
        dist = float(cost[r, c])
        if dist <= tolerance_v:
            pairs.append(PeakMatch(
                ref_voltage=float(ref_v[r]),
                curr_voltage=float(curr_v[c]),
                shift=float(curr_v[c] - ref_v[r]),
                matched=True,
                disappeared=False,
                confidence=1.0 - dist / tolerance_v,
            ))
            matched_ref.add(r)
            matched_curr.add(c)

    for r in range(len(ref_v)):
        if r not in matched_ref:
            pairs.append(PeakMatch(float(ref_v[r]), None, None, False, True, 0.0))

    for c in range(len(curr_v)):
        if c not in matched_curr:
            pairs.append(PeakMatch(None, float(curr_v[c]), None, False, False, 0.0))

    return pairs


def compute_lli(
    ref_peaks_v,
    curr_peaks_v,
    tolerance_v: float = TOLERANCE_V,
) -> LLIResult:
    """
    Match ICA peaks by nearest voltage using Hungarian assignment.

    Parameters
    ----------
    ref_peaks_v  : array-like of reference-cycle peak voltages (V)
    curr_peaks_v : array-like of current-cycle peak voltages (V)
    tolerance_v  : maximum match distance in volts (default 20 mV)

    Returns
    -------
    LLIResult -- matched pairs, mean shift, confidence, and disappearance penalty.
    Peaks farther apart than tolerance_v are treated as disappeared / appeared.
    """
    ref_v  = np.asarray(ref_peaks_v,  dtype=float)
    curr_v = np.asarray(curr_peaks_v, dtype=float)
    pairs  = _match_peaks(ref_v, curr_v, tolerance_v)

    matched    = [p for p in pairs if p.matched]
    disappeared = [p for p in pairs if p.disappeared]
    appeared   = [p for p in pairs if not p.matched and not p.disappeared]

    n_ref  = len(ref_v)
    shifts = [p.shift for p in matched]   # type: ignore[union-attr]

    return LLIResult(
        mean_shift=float(np.mean(shifts)) if shifts else 0.0,
        matched_pairs=pairs,
        n_matched=len(matched),
        n_disappeared=len(disappeared),
        n_appeared=len(appeared),
        confidence=len(matched) / max(n_ref, 1),
        disappearance_penalty=len(disappeared) / max(n_ref, 1),
    )

# =========================================================
# 6. FEATURE: LAM (PEAK AREA LOSS)
# =========================================================

def compute_lam(voltage, ref_dqdv, curr_dqdv):
    """LAM = loss of integrated ICA signal (active material loss)."""
    ref_area  = simpson(ref_dqdv,  x=voltage)
    curr_area = simpson(curr_dqdv, x=voltage)
    return 1.0 - (curr_area / (ref_area + 1e-8))

# =========================================================
# 7. FEATURE: RESISTANCE GROWTH (PEAK BROADENING)
# =========================================================

def compute_resistance_growth(
    ref_widths_mV: np.ndarray,
    curr_widths_mV: np.ndarray,
) -> float:
    """Fractional change in ICA peak width (mV) -> polarisation / impedance growth.

    Both arrays **must be in millivolts** (not sample indices).
    Normalising by ref_widths_mV makes the metric grid-spacing-invariant:
    the same physical broadening produces the same score regardless of the
    ICA grid resolution used.

    Parameters
    ----------
    ref_widths_mV  : peak widths (mV) from the reference cycle
    curr_widths_mV : peak widths (mV) from the current cycle

    Returns
    -------
    float : mean fractional width increase; positive = broader = more resistance.
    """
    ref_widths_mV  = np.asarray(ref_widths_mV,  dtype=float)
    curr_widths_mV = np.asarray(curr_widths_mV, dtype=float)

    if ref_widths_mV.size == 0 or curr_widths_mV.size == 0:
        return 0.0

    assert np.all(ref_widths_mV  > 0), (
        "ref_widths_mV contains non-positive values; "
        "pass widths in mV (not raw sample indices)."
    )
    assert np.all(curr_widths_mV > 0), (
        "curr_widths_mV contains non-positive values; "
        "pass widths in mV (not raw sample indices)."
    )

    n = min(len(ref_widths_mV), len(curr_widths_mV))
    growth = (curr_widths_mV[:n] - ref_widths_mV[:n]) / (ref_widths_mV[:n] + 1e-8)
    return float(np.mean(growth))

# =========================================================
# 8. PER-CYCLE INTERPRETATION
# =========================================================

def interpret_cycle(
    voltage:         np.ndarray,
    ref_dqdv:        np.ndarray,
    curr_dqdv:       np.ndarray,
    grid_spacing_mV: float | None = None,
) -> tuple[dict, LLIResult]:
    """
    Compute physics degradation descriptors for one cycle.

    Parameters
    ----------
    voltage         : uniform ICA voltage grid (V) shared by ref and curr
    ref_dqdv        : dQ/dV of the reference (early-life) cycle
    curr_dqdv       : dQ/dV of the cycle being assessed
    grid_spacing_mV : explicit grid step (mV); inferred from `voltage` when None

    Returns
    -------
    metrics : dict       -- flat scalars ready for a features DataFrame
    lli     : LLIResult  -- full peak-matching detail (pairs, confidence, shifts)

    Notes
    -----
    widths_mV are grid-spacing-invariant: a 10 mV peak produces the same value
    whether the grid was built at 1 mV or 5 mV resolution.
    """
    # Compute grid spacing once; share it so both calls are consistent
    if grid_spacing_mV is None:
        grid_spacing_mV = _assert_uniform_grid(voltage)

    ref  = extract_peaks(voltage, ref_dqdv,  grid_spacing_mV=grid_spacing_mV)
    curr = extract_peaks(voltage, curr_dqdv, grid_spacing_mV=grid_spacing_mV)

    lli    = compute_lli(ref["peak_voltage"], curr["peak_voltage"])
    lam    = compute_lam(voltage, ref_dqdv, curr_dqdv)
    resist = compute_resistance_growth(ref["widths_mV"], curr["widths_mV"])

    mean_ref_width_mV  = float(np.mean(ref["widths_mV"]))  if ref["widths_mV"].size  else float("nan")
    mean_curr_width_mV = float(np.mean(curr["widths_mV"])) if curr["widths_mV"].size else float("nan")

    metrics = {
        "LLI_mean_shift":            lli.mean_shift,
        "LLI_n_matched":             lli.n_matched,
        "LLI_n_disappeared":         lli.n_disappeared,
        "LLI_n_appeared":            lli.n_appeared,
        "LLI_confidence":            lli.confidence,
        "LLI_disappearance_penalty": lli.disappearance_penalty,
        "LAM_loss":                  lam,
        "Resistance_growth":         resist,
        "mean_ref_width_mV":         mean_ref_width_mV,
        "mean_curr_width_mV":        mean_curr_width_mV,
        "grid_spacing_mV":           grid_spacing_mV,
    }
    return metrics, lli

# =========================================================
# 9. PEAK SHIFT TRAJECTORY ACROSS CYCLES
# =========================================================

def compute_shift_trajectory(
    lli_results: list[LLIResult],
    cycle_numbers: list[int] | None = None,
) -> dict:
    """
    Summarise how peak shift evolves across a cell's cycle history.

    Parameters
    ----------
    lli_results  : ordered list of LLIResult, one entry per cycle
    cycle_numbers: matching cycle indices; defaults to 0, 1, 2, ...

    Returns
    -------
    dict with keys:
        shifts, confidences, disappeared_counts -- per-cycle lists
        shift_rate_per_cycle, shift_intercept   -- linear trend (only if >= 2 cycles)
        low_confidence_onset_cycle              -- first cycle where confidence < 0.5
                                                   (None if never drops below threshold)
    """
    if cycle_numbers is None:
        cycle_numbers = list(range(len(lli_results)))

    cycles      = np.array(cycle_numbers, dtype=float)
    shifts      = np.array([r.mean_shift    for r in lli_results], dtype=float)
    confidences = np.array([r.confidence    for r in lli_results], dtype=float)
    disappeared = np.array([r.n_disappeared for r in lli_results], dtype=int)

    result: dict = {
        "shifts":             shifts.tolist(),
        "confidences":        confidences.tolist(),
        "disappeared_counts": disappeared.tolist(),
    }

    if len(cycles) >= 2:
        slope, intercept = np.polyfit(cycles, shifts, 1)
        result["shift_rate_per_cycle"] = float(slope)
        result["shift_intercept"]      = float(intercept)

    low_conf = confidences < 0.5
    result["low_confidence_onset_cycle"] = (
        int(cycles[low_conf][0]) if low_conf.any() else None
    )

    return result

# =========================================================
# 10. BUILD FULL DATASET OVER ALL CYCLES
# =========================================================

def build_physics_features(df_ica, voltage_grid, reference_cycle=0):
    """
    df_ica:
        ICA DataFrame from ica_curve_adapter.ica_curve_to_dataframe().
        Required columns: cell_id, cycle_number, dqdv (numpy float64 arrays).
        If is_reference is present, that row is used as the reference;
        otherwise the row matching cycle_number == reference_cycle is used.

    returns:
        DataFrame with physics interpretation features per (cell_id, cycle_number).
        Includes mean_curr_width_mV and grid_spacing_mV for peak broadening plots.
    """
    voltage_grid = np.asarray(voltage_grid, dtype=float)

    # Validate and cache grid spacing once -- all rows share the same grid
    grid_spacing_mV = _assert_uniform_grid(voltage_grid)

    results = []

    for cell in df_ica["cell_id"].unique():
        df_cell = df_ica[df_ica["cell_id"] == cell]

        if "is_reference" in df_cell.columns and df_cell["is_reference"].any():
            ref_row = df_cell[df_cell["is_reference"]].iloc[0]
        else:
            ref_row = df_cell[df_cell["cycle_number"] == reference_cycle].iloc[0]

        ref_dqdv = ref_row["dqdv"]

        for _, row in df_cell.iterrows():
            phys_dict, _lli = interpret_cycle(
                voltage_grid, ref_dqdv, row["dqdv"],
                grid_spacing_mV=grid_spacing_mV,
            )
            results.append({
                "cell_id":      cell,
                "cycle_number": row["cycle_number"],
                **phys_dict,
            })

    return pd.DataFrame(results)

# =========================================================
# 11. PEAK BROADENING EVOLUTION PLOT
# =========================================================

def plot_peak_broadening_evolution(
    phys_df: pd.DataFrame,
    cell_ids: list[str] | None = None,
    normalise_to_first: bool = True,
    save_path: str | None = None,
) -> str | None:
    """Plot ICA peak width (mV) vs cycle number to visualise resistance growth.

    Requires that `phys_df` was produced by build_physics_features() and
    therefore contains a ``mean_curr_width_mV`` column.

    Parameters
    ----------
    phys_df            : DataFrame from build_physics_features()
    cell_ids           : subset of cells to plot; defaults to all
    normalise_to_first : if True, divide each cell's widths by its first
                         non-NaN value (relative broadening from 1.0);
                         if False, plot raw mV values
    save_path          : save figure here and return path; None = return None

    Returns
    -------
    save_path if figure was saved, else None.

    Notes
    -----
    Because widths_mV are grid-spacing-invariant, curves from cells processed
    at different ICA grid resolutions are directly comparable on this plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "mean_curr_width_mV" not in phys_df.columns:
        raise ValueError(
            "phys_df must contain 'mean_curr_width_mV'. "
            "Re-run build_physics_features() with the fixed code."
        )

    if cell_ids is None:
        cell_ids = sorted(phys_df["cell_id"].unique())

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0

    for cell in cell_ids:
        sub = (
            phys_df[phys_df["cell_id"] == cell]
            .sort_values("cycle_number")
            .dropna(subset=["mean_curr_width_mV"])
        )
        if sub.empty:
            continue

        y = sub["mean_curr_width_mV"].to_numpy(float)
        if normalise_to_first:
            first = y[y > 0][0] if (y > 0).any() else 1.0
            y = y / first

        ax.plot(
            sub["cycle_number"].to_numpy(),
            y,
            label=str(cell),
            linewidth=1.2,
            alpha=0.85,
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    y_label = "Normalised peak width (1 = initial)" if normalise_to_first else "Mean ICA peak width (mV)"
    ax.set_xlabel("Cycle number")
    ax.set_ylabel(y_label)
    ax.set_title("ICA peak broadening evolution -- resistance growth proxy")
    ax.axhline(1.0 if normalise_to_first else float("nan"), color="grey", ls="--",
               lw=0.8, alpha=0.5, label="_nolegend_")
    ax.grid(True, alpha=0.3)
    if plotted <= 12:
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return save_path
    plt.close(fig)
    return None

# =========================================================
# 12. LINK TO ML FEATURES
# =========================================================

def merge_ml_and_physics(ml_df, phys_df):
    """
    Combine:
        - deltaQ(V) statistical ML features (from step3 / feature_schema)
        - physics ICA interpretation features (from build_physics_features)
    """
    return ml_df.merge(phys_df, on=["cell_id", "cycle_number"], how="inner")