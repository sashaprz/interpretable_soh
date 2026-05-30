"""
mat_converter.py
----------------
Converts the Oxford Battery Degradation Dataset .mat file into per-cell CSVs
that the pipeline's step1 parser can ingest directly.

Oxford .mat structure
---------------------
  Cell1 / Cell2 / ... / Cell8
    cyc0000 / cyc0100 / cyc0200 / ...   (cycle snapshots, ~100-cycle intervals)
      C1ch   { t, v, q, T }   1C charge
      C1dc   { t, v, q, T }   1C discharge
      OCVch  { t, v, q, T }   slow OCV charge  (~C/25)
      OCVdc  { t, v, q, T }   slow OCV discharge (~C/25)  <-- best for ICA

  t : MATLAB datenum (days since Jan 0, 0000)
  v : voltage (V)
  q : capacity (Ah), 0 at start, negative during discharge
  T : temperature (°C)

Output CSV columns
------------------
  cycle_index, time (s), voltage (V), current (A), capacity (Ah), temperature (°C)

Memory strategy
---------------
Cells are loaded one at a time from the .mat file (scipy variable_names) and
OCV curves are downsampled to MAX_POINTS_PER_CYCLE before writing. The ICA step
resamples to its own voltage grid anyway, so the extra resolution is wasted RAM.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


_HALF_CYCLE_PREFERENCE = ["OCVdc", "C1dc", "OCVch", "C1ch"]

# Target points per cycle after downsampling. The ICA step resamples to a
# ~600-1100 point voltage grid (2 mV resolution over 3.0–5.2 V), so keeping
# more than this is pure overhead.
MAX_POINTS_PER_CYCLE = 1_500


def _cell_keys(mat_path: str | Path) -> list[str]:
    """Return sorted Cell* key names without loading cell data."""
    return sorted(
        name for name, _shape, _dtype in scipy.io.whosmat(str(mat_path))
        if name.startswith("Cell")
    )


def _convert_cell(cell_data: dict) -> list[dict]:
    """Extract rows from one cell's data dict."""
    rows: list[dict] = []

    for cyc_key in sorted(cell_data.keys()):
        if not cyc_key.startswith("cyc"):
            continue
        try:
            cycle_num = int(cyc_key[3:])
        except ValueError:
            continue

        cyc = cell_data[cyc_key]
        if not isinstance(cyc, dict):
            continue

        hc = None
        for pref in _HALF_CYCLE_PREFERENCE:
            hc = cyc.get(pref)
            if hc is not None:
                break
        if hc is None:
            continue

        t = np.asarray(hc["t"], dtype=float)
        v = np.asarray(hc["v"], dtype=float)
        q = np.asarray(hc["q"], dtype=float)
        T = np.asarray(hc["T"], dtype=float)

        if len(t) < 10:
            continue

        # Downsample to keep memory manageable on small cloud instances
        if len(t) > MAX_POINTS_PER_CYCLE:
            idx = np.round(np.linspace(0, len(t) - 1, MAX_POINTS_PER_CYCLE)).astype(int)
            t, v, q, T = t[idx], v[idx], q[idx], T[idx]

        t_s = (t - t[0]) * 86_400.0

        dt_h = np.gradient(t) * 24.0
        dq   = np.gradient(q)
        with np.errstate(invalid="ignore", divide="ignore"):
            current = np.where(np.abs(dt_h) > 1e-12, dq / dt_h, 0.0)

        for i in range(len(t)):
            rows.append({
                "cycle_index": cycle_num,
                "time":        t_s[i],
                "voltage":     v[i],
                "current":     current[i],
                "capacity":    q[i],
                "temperature": T[i],
            })

    return rows


def convert_oxford_mat(
    mat_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """
    Read an Oxford Battery Degradation Dataset .mat file and write one CSV
    per cell into output_dir. Cells are loaded one at a time to cap peak RAM.
    """
    mat_path   = Path(mat_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_keys = _cell_keys(mat_path)
    csv_paths: list[Path] = []

    for cell_key in cell_keys:
        # Load only this cell — avoids keeping all 8 in RAM simultaneously
        mat = scipy.io.loadmat(str(mat_path), simplify_cells=True,
                               variable_names=[cell_key])
        cell_data = mat.get(cell_key)
        del mat
        gc.collect()

        if not isinstance(cell_data, dict):
            continue

        rows = _convert_cell(cell_data)
        del cell_data
        gc.collect()

        if not rows:
            continue

        df = pd.DataFrame(rows)
        csv_path = output_dir / f"{cell_key.lower()}_cycle.csv"
        df.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)
        del df, rows
        gc.collect()

    return csv_paths


def is_mat(filename: str) -> bool:
    return Path(filename).suffix.lower() == ".mat"
