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
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


# Prefer slow OCV discharge for ICA quality; fall back to 1C discharge.
_HALF_CYCLE_PREFERENCE = ["OCVdc", "C1dc", "OCVch", "C1ch"]


def convert_oxford_mat(
    mat_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """
    Read an Oxford Battery Degradation Dataset .mat file and write one CSV
    per cell into output_dir.

    Returns a list of Path objects for the written CSV files.
    """
    mat = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_keys = sorted(k for k in mat if k.startswith("Cell"))
    csv_paths: list[Path] = []

    for cell_key in cell_keys:
        cell_data = mat[cell_key]
        if not isinstance(cell_data, dict):
            continue

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

            # Pick the best available half-cycle
            hc = None
            for pref in _HALF_CYCLE_PREFERENCE:
                hc = cyc.get(pref)
                if hc is not None:
                    break
            if hc is None:
                continue

            t = np.asarray(hc["t"], dtype=float)   # MATLAB datenum, days
            v = np.asarray(hc["v"], dtype=float)   # V
            q = np.asarray(hc["q"], dtype=float)   # Ah (negative = discharge)
            T = np.asarray(hc["T"], dtype=float)   # °C

            if len(t) < 10:
                continue

            # Time in seconds from the start of this half-cycle
            t_s = (t - t[0]) * 86_400.0

            # Current: I = dq/dt  [Ah / h = A]
            dt_h = np.gradient(t) * 24.0       # dt in hours
            dq   = np.gradient(q)              # dq in Ah
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

        if not rows:
            continue

        df = pd.DataFrame(rows)
        csv_path = output_dir / f"{cell_key.lower()}_cycle.csv"
        df.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)

    return csv_paths


def is_mat(filename: str) -> bool:
    return Path(filename).suffix.lower() == ".mat"
