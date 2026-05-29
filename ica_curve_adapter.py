"""
ica_curve_adapter.py
Adapter between ICACurve objects and tabular DataFrames.

One row = one (cell, cycle) ICA curve.
Scalar metadata sits in typed columns; voltage_grid and dqdv are stored
as numpy float64 arrays in object-dtype columns.

Serialization:
    save/load_ica_pickle  — lossless, no extra dependencies
    save/load_ica_parquet — portable; arrays stored as little-endian float64 bytes
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------

# Scalar (non-array) columns in canonical order.
_SCALAR_COLS: List[str] = [
    "cell_id",
    "cycle_number",
    "capacity_ah",
    "is_reference",
    "ref_cycle_number",
    "temperature",
    "protocol",
    "grid_spacing_mv",   # mean voltage grid spacing in mV; used for width unit conversion
]

# Object-dtype columns holding numpy float64 arrays.
_ARRAY_COLS: List[str] = ["voltage_grid", "dqdv"]

ALL_ICA_COLS: List[str] = _SCALAR_COLS + _ARRAY_COLS

# Parquet uses these byte-blob columns in place of the array columns.
_PARQUET_BYTE_COLS: List[str] = ["_voltage_grid_bytes", "_dqdv_bytes"]


# ---------------------------------------------------------------------------
# ICACurve dataclass
# ---------------------------------------------------------------------------

@dataclass
class ICACurve:
    """One (cell, cycle) ICA curve on a shared voltage grid.

    voltage_grid and dqdv must have the same length and be float64.

    is_reference marks the per-cell reference cycle used for delta Q(V)
    computation in step3's DeltaQVComputer.

    ref_cycle_number records which cycle_number is the reference for this
    cell so the linkage survives round-trips through a DataFrame.
    """

    cell_id: str
    cycle_number: int
    voltage_grid: np.ndarray   # shape (N,), dtype float64
    dqdv: np.ndarray           # shape (N,), dtype float64
    capacity_ah: Optional[float] = None
    is_reference: bool = False
    ref_cycle_number: Optional[int] = None
    temperature: Optional[float] = None
    protocol: Optional[str] = None
    grid_spacing_mv: Optional[float] = None  # auto-computed from voltage_grid if None

    def __post_init__(self) -> None:
        self.voltage_grid = np.asarray(self.voltage_grid, dtype=np.float64)
        self.dqdv         = np.asarray(self.dqdv,         dtype=np.float64)
        if self.voltage_grid.ndim != 1 or self.dqdv.ndim != 1:
            raise ValueError("voltage_grid and dqdv must be 1-D arrays.")
        if len(self.voltage_grid) != len(self.dqdv):
            raise ValueError(
                f"voltage_grid length {len(self.voltage_grid)} != "
                f"dqdv length {len(self.dqdv)}."
            )
        if len(self.voltage_grid) == 0:
            raise ValueError("voltage_grid and dqdv must not be empty.")
        if self.grid_spacing_mv is None and len(self.voltage_grid) > 1:
            self.grid_spacing_mv = float(np.mean(np.diff(self.voltage_grid)) * 1000)


# ---------------------------------------------------------------------------
# Forward adapter: ICACurve → DataFrame
# ---------------------------------------------------------------------------

def ica_curve_to_dataframe(curves: Sequence[ICACurve]) -> pd.DataFrame:
    """Flatten a collection of ICACurve objects into a tabular DataFrame.

    Each row represents one (cell, cycle) ICA curve.  voltage_grid and dqdv
    are stored as numpy float64 arrays in object-dtype columns.

    Raises
    ------
    ValueError
        If the collection is empty, or if any two curves have voltage grids
        of different lengths (they must all share the same grid).
    """
    if not curves:
        raise ValueError("curves must not be empty.")

    grid_len = len(curves[0].voltage_grid)
    bad = [
        (c.cell_id, c.cycle_number, len(c.voltage_grid))
        for c in curves
        if len(c.voltage_grid) != grid_len
    ]
    if bad:
        raise ValueError(
            f"All voltage grids must have the same length ({grid_len} pts from "
            f"the first curve).  Mismatched curves (cell_id, cycle_number, length): {bad}"
        )

    rows = [
        {
            "cell_id":          c.cell_id,
            "cycle_number":     int(c.cycle_number),
            "capacity_ah":      float(c.capacity_ah)      if c.capacity_ah      is not None else np.nan,
            "is_reference":     bool(c.is_reference),
            "ref_cycle_number": int(c.ref_cycle_number)   if c.ref_cycle_number is not None else pd.NA,
            "temperature":      float(c.temperature)      if c.temperature      is not None else np.nan,
            "protocol":         c.protocol,
            "grid_spacing_mv":  float(c.grid_spacing_mv)  if c.grid_spacing_mv  is not None else np.nan,
            "voltage_grid":     c.voltage_grid.copy(),
            "dqdv":             c.dqdv.copy(),
        }
        for c in curves
    ]

    df = pd.DataFrame(rows)
    df["ref_cycle_number"] = df["ref_cycle_number"].astype("Int64")  # nullable int
    validate_ica_dataframe(df)
    return df


# ---------------------------------------------------------------------------
# Reverse adapter: DataFrame → ICACurve
# ---------------------------------------------------------------------------

def dataframe_to_ica_curves(df: pd.DataFrame) -> List[ICACurve]:
    """Reconstruct ICACurve objects from a DataFrame produced by ica_curve_to_dataframe.

    Also accepts DataFrames loaded from parquet via load_ica_parquet — the
    byte-blob columns are automatically decoded.
    """
    df = _decode_parquet_arrays(df)
    validate_ica_dataframe(df)

    curves: List[ICACurve] = []
    for _, row in df.iterrows():
        rcn = row["ref_cycle_number"]
        curves.append(ICACurve(
            cell_id          = str(row["cell_id"]),
            cycle_number     = int(row["cycle_number"]),
            voltage_grid     = np.asarray(row["voltage_grid"], dtype=np.float64),
            dqdv             = np.asarray(row["dqdv"],         dtype=np.float64),
            capacity_ah      = None if _is_missing(row["capacity_ah"])     else float(row["capacity_ah"]),
            is_reference     = bool(row["is_reference"]),
            ref_cycle_number = None if pd.isna(rcn)                         else int(rcn),
            temperature      = None if _is_missing(row["temperature"])      else float(row["temperature"]),
            protocol         = None if pd.isna(row["protocol"])             else str(row["protocol"]),
            grid_spacing_mv  = None if _is_missing(row.get("grid_spacing_mv", np.nan))
                               else float(row["grid_spacing_mv"]),
        ))
    return curves


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_ica_dataframe(df: pd.DataFrame) -> None:
    """Raise ValueError on any structural violation.

    Checks:
    - Required columns present
    - One row per (cell_id, cycle_number) pair
    - All voltage_grid arrays are float64 and have the same length
    - dqdv length matches voltage_grid length for every row
    - No NaN values inside the arrays
    """
    required = set(_SCALAR_COLS) | set(_ARRAY_COLS)
    missing  = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"ICA DataFrame is missing column(s): {missing}")

    dupes = df.duplicated(subset=["cell_id", "cycle_number"])
    if dupes.any():
        bad = df.loc[dupes, ["cell_id", "cycle_number"]].to_dict("records")
        raise ValueError(f"Duplicate (cell_id, cycle_number) rows: {bad}")

    grid_lengths = df["voltage_grid"].map(lambda a: len(np.asarray(a)))
    if grid_lengths.nunique() != 1:
        raise ValueError(
            f"voltage_grid lengths are not uniform: {sorted(grid_lengths.unique().tolist())}"
        )

    expected_len = int(grid_lengths.iloc[0])
    bad_dqdv = df[df["dqdv"].map(lambda a: len(np.asarray(a))) != expected_len]
    if not bad_dqdv.empty:
        raise ValueError(
            f"dqdv length != voltage_grid length ({expected_len}) for "
            f"{len(bad_dqdv)} row(s): "
            + bad_dqdv[["cell_id", "cycle_number"]].to_string(index=False)
        )

    for col in _ARRAY_COLS:
        has_nan = df[col].map(
            lambda a: bool(np.any(~np.isfinite(np.asarray(a, dtype=np.float64))))
        )
        n_nan = int(has_nan.sum())
        if n_nan:
            raise ValueError(
                f"{n_nan} row(s) have NaN/Inf values in '{col}'."
            )


# ---------------------------------------------------------------------------
# Serialization — pickle
# ---------------------------------------------------------------------------

def save_ica_pickle(df: pd.DataFrame, path: str | Path) -> Path:
    """Save ICA DataFrame to a pickle file. Arrays preserved exactly."""
    validate_ica_dataframe(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_ica_pickle(path: str | Path) -> pd.DataFrame:
    """Load and validate an ICA DataFrame from a pickle file."""
    with open(Path(path), "rb") as fh:
        df = pickle.load(fh)
    validate_ica_dataframe(df)
    return df


# ---------------------------------------------------------------------------
# Serialization — parquet
# ---------------------------------------------------------------------------

def save_ica_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Save ICA DataFrame to a parquet file.

    Arrays are encoded as little-endian float64 bytes; a _grid_length column
    is written so load_ica_parquet can reconstruct them unambiguously.
    """
    validate_ica_dataframe(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat = df[_SCALAR_COLS].copy()
    flat["_grid_length"] = int(len(df["voltage_grid"].iloc[0]))
    flat["_voltage_grid_bytes"] = df["voltage_grid"].map(
        lambda a: np.asarray(a, dtype="<f8").tobytes()
    )
    flat["_dqdv_bytes"] = df["dqdv"].map(
        lambda a: np.asarray(a, dtype="<f8").tobytes()
    )
    flat.to_parquet(path, index=False)
    return path


def load_ica_parquet(path: str | Path) -> pd.DataFrame:
    """Load and validate an ICA DataFrame from a parquet file."""
    flat = pd.read_parquet(Path(path))
    df   = _decode_parquet_arrays(flat)
    validate_ica_dataframe(df)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_parquet_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """Convert _*_bytes columns back to numpy arrays.  No-op if already decoded."""
    if "_voltage_grid_bytes" not in df.columns:
        return df

    grid_len = int(df["_grid_length"].iloc[0])
    df = df.copy()
    df["voltage_grid"] = df["_voltage_grid_bytes"].map(
        lambda b: np.frombuffer(b, dtype="<f8").reshape(grid_len).copy()
    )
    df["dqdv"] = df["_dqdv_bytes"].map(
        lambda b: np.frombuffer(b, dtype="<f8").reshape(grid_len).copy()
    )
    df = df.drop(columns=["_grid_length", "_voltage_grid_bytes", "_dqdv_bytes"],
                 errors="ignore")
    return df


def _is_missing(val) -> bool:
    """True if val is None, NaN, or pandas NA."""
    try:
        return val is None or pd.isna(val)
    except (TypeError, ValueError):
        return False
