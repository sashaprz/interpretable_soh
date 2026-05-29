"""
dataset_loaders.py
==================
Format-specific loaders for Oxford Battery Degradation (.mat) and
Severson et al. batch.pkl datasets.

Every loader returns list[CycleRecord] -- one record per (cell, cycle) -- in
canonical SI units:
    time      seconds
    voltage   volts
    current   amps  (positive = charge)
    capacity  Ah    (cumulative absolute, reset at start of each cycle)

Add new loaders via DatasetLoaderRegistry.register().

Note: step3_deltaQ(V)_feature_extraction.py also defines a CycleRecord for a
different purpose (half-cycle DataFrame + SOH label for feature extraction).
The CycleRecord here is the raw-data counterpart used before features are
computed.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("battery_pipeline.loaders")


# ---------------------------------------------------------------------------
# 1. Shared CycleRecord (raw, full-cycle)
# ---------------------------------------------------------------------------

@dataclass
class CycleRecord:
    """Format-agnostic single-cycle record in canonical SI units.

    Fields
    ------
    cell_id      : str   -- unique identifier (barcode, b1c0, ...)
    cycle_index  : int   -- preserved as-is from the source
    time         : 1-D float64, seconds
    voltage      : 1-D float64, volts
    current      : 1-D float64, amps  (positive = charge)
    capacity     : 1-D float64, Ah    (cumulative absolute from cycle start)
    temperature  : 1-D float64, degrees C  (None if unavailable)
    soh          : float  Qd / Qd_initial  (None if unknown)
    metadata     : dataset-specific tags (chemistry, protocol, dataset name)
    """
    cell_id: str
    cycle_index: int
    time: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    capacity: np.ndarray
    temperature: Optional[np.ndarray] = None
    soh: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2. MATLAB struct helpers
# ---------------------------------------------------------------------------

def _is_hdf5(path: Path) -> bool:
    """True if the file starts with the HDF5 magic bytes (MATLAB v7.3 .mat)."""
    with open(path, "rb") as fh:
        return fh.read(4) == b"\x89HDF"


def _mat_obj_to_python(obj: Any) -> Any:
    """Recursively convert scipy.io.loadmat output to plain Python / NumPy.

    Handles:
      - numpy structured arrays (dtype.names set)    -> dict or list[dict]
      - numpy object arrays (MATLAB cell arrays)     -> list or unwrapped scalar
      - plain ndarrays                               -> squeezed
    """
    if not isinstance(obj, np.ndarray):
        return obj
    if obj.dtype.names:                               # MATLAB struct
        if obj.size == 1:
            flat = obj.flat[0]
            return {n: _mat_obj_to_python(flat[n]) for n in obj.dtype.names}
        return [
            {n: _mat_obj_to_python(row[n]) for n in obj.dtype.names}
            for row in obj.flat
        ]
    if obj.dtype == object:                           # MATLAB cell array
        if obj.size == 1:
            return _mat_obj_to_python(obj.flat[0])
        return [_mat_obj_to_python(x) for x in obj.flat]
    arr = obj.squeeze()
    return arr.item() if arr.ndim == 0 else arr


def _hdf5_to_python(obj: Any) -> Any:
    """Recursively convert h5py Group/Dataset to plain Python / NumPy."""
    import h5py  # noqa: PLC0415
    if isinstance(obj, h5py.Dataset):
        data = obj[()]
        matlab_class = obj.attrs.get("MATLAB_class", b"")
        if isinstance(matlab_class, (bytes, np.bytes_)) and matlab_class == b"char":
            return "".join(chr(int(c)) for c in data.flat)
        return data.squeeze() if hasattr(data, "squeeze") else data
    if isinstance(obj, h5py.Group):
        return {k: _hdf5_to_python(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# 3. Oxford Battery Degradation Dataset loader
# ---------------------------------------------------------------------------

def _find_field(d: dict, *candidates: str) -> Optional[np.ndarray]:
    """Case-insensitive dict lookup; returns a 1-D float64 ndarray or None."""
    lower_map = {k.lower().strip(): k for k in d}
    for c in candidates:
        orig = lower_map.get(c.lower())
        if orig is None:
            continue
        v = d[orig]
        if isinstance(v, (int, float)):
            return np.array([float(v)], dtype=np.float64)
        if isinstance(v, (list, tuple)):
            try:
                return np.asarray(v, dtype=np.float64).ravel()
            except (ValueError, TypeError):
                continue
        if isinstance(v, np.ndarray):
            return v.ravel().astype(np.float64)
    return None


def _oxford_cell_to_records(cell_data: dict, cell_id: str) -> list[CycleRecord]:
    """Convert one flattened Oxford cell dict to CycleRecord objects.

    Tries two layouts:
      1. Flat arrays + cycle_index column  (most common)
      2. Nested per-cycle dicts under a "cycles" / "data" key
    """
    # --- layout 1: flat arrays ---
    time = _find_field(cell_data, "time", "time_s", "t", "test_time", "Time")
    voltage = _find_field(cell_data, "voltage", "v", "ewe", "Voltage", "V")
    current = _find_field(cell_data, "current", "i", "current_a", "Current", "I")
    capacity = _find_field(
        cell_data,
        "capacity", "q", "charge_capacity", "discharge_capacity",
        "Capacity", "Q", "q_charge", "q_discharge",
    )
    temperature = _find_field(cell_data, "temperature", "temp", "T", "t_cell", "Temperature")
    cycle_idx_arr = _find_field(
        cell_data,
        "cycle_index", "cycle_indicator", "cycle", "cycle_no",
        "Cycle_Index", "Cycle_Indicator", "cyc",
    )

    if time is not None and voltage is not None and current is not None:
        if cycle_idx_arr is None:
            cycle_idx_arr = np.zeros(len(time), dtype=np.int64)
        cycle_ids = np.unique(cycle_idx_arr.astype(np.int64))
        records: list[CycleRecord] = []
        for cid in cycle_ids:
            mask = cycle_idx_arr.astype(np.int64) == cid
            t_slice = time[mask]
            if capacity is not None:
                cap_slice = capacity[mask]
            else:
                dt = np.gradient(t_slice) if len(t_slice) > 1 else np.zeros(1)
                cap_slice = np.cumsum(np.abs(current[mask]) * dt / 3600.0)
            temp_slice = temperature[mask] if temperature is not None else None
            records.append(CycleRecord(
                cell_id=cell_id,
                cycle_index=int(cid),
                time=t_slice,
                voltage=voltage[mask],
                current=current[mask],
                capacity=cap_slice,
                temperature=temp_slice,
                metadata={"dataset": "oxford"},
            ))
        return records

    # --- layout 2: nested per-cycle dicts ---
    cycles_key = next(
        (k for k in cell_data if k.lower().strip() in ("cycles", "cycle_data", "data", "cyc")),
        None,
    )
    if cycles_key is not None:
        nested = cell_data[cycles_key]
        items: list[tuple[int, dict]] = []
        if isinstance(nested, dict):
            for k, v in nested.items():
                try:
                    items.append((int(k), v))
                except (ValueError, TypeError):
                    pass
        elif isinstance(nested, list):
            items = list(enumerate(nested))

        records = []
        for cid, cyc in items:
            if not isinstance(cyc, dict):
                continue
            t = _find_field(cyc, "time", "t", "Time")
            v = _find_field(cyc, "voltage", "v", "V", "Voltage")
            i = _find_field(cyc, "current", "i", "I", "Current")
            cap = _find_field(cyc, "capacity", "q", "Q", "Capacity")
            temp = _find_field(cyc, "temperature", "T", "Temperature", "temp")
            if t is None or v is None or i is None:
                continue
            if cap is None:
                dt = np.gradient(t) if len(t) > 1 else np.zeros(1)
                cap = np.cumsum(np.abs(i) * dt / 3600.0)
            records.append(CycleRecord(
                cell_id=cell_id,
                cycle_index=cid,
                time=t, voltage=v, current=i, capacity=cap,
                temperature=temp,
                metadata={"dataset": "oxford"},
            ))
        return records

    logger.warning("Oxford cell %s: no recognisable data layout; skipping.", cell_id)
    return []


def load_oxford_mat(path: str | Path) -> list[CycleRecord]:
    """Load Oxford Battery Degradation Dataset from a .mat file.

    Auto-detects MATLAB v7 (scipy.io) vs HDF5/v7.3 (h5py).
    Handles arbitrarily nested MATLAB struct hierarchies.

    Returns
    -------
    list[CycleRecord] sorted by (cell_id, cycle_index).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if _is_hdf5(path):
        import h5py  # noqa: PLC0415
        logger.info("Oxford .mat: HDF5 format (v7.3); using h5py.")
        with h5py.File(path, "r") as fh:
            raw: dict = _hdf5_to_python(fh)
    else:
        import scipy.io as sio  # noqa: PLC0415
        logger.info("Oxford .mat: MATLAB v7; using scipy.io.")
        try:
            raw = sio.loadmat(str(path), simplify_cells=True)
        except TypeError:
            # simplify_cells unavailable in older scipy
            raw = _mat_obj_to_python(sio.loadmat(str(path)))

    # Drop scipy meta-keys (__header__, __version__, __globals__)
    data = {k: v for k, v in raw.items() if not k.startswith("__")}

    records: list[CycleRecord] = []
    for top_key, top_val in data.items():
        if isinstance(top_val, list):
            for i, item in enumerate(top_val):
                if isinstance(item, dict):
                    records.extend(_oxford_cell_to_records(item, f"{top_key}_{i}"))
        elif isinstance(top_val, dict):
            # Either a single cell (values are arrays) or a dict-of-cells
            has_arrays = any(isinstance(v, np.ndarray) for v in top_val.values())
            if has_arrays:
                records.extend(_oxford_cell_to_records(top_val, top_key))
            else:
                for sub_key, sub_val in top_val.items():
                    cell_id = f"{top_key}_{sub_key}"
                    if isinstance(sub_val, dict):
                        records.extend(_oxford_cell_to_records(sub_val, cell_id))
                    elif isinstance(sub_val, list):
                        for i, item in enumerate(sub_val):
                            if isinstance(item, dict):
                                records.extend(_oxford_cell_to_records(item, f"{cell_id}_{i}"))
        elif isinstance(top_val, np.ndarray) and top_val.dtype.names:
            converted = _mat_obj_to_python(top_val)
            if isinstance(converted, dict):
                records.extend(_oxford_cell_to_records(converted, top_key))
            elif isinstance(converted, list):
                for i, item in enumerate(converted):
                    if isinstance(item, dict):
                        records.extend(_oxford_cell_to_records(item, f"{top_key}_{i}"))

    records.sort(key=lambda r: (r.cell_id, r.cycle_index))
    logger.info("Oxford loader: %d CycleRecords from %s", len(records), path.name)
    return records


# ---------------------------------------------------------------------------
# 4. Severson et al. 2019 batch.pkl loader
# ---------------------------------------------------------------------------

def load_severson_batch(
    path: str | Path,
    time_unit: str = "min",
) -> list[CycleRecord]:
    """Load the Severson et al. (Nature Energy 2019) batch.pkl dataset.

    Accepts batch as either ``list[cell_dict]`` or ``dict[str, cell_dict]``.

    Each cell_dict is expected to contain::

        cycles       : {str(cycle_num): {I, Qc, Qd, t, V, T}}
        summary      : {QDischarge, cycle, ...}  (used to derive SOH)
        charge_policy: str   (optional)
        barcode      : str   (optional, becomes cell_id)
        cathode      : str   (optional, e.g. "LFP")

    Input units:  time=minutes (configurable), current=A, capacity=Ah, V=V
    Output units: time=seconds, everything else unchanged.

    Parameters
    ----------
    path      : path to batch.pkl
    time_unit : unit of the "t" arrays inside the file ("min", "h", or "s")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "rb") as fh:
        batch = pickle.load(fh)

    if isinstance(batch, dict):
        cells: list[dict] = list(batch.values())
        cell_keys: list[str] = [str(k) for k in batch.keys()]
    elif isinstance(batch, list):
        cells = batch
        cell_keys = [str(i) for i in range(len(batch))]
    else:
        raise ValueError(f"Unexpected batch type: {type(batch).__name__!r}")

    time_scale = {"min": 60.0, "h": 3600.0, "s": 1.0}.get(time_unit, 60.0)

    records: list[CycleRecord] = []
    for key, cell in zip(cell_keys, cells):
        if not isinstance(cell, dict):
            logger.warning("Cell entry %r is not a dict; skipping.", key)
            continue

        cell_id = str(cell.get("barcode", key))
        protocol = cell.get("charge_policy") or cell.get("charge_protocol")
        chemistry = str(cell.get("cathode", "LFP"))

        # Build cycle -> SOH lookup from summary block
        soh_map: dict[int, float] = {}
        summary = cell.get("summary", {})
        q_dis = summary.get("QDischarge", summary.get("Qd"))
        cyc_nums = summary.get("cycle", summary.get("cycles"))
        if q_dis is not None and cyc_nums is not None:
            q_arr = np.asarray(q_dis, dtype=float).ravel()
            c_arr = np.asarray(cyc_nums, dtype=float).ravel()
            if q_arr.size > 0 and float(q_arr[0]) > 0.0:
                q0 = float(q_arr[0])
                for cn, qn in zip(c_arr, q_arr):
                    soh_map[int(cn)] = float(qn) / q0

        cycles_dict: dict = cell.get("cycles", {})
        for cycle_key, cyc_data in cycles_dict.items():
            try:
                cycle_num = int(cycle_key)
            except (ValueError, TypeError):
                logger.warning("Non-integer cycle key %r in cell %s; skipping.", cycle_key, cell_id)
                continue
            if not isinstance(cyc_data, dict):
                continue

            try:
                t = np.asarray(cyc_data["t"], dtype=float).ravel() * time_scale
                v = np.asarray(cyc_data["V"], dtype=float).ravel()
                i_arr = np.asarray(cyc_data["I"], dtype=float).ravel()
            except KeyError as exc:
                logger.warning("Cycle %d cell %s missing field %s; skipping.", cycle_num, cell_id, exc)
                continue

            qc = np.asarray(cyc_data.get("Qc", np.zeros_like(t)), dtype=float).ravel()
            qd = np.asarray(cyc_data.get("Qd", np.zeros_like(t)), dtype=float).ravel()
            # Positive current = charge -> use Qc; negative = discharge -> use Qd
            capacity = np.where(i_arr >= 0.0, qc, qd)

            raw_temp = cyc_data.get("T")
            if raw_temp is not None:
                temp = np.asarray(raw_temp, dtype=float).ravel()
                temperature: Optional[np.ndarray] = None if np.all(np.isnan(temp)) else temp
            else:
                temperature = None

            records.append(CycleRecord(
                cell_id=cell_id,
                cycle_index=cycle_num,
                time=t,
                voltage=v,
                current=i_arr,
                capacity=capacity,
                temperature=temperature,
                soh=soh_map.get(cycle_num),
                metadata={
                    "dataset": "severson",
                    "chemistry": chemistry,
                    "protocol": protocol,
                    "source_key": key,
                },
            ))

    records.sort(key=lambda r: (r.cell_id, r.cycle_index))
    logger.info("Severson loader: %d CycleRecords from %s", len(records), path.name)
    return records


# ---------------------------------------------------------------------------
# 5. DatasetLoaderRegistry
# ---------------------------------------------------------------------------

LoaderFn = Callable[..., list[CycleRecord]]


class DatasetLoaderRegistry:
    """Central registry mapping dataset names to loader callables.

    Each loader must accept ``(path, **kwargs)`` and return ``list[CycleRecord]``.

    Usage::

        # Explicit name
        records = DatasetLoaderRegistry.load("oxford", "path/to/data.mat")
        records = DatasetLoaderRegistry.load("severson", "path/to/batch.pkl")

        # Auto-detect by extension
        records = DatasetLoaderRegistry.auto_load("path/to/data.mat")

        # Register a third-party loader
        DatasetLoaderRegistry.register("maccor", my_loader, extensions=[".002"])
    """

    _loaders: dict[str, LoaderFn] = {}
    _ext_map: dict[str, str] = {}    # ".mat" -> "oxford", etc.

    @classmethod
    def register(
        cls,
        name: str,
        loader: LoaderFn,
        extensions: Optional[list[str]] = None,
    ) -> None:
        """Register *loader* under *name*, optionally binding file extensions."""
        cls._loaders[name] = loader
        for ext in extensions or []:
            cls._ext_map[ext.lower()] = name

    @classmethod
    def load(cls, name: str, path: str | Path, **kwargs) -> list[CycleRecord]:
        """Load by explicit dataset name."""
        if name not in cls._loaders:
            raise KeyError(
                f"Unknown dataset {name!r}. Registered: {sorted(cls._loaders)}"
            )
        return cls._loaders[name](path, **kwargs)

    @classmethod
    def auto_load(cls, path: str | Path, **kwargs) -> list[CycleRecord]:
        """Infer the loader from file extension; raises if extension is unknown."""
        ext = Path(path).suffix.lower()
        if ext not in cls._ext_map:
            raise ValueError(
                f"No loader registered for extension {ext!r}. "
                f"Known extensions: {sorted(cls._ext_map)}. "
                f"Use DatasetLoaderRegistry.load(name, path) to be explicit."
            )
        return cls.load(cls._ext_map[ext], path, **kwargs)

    @classmethod
    def registered(cls) -> list[str]:
        """Return sorted list of registered dataset names."""
        return sorted(cls._loaders)


# ---------------------------------------------------------------------------
# 6. Register built-in loaders
# ---------------------------------------------------------------------------

DatasetLoaderRegistry.register("oxford", load_oxford_mat, extensions=[".mat"])
DatasetLoaderRegistry.register("severson", load_severson_batch, extensions=[".pkl"])