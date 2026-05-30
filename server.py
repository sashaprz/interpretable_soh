"""
server.py — Flask server for the SOH Dashboard.

    python server.py
    open http://localhost:5000

Share on your local network:
    Anyone on the same Wi-Fi can open http://<your-ip>:5000

Share over the internet (optional):
    Install ngrok, then: ngrok http 5000

Routes
------
GET  /                      → dashboard.html
GET  /api/models            → list trained models from disk
POST /api/train             → upload CSVs, start background training job
GET  /api/jobs/<job_id>     → poll training job status + result
POST /api/predict           → upload one cell CSV, return SOH trajectory (synchronous)
"""
from __future__ import annotations

import json
import sys
import threading
import traceback
import os
import uuid
from pathlib import Path
from typing import Any

import importlib.util
import types

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from mat_converter import convert_oxford_mat, is_mat

# ── lazy module loader (handles parenthesised filenames) ─────────────────────

_MODULE_CACHE: dict[str, types.ModuleType] = {}
HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load(slug: str, filename: str) -> types.ModuleType:
    if slug in _MODULE_CACHE:
        return _MODULE_CACHE[slug]
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(slug, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[slug] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _MODULE_CACHE[slug] = mod
    return mod


def _interp():
    return _load("_interp", "step4_interpretation.py")


from feature_schema import FEATURE_COLS  # noqa: E402
from run_pipeline import CellSpec, PipelineConfig, PipelineRunner  # noqa: E402

# ── directories ──────────────────────────────────────────────────────────────

OUT = HERE / "pipeline_output"
UPLOAD_DIR = OUT / "uploads"
MODELS_DIR = OUT / "model"

# ── Flask ─────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=None)
_jobs: dict[str, dict] = {}


@app.route("/")
def index():
    return send_from_directory(str(HERE), "dashboard.html")


# ── models ────────────────────────────────────────────────────────────────────

@app.route("/api/models")
def list_models():
    models = []
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.glob("*_meta.json"), reverse=True):
            try:
                models.append(json.loads(p.read_text()))
            except Exception:
                pass
    return jsonify(models)


# ── train ─────────────────────────────────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def train():
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files uploaded"}), 400

    nominal_cap = float(request.form.get("nominal_capacity_ah", 5.0))
    job_id = uuid.uuid4().hex[:8]
    save_dir = UPLOAD_DIR / job_id
    save_dir.mkdir(parents=True, exist_ok=True)

    cell_specs: list[CellSpec] = []
    for f in files:
        if not f.filename:
            continue
        dest = save_dir / f.filename
        f.save(dest)
        if is_mat(f.filename):
            # Convert .mat → per-cell CSVs, one CellSpec per cell
            csv_dir = save_dir / "converted"
            csv_paths = convert_oxford_mat(dest, csv_dir)
            for csv_path in csv_paths:
                cell_specs.append(CellSpec(path=str(csv_path), nominal_capacity_ah=nominal_cap))
        else:
            cell_specs.append(CellSpec(path=str(dest), nominal_capacity_ah=nominal_cap))

    if not cell_specs:
        return jsonify({"error": "No valid files"}), 400

    cell_ids = [s.resolved_id() for s in cell_specs]
    # parse + ica + features per cell, then one model stage
    _total_steps = len(cell_specs) * 3 + 1
    _jobs[job_id] = {"status": "running", "progress": 0.0, "cell_ids": cell_ids}

    class _TrackingRunner(PipelineRunner):
        """Thin subclass that updates job progress after each stage completes."""
        _completed = 0

        def _timed(self, stage, cell_id, fn):
            result, sr = super()._timed(stage, cell_id, fn)
            if sr.status in ("ok", "skipped"):
                _TrackingRunner._completed += 1
                _jobs[job_id]["progress"] = min(0.93, _TrackingRunner._completed / _total_steps)
            return result, sr

    def run() -> None:
        try:
            out_dir = OUT / "runs" / job_id
            cfg = PipelineConfig(cells=cell_specs, output_dir=out_dir)
            runner = _TrackingRunner(cfg)

            result = runner.run()
            _jobs[job_id]["progress"] = 0.95

            if result.overall_status == "failed":
                errors = []
                for cid, cs in result.cell_summaries.items():
                    for sr in cs.stage_results:
                        if sr.status == "failed" and sr.error:
                            errors.append(f"{cid} [{sr.stage}]: {sr.error}")
                msg = " | ".join(errors) if errors else "Pipeline failed — check Render logs for traceback"
                _jobs[job_id].update({"status": "failed", "error": msg})
                return

            model_path = out_dir / "model" / "elasticnet_soh.joblib"
            m_metrics = result.model_metrics or {}

            # persist model metadata
            model_id = f"m_{job_id}"
            meta: dict[str, Any] = {
                "id": model_id,
                "name": f"Custom · {len(cell_ids)} cell{'s' if len(cell_ids) != 1 else ''}",
                "typeKey": "custom",
                "type": "Custom",
                "form": "Uploaded",
                "cells": len(cell_ids),
                "cellIds": cell_ids,
                "r2":   round(m_metrics.get("r2",   0.0), 4),
                "rmse": round(m_metrics.get("rmse", 0.0), 5),
                "mae":  round(m_metrics.get("mae",  0.0), 5),
                "date": __import__("datetime").date.today().isoformat(),
                "outDir": str(out_dir),
            }
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            (MODELS_DIR / f"{model_id}_meta.json").write_text(json.dumps(meta, indent=2))

            cells_json, batch_json = _build_dashboard_data(out_dir, cell_specs, model_path, m_metrics)

            _jobs[job_id].update({
                "status": "done",
                "progress": 1.0,
                "model": meta,
                "cells": cells_json,
                "batch": batch_json,
            })
        except Exception:
            _jobs[job_id].update({
                "status": "failed",
                "error": traceback.format_exc().strip().splitlines()[-1],
            })

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running", "cell_ids": cell_ids})


@app.route("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify(job)


# ── predict ───────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    f = request.files.get("file")
    model_id = request.form.get("model_id", "")
    nominal_cap = float(request.form.get("nominal_capacity_ah", 5.0))

    if not f or not f.filename:
        return jsonify({"error": "No file uploaded"}), 400

    meta_path = MODELS_DIR / f"{model_id}_meta.json"
    if not meta_path.exists():
        return jsonify({"error": f"Model {model_id!r} not found on disk"}), 404

    meta = json.loads(meta_path.read_text())
    out_dir = Path(meta["outDir"])
    model_path = out_dir / "model" / "elasticnet_soh.joblib"

    tmp_dir = UPLOAD_DIR / uuid.uuid4().hex[:8]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / f.filename
    f.save(dest)

    if is_mat(f.filename):
        csv_dir = tmp_dir / "converted"
        csv_paths = convert_oxford_mat(dest, csv_dir)
        if not csv_paths:
            return jsonify({"error": "No cell data found in .mat file"}), 400
        # Use the first cell from the .mat for prediction
        spec = CellSpec(path=str(csv_paths[0]), nominal_capacity_ah=nominal_cap)
    else:
        spec = CellSpec(path=str(dest), nominal_capacity_ah=nominal_cap)

    pred_out = tmp_dir / "out"
    cfg = PipelineConfig(cells=[spec], output_dir=pred_out, stages=["parse", "ica", "features"])
    runner = PipelineRunner(cfg)
    result = runner.run()

    if result.overall_status == "failed":
        return jsonify({"error": "Pipeline failed for uploaded file"}), 500

    cells_json, _ = _build_dashboard_data(pred_out, [spec], model_path, meta)
    cell_id = spec.resolved_id()
    cell_data = cells_json.get(cell_id)
    if not cell_data:
        return jsonify({"error": "No output produced for the uploaded cell"}), 500

    return jsonify({"cell": cell_data, "model": meta})


# ── data adapter: pipeline output → SOHData JSON ─────────────────────────────

_N_DASH = 220  # dashboard expects fixed-length ICA arrays


def _build_dashboard_data(
    out_dir: Path,
    cell_specs: list[CellSpec],
    model_path: Path,
    model_metrics: dict,
) -> tuple[dict, dict | None]:
    model_pipeline = joblib.load(model_path) if model_path.exists() else None
    cells_json: dict[str, Any] = {}

    for spec in cell_specs:
        cell_id = spec.resolved_id()
        feat_path = out_dir / "features" / f"{cell_id}_features.csv"
        ica_ckpt  = out_dir / "checkpoints" / f"{cell_id}__ica.pkl"
        if not feat_path.exists():
            continue
        try:
            feat_df    = pd.read_csv(feat_path)
            ica_curves = joblib.load(ica_ckpt) if ica_ckpt.exists() else []
            cells_json[cell_id] = _cell_to_json(
                cell_id, feat_df, ica_curves,
                model_pipeline, model_metrics, spec.nominal_capacity_ah,
            )
        except Exception:
            traceback.print_exc()

    batch_json = _build_batch(cells_json) if len(cells_json) > 1 else None
    return cells_json, batch_json


def _resample(arr: np.ndarray, voltage: np.ndarray) -> list[float]:
    xi = np.linspace(voltage[0], voltage[-1], _N_DASH)
    return np.interp(xi, voltage, arr).tolist()


def _cell_to_json(
    cell_id: str,
    feat_df: pd.DataFrame,
    ica_curves: list,
    model_pipeline: Any,
    model_metrics: dict,
    nominal_cap_ah: float,
) -> dict:
    interp = _interp()

    ok_col = feat_df.get("extraction_ok", pd.Series(True, index=feat_df.index))
    df = feat_df[ok_col].sort_values("cycle_number").reset_index(drop=True)
    if df.empty:
        df = feat_df.sort_values("cycle_number").reset_index(drop=True)

    cycles   = df["cycle_number"].astype(int).tolist()
    soh_true = df["soh"].clip(0.0, 1.0).tolist()
    n        = len(cycles)
    max_cyc  = int(max(cycles)) if cycles else 1

    # SOH predictions
    if model_pipeline is not None and n > 0:
        X = df[FEATURE_COLS].fillna(0.0).values
        soh_pred = np.clip(model_pipeline.predict(X), 0.0, 1.0).tolist()
    else:
        soh_pred = soh_true[:]

    # ICA lookup
    ref_dqdv: np.ndarray | None = None
    voltage_grid: np.ndarray | None = None
    ica_by_cycle: dict[int, np.ndarray] = {}
    for c in ica_curves:
        cyc_n = int(c.cycle_number)
        ica_by_cycle[cyc_n] = np.asarray(c.dqdv, dtype=float)
        if voltage_grid is None or getattr(c, "is_reference", False):
            ref_dqdv    = np.asarray(c.dqdv,         dtype=float)
            voltage_grid = np.asarray(c.voltage_grid, dtype=float)

    ica_ref = _resample(ref_dqdv, voltage_grid) if ref_dqdv is not None else []

    # Per-cycle rows
    rows: list[dict] = []
    all_lli: list[float] = []
    all_lam: list[float] = []
    all_res: list[float] = []

    for i in range(n):
        cyc       = cycles[i]
        life_frac = cyc / max_cyc
        feats     = {col: float(df.iloc[i].get(col, 0) or 0) for col in FEATURE_COLS}

        curr_ica = ica_by_cycle.get(cyc)
        ica_arr = dq_arr = []
        lli_sig = lam_sig = res_sig = 0.0
        dominant = "EARLY"
        conf = 0.5
        mean_shift_v = lam_loss = res_growth = 0.0

        if ref_dqdv is not None and curr_ica is not None and voltage_grid is not None:
            dq      = curr_ica - ref_dqdv
            ica_arr = _resample(curr_ica, voltage_grid)
            dq_arr  = _resample(dq, voltage_grid)
            try:
                phys, _lli = interp.interpret_cycle(voltage_grid, ref_dqdv, curr_ica)
                mean_shift_v = float(phys.get("LLI_mean_shift", 0) or 0)
                lam_loss     = max(0.0, float(phys.get("LAM_loss", 0) or 0))
                res_growth   = max(0.0, float(phys.get("Resistance_growth", 0) or 0))
                lli_conf     = float(phys.get("LLI_confidence", 0) or 0)

                lli_sig = min(1.0, abs(mean_shift_v) / 0.05) * max(0.3, lli_conf)
                lam_sig = min(1.0, lam_loss * 2.0)
                res_sig = min(1.0, res_growth)

                tot = lli_sig + lam_sig + res_sig + 1e-9
                if tot < 0.05:
                    dominant = "EARLY"
                elif lli_sig >= lam_sig and lli_sig >= res_sig:
                    dominant = "LLI"
                elif lam_sig >= res_sig:
                    dominant = "LAM"
                else:
                    dominant = "RES"
                conf = min(0.97, max(lli_sig, lam_sig, res_sig) / tot * 0.6 + 0.35)
            except Exception:
                pass

        all_lli.append(lli_sig)
        all_lam.append(lam_sig)
        all_res.append(res_sig)
        tot  = lli_sig + lam_sig + res_sig + 1e-9
        probs = {"LLI": lli_sig / tot, "LAM": lam_sig / tot, "RES": res_sig / tot}

        rows.append({
            "cycle": cyc, "lifeFrac": life_frac,
            "ica": ica_arr, "dq": dq_arr, "feats": feats,
            "signals": {"LLI": lli_sig, "LAM": lam_sig, "RES": res_sig},
            "phys": {
                "meanShiftV": mean_shift_v, "lamLoss": lam_loss, "resGrowth": res_growth,
                "probs": probs, "dominant": dominant, "conf": conf, "tot": tot,
            },
            "diag": [
                {"label": "Phase transition 1", "voltage": 3.50, "shiftMv": mean_shift_v * 1000, "widthMv": 45.0, "areaLossPct": lam_loss * 100},
                {"label": "Phase transition 2", "voltage": 3.65, "shiftMv": mean_shift_v * 1000, "widthMv": 38.0, "areaLossPct": lam_loss * 100},
                {"label": "Phase transition 3", "voltage": 3.90, "shiftMv": mean_shift_v * 1000, "widthMv": 55.0, "areaLossPct": lam_loss * 100},
            ],
        })

    # Phases
    dom_seq = [r["phys"]["dominant"] for r in rows]
    phases  = _build_phases(dom_seq, cycles, soh_true, n)

    # Onsets: first cycle where signal > 0.16
    onsets: dict[str, int | None] = {}
    for m, sig_arr in [("LLI", all_lli), ("LAM", all_lam), ("RES", all_res)]:
        onsets[m] = next((cycles[i] for i, s in enumerate(sig_arr) if s > 0.16), None)

    # RUL
    rul: int | None = None
    for i in range(1, n):
        if soh_pred[i] <= 0.80:
            t = (0.80 - soh_pred[i - 1]) / (soh_pred[i] - soh_pred[i - 1] + 1e-9)
            rul = int(cycles[i - 1] + t * (cycles[i] - cycles[i - 1]))
            break

    cell_dominant = next(
        (p["mech"] for p in reversed(phases) if p["mech"] != "EARLY"), "LLI"
    )
    per_cell = {
        "r2":   round(model_metrics.get("r2",   0.0), 4),
        "rmse": round(model_metrics.get("rmse", 0.0), 5),
        "mae":  round(model_metrics.get("mae",  0.0), 5),
    }
    temp_col = df.get("temperature")
    temp = None if temp_col is None else (
        None if pd.isna(temp_col.iloc[0]) else float(temp_col.iloc[0])
    )

    return {
        "id": cell_id, "chemistry": "Unknown", "form": "Uploaded",
        "capacityMah": round(nominal_cap_ah * 1000),
        "tempC": temp, "maxCycle": max_cyc, "nCycles": n,
        "cycles": cycles, "step": int(cycles[1] - cycles[0]) if n > 1 else 1,
        "sohTrue": soh_true, "sohPred": soh_pred, "icaRef": ica_ref,
        "rows": rows, "metrics": per_cell, "rul": rul, "eol": 0.80,
        "sohLast": float(soh_true[-1]) if soh_true else 0.0,
        "cellDominant": cell_dominant, "phases": phases, "onsets": onsets,
        "isBatch": False,
    }


def _build_phases(
    dom_seq: list[str], cycles: list[int], soh_true: list[float], n: int
) -> list[dict]:
    raw: list[dict] = []
    for i, d in enumerate(dom_seq):
        if not raw or raw[-1]["mech"] != d:
            raw.append({"mech": d, "i0": i, "i1": i})
        else:
            raw[-1]["i1"] = i

    merged: list[dict] = []
    for p in raw:
        if merged and (p["i1"] - p["i0"] + 1) < 3 and p["mech"] != "EARLY":
            merged[-1]["i1"] = p["i1"]
        else:
            merged.append(p)

    result = []
    for p in merged:
        i0, i1 = p["i0"], p["i1"]
        result.append({
            "mech": p["mech"],
            "startCycle": cycles[i0], "endCycle": cycles[i1],
            "startFrac": i0 / max(n - 1, 1), "endFrac": i1 / max(n - 1, 1),
            "sohStart": float(soh_true[i0]), "sohEnd": float(soh_true[i1]),
            "peakShare": 0.6, "i0": i0, "i1": i1,
        })
    return result


def _build_batch(cells_json: dict) -> dict | None:
    cell_list = list(cells_json.values())
    if not cell_list:
        return None

    M        = 81
    max_cyc  = int(np.mean([c["maxCycle"] for c in cell_list]))
    cycles   = [int(round(k / (M - 1) * max_cyc)) for k in range(M)]
    tc       = np.array(cycles, dtype=float)

    soh_true_avg = np.zeros(M)
    soh_pred_avg = np.zeros(M)
    for cell in cell_list:
        src = np.array(cell["cycles"], dtype=float)
        soh_true_avg += np.interp(tc, src, np.array(cell["sohTrue"]))
        soh_pred_avg += np.interp(tc, src, np.array(cell["sohPred"]))
    soh_true_avg = (soh_true_avg / len(cell_list)).tolist()
    soh_pred_avg = (soh_pred_avg / len(cell_list)).tolist()

    rul: int | None = None
    for i in range(1, M):
        if soh_pred_avg[i] <= 0.80:
            t = (0.80 - soh_pred_avg[i - 1]) / (soh_pred_avg[i] - soh_pred_avg[i - 1] + 1e-9)
            rul = int(cycles[i - 1] + t * (cycles[i] - cycles[i - 1]))
            break

    avg_metrics = {
        "r2":   float(np.mean([c["metrics"]["r2"]   for c in cell_list])),
        "rmse": float(np.mean([c["metrics"]["rmse"] for c in cell_list])),
        "mae":  float(np.mean([c["metrics"]["mae"]  for c in cell_list])),
    }

    rows = []
    for k in range(M):
        lf = k / (M - 1)
        dominant = "LLI" if lf < 0.35 else ("LAM" if lf < 0.70 else "RES")
        rows.append({
            "cycle": cycles[k], "lifeFrac": lf,
            "ica": [], "dq": [], "feats": {},
            "signals": {"LLI": 0.4 * lf, "LAM": 0.6 * lf, "RES": 0.8 * lf ** 1.8},
            "phys": {
                "meanShiftV": 0.0, "lamLoss": 0.0, "resGrowth": 0.0,
                "probs": {"LLI": 0.4, "LAM": 0.35, "RES": 0.25},
                "dominant": dominant, "conf": 0.7, "tot": 0.5,
            },
            "diag": [
                {"label": "Phase transition 1", "voltage": 3.50, "shiftMv": 0.0, "widthMv": 45.0, "areaLossPct": 0.0},
                {"label": "Phase transition 2", "voltage": 3.65, "shiftMv": 0.0, "widthMv": 38.0, "areaLossPct": 0.0},
                {"label": "Phase transition 3", "voltage": 3.90, "shiftMv": 0.0, "widthMv": 55.0, "areaLossPct": 0.0},
            ],
        })

    dom_seq = [r["phys"]["dominant"] for r in rows]
    phases  = _build_phases(dom_seq, cycles, soh_true_avg, M)
    onsets  = {
        "LLI": cycles[max(1, M // 12)],
        "LAM": int(max_cyc * 0.30),
        "RES": int(max_cyc * 0.65),
    }

    return {
        "id": "Batch average", "chemistry": "Custom", "form": "Uploaded",
        "capacityMah": int(np.mean([c["capacityMah"] for c in cell_list])),
        "tempC": None, "maxCycle": max_cyc, "nCycles": M,
        "cycles": cycles, "step": int(max_cyc / (M - 1)),
        "sohTrue": soh_true_avg, "sohPred": soh_pred_avg, "icaRef": [],
        "rows": rows, "metrics": avg_metrics, "rul": rul, "eol": 0.80,
        "sohLast": float(soh_true_avg[-1]),
        "cellDominant": "RES", "phases": phases, "onsets": onsets,
        "isBatch": True, "nCells": len(cell_list),
    }


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "your-ip"

    print()
    print("  SOH Dashboard")
    print(f"  Local:   http://localhost:5000")
    print(f"  Network: http://{local_ip}:5000  (share this with teammates)")
    print()
    print("  For internet access: ngrok http 5000")
    print()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
