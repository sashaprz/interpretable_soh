"""
Tests for run_pipeline.py — config loading, stage wiring, checkpoint behaviour,
summary JSON shape, and error isolation.  Uses only synthetic data; no real cycler
files required.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from run_pipeline import (
    CellSpec,
    PipelineConfig,
    PipelineRunner,
    StageResult,
    _build_cycle_records,
    _ckpt,
    _load_step,
    config_from_cli,
    load_config,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content))
    return path


def _dummy_csv(path: Path, n_rows: int = 10) -> Path:
    """Write a minimal CSV that exists on disk (content doesn't matter for config tests)."""
    pd.DataFrame({"a": range(n_rows)}).to_csv(path, index=False)
    return path


def _fake_features(cell_id: str, n: int = 5) -> pd.DataFrame:
    """Synthetic features DataFrame matching the feature_schema contract."""
    from feature_schema import FEATURE_COLS
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        row = {"cell_id": cell_id, "cycle_number": i, "soh": 1.0 - 0.01 * i,
               "extraction_ok": True, "temperature": 25.0, "protocol": "test",
               "ref_cycle_index": 0, "config_hash": "abc123"}
        row.update({col: rng.normal() for col in FEATURE_COLS})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_explicit_cells(tmp_path):
    _dummy_csv(tmp_path / "cell_01.csv")
    _dummy_csv(tmp_path / "cell_02.csv")
    cfg_yaml = _write_yaml(tmp_path / "cfg.yaml", f"""
        dataset:
          cells:
            cell_01:
              path: {tmp_path / "cell_01.csv"}
              nominal_capacity_ah: 5.0
            cell_02:
              path: {tmp_path / "cell_02.csv"}
              nominal_capacity_ah: 4.8
        output_dir: {tmp_path / "out"}
    """)
    cfg = load_config(cfg_yaml)
    assert len(cfg.cells) == 2
    assert cfg.cells[0].cell_id == "cell_01"
    assert cfg.cells[1].nominal_capacity_ah == pytest.approx(4.8)


def test_load_config_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _dummy_csv(data_dir / "a.csv")
    _dummy_csv(data_dir / "b.csv")
    cfg_yaml = _write_yaml(tmp_path / "cfg.yaml", f"""
        dataset:
          dir: {data_dir}
          nominal_capacity_ah: 3.0
        output_dir: {tmp_path / "out"}
    """)
    cfg = load_config(cfg_yaml)
    assert len(cfg.cells) == 2
    assert all(c.nominal_capacity_ah == pytest.approx(3.0) for c in cfg.cells)


def test_load_config_missing_capacity_raises(tmp_path):
    _dummy_csv(tmp_path / "c.csv")
    cfg_yaml = _write_yaml(tmp_path / "cfg.yaml", f"""
        dataset:
          cells:
            c: {{path: {tmp_path / "c.csv"}}}
        output_dir: {tmp_path / "out"}
    """)
    with pytest.raises(ValueError, match="nominal_capacity_ah"):
        load_config(cfg_yaml)


def test_load_config_pipeline_section(tmp_path):
    _dummy_csv(tmp_path / "c.csv")
    cfg_yaml = _write_yaml(tmp_path / "cfg.yaml", f"""
        dataset:
          cells:
            c:
              path: {tmp_path / "c.csv"}
              nominal_capacity_ah: 2.0
        output_dir: {tmp_path / "out"}
        pipeline:
          resume: false
          stages: [parse, features]
    """)
    cfg = load_config(cfg_yaml)
    assert cfg.resume is False
    assert cfg.stages == ["parse", "features"]


def test_load_config_cli_override(tmp_path):
    _dummy_csv(tmp_path / "c.csv")
    cfg_yaml = _write_yaml(tmp_path / "cfg.yaml", f"""
        dataset:
          cells:
            c:
              path: {tmp_path / "c.csv"}
              nominal_capacity_ah: 2.0
        output_dir: {tmp_path / "out"}
    """)
    cfg = load_config(cfg_yaml, output_dir=str(tmp_path / "override_out"))
    assert cfg.output_dir == tmp_path / "override_out"


# ---------------------------------------------------------------------------
# config_from_cli
# ---------------------------------------------------------------------------

def test_config_from_cli_single_file(tmp_path):
    f = _dummy_csv(tmp_path / "cell.csv")
    cfg = config_from_cli(str(f), str(tmp_path / "out"), 5.0)
    assert len(cfg.cells) == 1
    assert cfg.cells[0].nominal_capacity_ah == pytest.approx(5.0)


def test_config_from_cli_directory(tmp_path):
    for name in ["a.csv", "b.csv", "c.csv"]:
        _dummy_csv(tmp_path / name)
    cfg = config_from_cli(str(tmp_path), str(tmp_path / "out"), 5.0)
    assert len(cfg.cells) == 3


def test_config_from_cli_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="No .csv"):
        config_from_cli(str(tmp_path), str(tmp_path / "out"), 5.0)


# ---------------------------------------------------------------------------
# CellSpec.resolved_id
# ---------------------------------------------------------------------------

def test_cell_spec_resolved_id_explicit():
    s = CellSpec(path="data/foo.csv", nominal_capacity_ah=5.0, cell_id="my_cell")
    assert s.resolved_id() == "my_cell"


def test_cell_spec_resolved_id_from_stem():
    s = CellSpec(path="data/cell_42.csv", nominal_capacity_ah=5.0)
    assert s.resolved_id() == "cell_42"


# ---------------------------------------------------------------------------
# Checkpoint path helper
# ---------------------------------------------------------------------------

def test_ckpt_path_structure(tmp_path):
    p = _ckpt(tmp_path, "cell_01", "parse")
    assert p.parent.name == "checkpoints"
    assert "cell_01" in p.name and "parse" in p.name


# ---------------------------------------------------------------------------
# StageResult
# ---------------------------------------------------------------------------

def test_stage_result_defaults():
    sr = StageResult(stage="parse", status="ok", elapsed_s=1.23)
    assert sr.error is None
    assert sr.artifact is None
    assert sr.resumed is False


# ---------------------------------------------------------------------------
# _build_cycle_records
# ---------------------------------------------------------------------------

def test_build_cycle_records_from_synthetic_parsed():
    """Verify bridge from ParsedDataset-like object to CycleRecord list."""
    _s3_mod = _load_step("_step3", "step3_deltaQ(V)_feature_extraction.py")
    rng = np.random.default_rng(1)

    def _half(n=50):
        t = np.linspace(0, 3600, n)
        v = np.linspace(4.1, 2.6, n)
        i = np.full(n, -1.0)
        return pd.DataFrame({"time": t, "voltage": v, "current": i})

    # Build a mock ParsedDataset
    cycles = {}
    for cid in range(3):
        cycles[cid] = {
            "discharge": _half(),
            "charge":    _half(),
            "meta": {"discharge_capacity_ah": 4.5, "charge_capacity_ah": 4.6},
        }

    class FakeParsed:
        pass

    fp = FakeParsed()
    fp.cycles = cycles

    records = _build_cycle_records(fp, "test_cell", nominal_capacity_ah=5.0, half_cycle="discharge")
    assert len(records) == 3
    assert all(r.cell_id == "test_cell" for r in records)
    assert all(abs(r.soh - 0.9) < 0.01 for r in records)  # 4.5 / 5.0 = 0.9


def test_build_cycle_records_skips_short_halves():
    _s3_mod = _load_step("_step3", "step3_deltaQ(V)_feature_extraction.py")
    small = pd.DataFrame({"time": [0, 1], "voltage": [4.0, 3.9], "current": [-1.0, -1.0]})
    good  = pd.DataFrame({
        "time": np.linspace(0, 3600, 50),
        "voltage": np.linspace(4.1, 2.6, 50),
        "current": np.full(50, -1.0),
    })

    class FakeParsed:
        cycles = {
            0: {"discharge": small, "charge": small, "meta": {"discharge_capacity_ah": 4.5}},
            1: {"discharge": good,  "charge": good,  "meta": {"discharge_capacity_ah": 4.5}},
        }

    records = _build_cycle_records(FakeParsed(), "c", 5.0, "discharge")
    assert len(records) == 1    # only the good one


# ---------------------------------------------------------------------------
# PipelineRunner._timed — checkpoint behaviour
# ---------------------------------------------------------------------------

def test_timed_skips_stage_not_in_stages(tmp_path):
    cfg     = PipelineConfig(cells=[], output_dir=tmp_path, stages=["parse"])
    runner  = PipelineRunner(cfg)
    called  = []
    result, sr = runner._timed("ica", "cell_01", lambda: called.append(1) or "done")
    assert result is None
    assert sr.status == "skipped"
    assert not called


def test_timed_runs_and_writes_checkpoint(tmp_path):
    cfg    = PipelineConfig(cells=[], output_dir=tmp_path, stages=["parse"], resume=True)
    runner = PipelineRunner(cfg)
    calls  = []

    result, sr = runner._timed("parse", "cell_01", lambda: calls.append(1) or "payload")
    assert sr.status == "ok"
    assert result == "payload"
    assert sr.artifact.exists()
    assert len(calls) == 1


def test_timed_resume_loads_checkpoint(tmp_path):
    cfg    = PipelineConfig(cells=[], output_dir=tmp_path, stages=["parse"], resume=True)
    runner = PipelineRunner(cfg)
    calls  = []

    # First run — writes checkpoint
    runner._timed("parse", "cell_01", lambda: calls.append(1) or "v1")
    assert len(calls) == 1

    # Second run — should use checkpoint, not call fn again
    result, sr = runner._timed("parse", "cell_01", lambda: calls.append(2) or "v2")
    assert result == "v1"
    assert sr.resumed is True
    assert len(calls) == 1   # fn was NOT called again


def test_timed_no_resume_reruns(tmp_path):
    cfg    = PipelineConfig(cells=[], output_dir=tmp_path, stages=["parse"], resume=False)
    runner = PipelineRunner(cfg)
    calls  = []

    runner._timed("parse", "cell_01", lambda: calls.append(1) or "v1")
    runner._timed("parse", "cell_01", lambda: calls.append(2) or "v2")
    assert len(calls) == 2


def test_timed_captures_exception(tmp_path):
    cfg    = PipelineConfig(cells=[], output_dir=tmp_path, stages=["parse"])
    runner = PipelineRunner(cfg)

    def boom():
        raise RuntimeError("stage failed!")

    result, sr = runner._timed("parse", "cell_01", boom)
    assert result is None
    assert sr.status == "failed"
    assert "RuntimeError" in sr.error or "stage failed" in sr.error


def test_timed_failed_stage_does_not_write_checkpoint(tmp_path):
    cfg    = PipelineConfig(cells=[], output_dir=tmp_path, stages=["parse"])
    runner = PipelineRunner(cfg)
    runner._timed("parse", "cell_01", lambda: (_ for _ in ()).throw(ValueError("oops")))
    ckpt = _ckpt(tmp_path, "cell_01", "parse")
    assert not ckpt.exists()


# ---------------------------------------------------------------------------
# Summary JSON output
# ---------------------------------------------------------------------------

def _make_finished_result(tmp_path) -> tuple:
    from run_pipeline import CellSummary, PipelineResult
    cs = CellSummary(
        cell_id="c1",
        stage_results=[
            StageResult(stage="parse",    status="ok",     elapsed_s=0.5),
            StageResult(stage="features", status="ok",     elapsed_s=0.2),
            StageResult(stage="model",    status="skipped", elapsed_s=0.0),
        ],
        n_cycles_parsed=10,
        n_ica_curves=2,
        n_features_ok=9,
        per_cell_metrics={"rmse": 0.01, "mae": 0.008, "r2": 0.95},
    )
    result = PipelineResult(
        cell_summaries={"c1": cs},
        model_metrics={"rmse": 0.01, "mae": 0.008, "r2": 0.95},
        overall_status="ok",
    )
    return result


def test_write_summary_creates_json(tmp_path):
    cfg    = PipelineConfig(cells=[], output_dir=tmp_path, stages=[])
    runner = PipelineRunner(cfg)
    result = _make_finished_result(tmp_path)
    path   = runner._write_summary(result)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["overall_status"] == "ok"
    assert "c1" in data["cells"]


def test_summary_json_has_required_keys(tmp_path):
    cfg    = PipelineConfig(cells=[], output_dir=tmp_path, stages=[])
    runner = PipelineRunner(cfg)
    result = _make_finished_result(tmp_path)
    path   = runner._write_summary(result)
    data   = json.loads(path.read_text())
    cell   = data["cells"]["c1"]
    for key in ("n_cycles_parsed", "n_ica_curves", "n_features_ok",
                "per_cell_metrics", "stages"):
        assert key in cell, f"Missing key: {key}"


def test_summary_json_stage_entries(tmp_path):
    cfg    = PipelineConfig(cells=[], output_dir=tmp_path, stages=[])
    runner = PipelineRunner(cfg)
    result = _make_finished_result(tmp_path)
    path   = runner._write_summary(result)
    data   = json.loads(path.read_text())
    stages = data["cells"]["c1"]["stages"]
    assert len(stages) == 3
    statuses = {s["stage"]: s["status"] for s in stages}
    assert statuses["parse"]   == "ok"
    assert statuses["model"]   == "skipped"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def test_main_requires_source():
    with pytest.raises(SystemExit) as exc:
        main(["--output-dir", "/tmp"])
    assert exc.value.code != 0


def test_main_dataset_requires_nominal_capacity(tmp_path):
    f = _dummy_csv(tmp_path / "cell.csv")
    rc = main(["--dataset", str(f), "--output-dir", str(tmp_path / "out")])
    assert rc == 2


def test_main_config_not_found_raises(tmp_path):
    with pytest.raises((FileNotFoundError, Exception)):
        main(["--config", str(tmp_path / "nonexistent.yaml")])
