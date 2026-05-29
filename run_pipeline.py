"""
run_pipeline.py
===============
End-to-end battery degradation pipeline.

Chain:  step1 (parse) → step2 (ICA) → step3 (features) → step4 (model)

Usage — with a YAML config:
    python run_pipeline.py --config pipeline.yaml

Usage — quick CLI (no YAML needed):
    python run_pipeline.py --dataset ./data/ --nominal-capacity 5.0 --output-dir ./out/
    python run_pipeline.py --dataset cell_01.csv --nominal-capacity 5.0 --output-dir ./out/

Resume (default):
    Re-running the same command skips any stage whose checkpoint already exists.
    Pass --no-resume to force a full re-run.

Example pipeline.yaml:
    dataset:
      cells:
        cell_01:
          path: ./data/cell_01.csv
          nominal_capacity_ah: 5.0
        cell_02:
          path: ./data/cell_02.csv
          nominal_capacity_ah: 5.0
    output_dir: ./pipeline_output/
    pipeline:
      resume: true
      stages: [parse, ica, features, model]
    parse:
      min_points_per_cycle: 20
    ica:
      dv_mv: 2.0
      savgol_window_mv: 20.0
    features:
      v_min: 2.5
      v_max: 4.2
      dv: 0.005
      chemistry: generic
      half_cycle: discharge
    model:
      alpha: 0.001
      l1_ratio: 0.5
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import logging
import sys
import time
import traceback
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FMT  = "%(asctime)s  %(levelname)-8s  %(name)-16s  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _setup_logging(output_dir: Path, level: int = logging.INFO) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pipeline.log"
    root = logging.getLogger()
    if root.handlers:                     # don't duplicate handlers on re-run
        return logging.getLogger("pipeline")
    fmt = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
    for h in [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a")]:
        h.setFormatter(fmt)
        root.addHandler(h)
    root.setLevel(level)
    return logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Dynamic module loader
# Step2 and step3 have parentheses in their filenames, so normal `import`
# is illegal.  We load them by file path and register in sys.modules.
# ---------------------------------------------------------------------------

_MODULE_CACHE: dict[str, types.ModuleType] = {}


def _load_step(slug: str, filename: str) -> types.ModuleType:
    if slug in _MODULE_CACHE:
        return _MODULE_CACHE[slug]
    here = Path(__file__).parent
    if str(here) not in sys.path:          # ensure sibling imports (feature_schema, etc.) resolve
        sys.path.insert(0, str(here))
    path = here / filename
    if not path.exists():
        raise FileNotFoundError(f"Step module not found: {path}")
    spec   = importlib.util.spec_from_file_location(slug, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[slug] = module
    spec.loader.exec_module(module)        # type: ignore[union-attr]
    _MODULE_CACHE[slug] = module
    return module


def _s1() -> types.ModuleType: return _load_step("_step1", "step1_data_parsing.py")
def _s2() -> types.ModuleType: return _load_step("_step2", "step2_Q(v)_extraction.py")
def _s3() -> types.ModuleType: return _load_step("_step3", "step3_deltaQ(V)_feature_extraction.py")
def _s4() -> types.ModuleType: return _load_step("_step4", "step4_soh_model.py")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CellSpec:
    """Dataset specification for one cell."""
    path:                str
    nominal_capacity_ah: float
    cell_id:             Optional[str]  = None   # defaults to filename stem
    current_unit:        Optional[str]  = None   # "A" | "mA"
    capacity_unit:       Optional[str]  = None   # "Ah" | "mAh"
    time_unit:           Optional[str]  = None   # "s" | "h" | "min"

    def resolved_id(self) -> str:
        return self.cell_id or Path(self.path).stem


@dataclass
class PipelineConfig:
    cells:      list[CellSpec]
    output_dir: Path

    parse:    dict[str, Any] = field(default_factory=dict)
    ica:      dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    model:    dict[str, Any] = field(default_factory=dict)

    resume: bool       = True
    stages: list[str]  = field(default_factory=lambda: ["parse", "ica", "features", "model"])


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage:      str
    status:     str                      # "ok" | "failed" | "skipped"
    elapsed_s:  float
    artifact:   Optional[Path] = None
    error:      Optional[str]  = None
    resumed:    bool           = False


@dataclass
class CellSummary:
    cell_id:           str
    stage_results:     list[StageResult] = field(default_factory=list)
    n_cycles_parsed:   int               = 0
    n_ica_curves:      int               = 0
    n_features_ok:     int               = 0
    per_cell_metrics:  Optional[dict]    = None
    flagged_cycles:    Optional[dict]    = None


@dataclass
class PipelineResult:
    cell_summaries:  dict[str, CellSummary]
    model_metrics:   Optional[dict] = None
    overall_status:  str            = "ok"
    summary_path:    Optional[Path] = None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _safe_kwargs(cls: type, raw: dict, exclude: frozenset[str] = frozenset()) -> dict:
    """Return only the keys in raw that are actual dataclass fields on cls,
    skipping complex types (Path) and any explicitly excluded names."""
    valid = {f.name for f in dataclasses.fields(cls)} - exclude
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in valid:
            continue
        # Silently skip Path fields supplied as strings — those go through constructors.
        ann = cls.__dataclass_fields__[k].type  # type: ignore[attr-defined]
        if "Path" in str(ann):
            continue
        out[k] = v
    return out


def load_config(
    yaml_path:           str | Path,
    output_dir:          Optional[str]   = None,
    nominal_capacity_ah: Optional[float] = None,
) -> PipelineConfig:
    """Load PipelineConfig from a YAML file, with optional CLI overrides."""
    raw = yaml.safe_load(Path(yaml_path).read_text())
    ds  = raw.get("dataset", {})

    cells: list[CellSpec] = []
    if "cells" in ds:
        for cid, spec in ds["cells"].items():
            cap = spec.get("nominal_capacity_ah",
                           nominal_capacity_ah or ds.get("nominal_capacity_ah"))
            if cap is None:
                raise ValueError(f"nominal_capacity_ah missing for cell {cid!r}")
            cells.append(CellSpec(
                path=spec["path"], nominal_capacity_ah=float(cap), cell_id=cid,
                current_unit=spec.get("current_unit"),
                capacity_unit=spec.get("capacity_unit"),
                time_unit=spec.get("time_unit"),
            ))
    elif "dir" in ds:
        cap = nominal_capacity_ah or ds.get("nominal_capacity_ah")
        if cap is None:
            raise ValueError("nominal_capacity_ah required when using dataset.dir")
        pattern = ds.get("pattern", "*.csv")
        for p in sorted(Path(ds["dir"]).glob(pattern)):
            cells.append(CellSpec(path=str(p), nominal_capacity_ah=float(cap)))
    else:
        raise ValueError("Config requires dataset.cells or dataset.dir")

    if not cells:
        raise ValueError("No cells found in config")

    pipe_section = raw.get("pipeline", {})
    return PipelineConfig(
        cells=cells,
        output_dir=Path(output_dir or raw.get("output_dir", "./pipeline_output")),
        parse=raw.get("parse", {}),
        ica=raw.get("ica", {}),
        features=raw.get("features", {}),
        model=raw.get("model", {}),
        resume=pipe_section.get("resume", True),
        stages=pipe_section.get("stages", ["parse", "ica", "features", "model"]),
    )


def config_from_cli(
    dataset:             str,
    output_dir:          str,
    nominal_capacity_ah: float,
) -> PipelineConfig:
    """Build a minimal PipelineConfig from CLI arguments (no YAML required)."""
    p = Path(dataset)
    if p.is_dir():
        files = sorted([*p.glob("*.csv"), *p.glob("*.parquet"), *p.glob("*.pq")])
        if not files:
            raise ValueError(f"No .csv / .parquet files found in {p}")
        cells = [CellSpec(path=str(f), nominal_capacity_ah=nominal_capacity_ah) for f in files]
    else:
        cells = [CellSpec(path=str(p), nominal_capacity_ah=nominal_capacity_ah)]
    return PipelineConfig(cells=cells, output_dir=Path(output_dir))


# ---------------------------------------------------------------------------
# Bridge: ParsedDataset → CycleRecord list (step1 → step3)
# ---------------------------------------------------------------------------

def _build_cycle_records(parsed: Any, cell_id: str,
                         nominal_capacity_ah: float, half_cycle: str) -> list:
    """Convert ParsedDataset cycle objects into CycleRecord instances for step3."""
    s3 = _s3()
    records = []
    for cid in sorted(parsed.cycles):
        cyc  = parsed.cycles[cid]
        half = cyc.get(half_cycle)
        if half is None or len(half) < 10:
            continue
        cap_key = f"{half_cycle}_capacity_ah"
        cap_ah  = float(cyc["meta"].get(cap_key) or 0.0)
        soh     = (cap_ah / nominal_capacity_ah) if nominal_capacity_ah > 0 else None
        records.append(s3.CycleRecord(
            cell_id=cell_id,
            cycle_number=int(cid),
            half=half,
            soh=soh,
        ))
    return records


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _ckpt(output_dir: Path, cell_id: str, stage: str) -> Path:
    return output_dir / "checkpoints" / f"{cell_id}__{stage}.pkl"


class PipelineRunner:
    """Orchestrates the four-stage battery degradation pipeline.

    Stages run per-cell (parse → ica → features), then once over all cells (model).
    Each stage is checkpointed; resuming a run re-uses existing checkpoints.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg  = cfg
        self._out = cfg.output_dir
        self._log = logging.getLogger("pipeline")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        self._out.mkdir(parents=True, exist_ok=True)

        cell_summaries: dict[str, CellSummary] = {}
        all_feature_dfs: list[pd.DataFrame]    = []

        for spec in self.cfg.cells:
            cell_id = spec.resolved_id()
            self._log.info("=== Cell %s ===", cell_id)
            summary = CellSummary(cell_id=cell_id)

            # --- Stage 1: parse ---
            parsed, sr = self._timed("parse", cell_id, lambda: self._do_parse(spec, cell_id))
            summary.stage_results.append(sr)
            if parsed is None:
                cell_summaries[cell_id] = summary
                continue
            summary.n_cycles_parsed = len(parsed.cycles)
            summary.flagged_cycles  = {str(k): v for k, v in parsed.flagged().items()}

            # --- Stage 2: ICA (failure doesn't block features) ---
            ica_curves, sr = self._timed("ica", cell_id, lambda: self._do_ica(parsed, cell_id))
            summary.stage_results.append(sr)
            if ica_curves is not None:
                summary.n_ica_curves = len(ica_curves)

            # --- Stage 3: features ---
            feat_df, sr = self._timed(
                "features", cell_id, lambda: self._do_features(parsed, spec, cell_id)
            )
            summary.stage_results.append(sr)
            if feat_df is not None:
                summary.n_features_ok = int(feat_df["extraction_ok"].sum()) \
                    if "extraction_ok" in feat_df.columns else len(feat_df)
                all_feature_dfs.append(feat_df)

            cell_summaries[cell_id] = summary

        # --- Stage 4: model (all cells together) ---
        model_metrics = None
        if "model" in self.cfg.stages and all_feature_dfs:
            combined      = pd.concat(all_feature_dfs, ignore_index=True)
            features_path = self._out / "features" / "all_features.csv"
            features_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(features_path, index=False)
            self._log.info("Combined feature matrix: %d rows from %d cells",
                           len(combined), len(all_feature_dfs))

            model_results, sr = self._timed(
                "model", "all_cells", lambda: self._do_model(features_path)
            )
            # Attach per-cell metrics back to each cell's summary
            if model_results is not None:
                model_metrics = model_results.overall_metrics
                for cid, m in model_results.per_cell_metrics.items():
                    if cid in cell_summaries:
                        cell_summaries[cid].per_cell_metrics = m

        failed = sum(
            any(sr.status == "failed" for sr in cs.stage_results)
            for cs in cell_summaries.values()
        )
        overall = "failed" if failed == len(cell_summaries) else \
                  "partial" if failed else "ok"

        result = PipelineResult(
            cell_summaries=cell_summaries,
            model_metrics=model_metrics,
            overall_status=overall,
        )
        result.summary_path = self._write_summary(result)
        return result

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _do_parse(self, spec: CellSpec, cell_id: str) -> Any:
        s1  = _s1()
        raw = {k: v for k, v in self.cfg.parse.items()
               if k not in {"aliases", "cache_dir"}}
        parse_cfg = s1.ParseConfig(
            **_safe_kwargs(s1.ParseConfig, raw, exclude=frozenset({"aliases", "cache_dir"}))
        )
        # Point step1's cache to our output tree so artifacts are co-located
        parse_cfg.cache_dir = self._out / "parse_cache"
        cell_cfg = s1.CellConfig(
            nominal_capacity_ah=spec.nominal_capacity_ah,
            current_unit=spec.current_unit,
            capacity_unit=spec.capacity_unit,
            time_unit=spec.time_unit,
        )
        parsed = s1.parse_file(spec.path, cell_cfg, parse_cfg, use_cache=False)
        self._log.info("  Parsed %d cycles  |  ICA-suitable: %d  |  flagged: %d",
                       len(parsed.cycles), len(parsed.ica_cycles()), len(parsed.flagged()))
        return parsed

    def _do_ica(self, parsed: Any, cell_id: str) -> list:
        s2      = _s2()
        ica_raw = _safe_kwargs(s2.ICAConfig, self.cfg.ica)
        ica_cfg = s2.ICAConfig(**ica_raw)
        curves  = s2.run_ica(
            parsed, ica_cfg,
            cache_dir=self._out / "ica_cache" / cell_id,
            use_cache=False,
        )
        # Save ICA plots alongside other artifacts
        plot_dir = self._out / "ica_plots" / cell_id
        plot_dir.mkdir(parents=True, exist_ok=True)
        for curve in curves:
            s2.plot_ica(curve, save_path=str(
                plot_dir / f"cyc{curve.cycle_index}_{curve.half}.png"
            ))
        self._log.info("  ICA curves: %d  |  plots -> %s", len(curves), plot_dir)
        return curves

    def _do_features(self, parsed: Any, spec: CellSpec, cell_id: str) -> pd.DataFrame:
        s3       = _s3()
        feat_raw = _safe_kwargs(s3.FeatureConfig, self.cfg.features,
                                exclude=frozenset({"cache_dir"}))
        feat_cfg = s3.FeatureConfig(**feat_raw)

        records = _build_cycle_records(
            parsed, cell_id, spec.nominal_capacity_ah, feat_cfg.half_cycle
        )
        if not records:
            raise RuntimeError(f"No usable {feat_cfg.half_cycle} half-cycles in {cell_id}")

        builder = s3.FeatureMatrixBuilder(feat_cfg, spec.nominal_capacity_ah)
        df      = builder.build(records)

        out_dir  = self._out / "features"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{cell_id}_features.csv"
        df.to_csv(csv_path, index=False)

        ok_frac  = df["extraction_ok"].mean() if "extraction_ok" in df.columns else 1.0
        self._log.info("  Features: %d cycles  |  extraction OK: %.0f%%  |  -> %s",
                       len(df), ok_frac * 100, csv_path)
        return df

    def _do_model(self, features_path: Path) -> Any:
        s4      = _s4()
        m_raw   = {k: v for k, v in self.cfg.model.items()
                   if k in ("alpha", "l1_ratio", "max_iter", "random_state")}
        trainer = s4.SOHModelTrainer(**m_raw)
        results = trainer.run(features_path, self._out / "model")
        m = results.overall_metrics
        self._log.info("  Model  RMSE=%.5f  MAE=%.5f  R²=%.5f", m["rmse"], m["mae"], m["r2"])
        return results

    # ------------------------------------------------------------------
    # Checkpoint / resume / timing wrapper
    # ------------------------------------------------------------------

    def _timed(self, stage: str, cell_id: str, fn) -> tuple[Any, StageResult]:
        """Execute fn() with timing, checkpointing, resume, and exception handling."""
        if stage not in self.cfg.stages:
            return None, StageResult(stage=stage, status="skipped", elapsed_s=0.0)

        ckpt = _ckpt(self._out, cell_id, stage)

        if self.cfg.resume and ckpt.exists():
            self._log.info("[%s] %s: checkpoint found — skipping", stage, cell_id)
            try:
                return joblib.load(ckpt), StageResult(
                    stage=stage, status="skipped", elapsed_s=0.0,
                    artifact=ckpt, resumed=True,
                )
            except Exception as exc:
                self._log.warning("[%s] %s: checkpoint unreadable (%s) — rerunning",
                                  stage, cell_id, exc)

        t0 = time.perf_counter()
        self._log.info("[%s] %s: starting", stage, cell_id)
        try:
            result  = fn()
            elapsed = time.perf_counter() - t0
            self._log.info("[%s] %s: done in %.2fs", stage, cell_id, elapsed)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(result, ckpt)
            return result, StageResult(stage=stage, status="ok",
                                       elapsed_s=round(elapsed, 3), artifact=ckpt)
        except Exception:
            elapsed  = time.perf_counter() - t0
            full_tb  = traceback.format_exc()
            last_err = full_tb.strip().splitlines()[-1]
            self._log.error("[%s] %s: FAILED in %.2fs\n%s", stage, cell_id, elapsed, full_tb)
            return None, StageResult(stage=stage, status="failed",
                                     elapsed_s=round(elapsed, 3), error=last_err)

    # ------------------------------------------------------------------
    # JSON summary
    # ------------------------------------------------------------------

    def _write_summary(self, result: PipelineResult) -> Path:
        def _sr_dict(sr: StageResult) -> dict:
            return {
                "stage":     sr.stage,
                "status":    sr.status,
                "elapsed_s": sr.elapsed_s,
                "resumed":   sr.resumed,
                "error":     sr.error,
                "artifact":  str(sr.artifact) if sr.artifact else None,
            }

        payload: dict[str, Any] = {
            "overall_status": result.overall_status,
            "model_metrics":  result.model_metrics,
            "cells": {
                cid: {
                    "n_cycles_parsed":  cs.n_cycles_parsed,
                    "n_ica_curves":     cs.n_ica_curves,
                    "n_features_ok":    cs.n_features_ok,
                    "per_cell_metrics": cs.per_cell_metrics,
                    "flagged_cycles":   cs.flagged_cycles or {},
                    "stages":           [_sr_dict(sr) for sr in cs.stage_results],
                }
                for cid, cs in result.cell_summaries.items()
            },
        }

        summary_path = self._out / "pipeline_summary.json"
        summary_path.write_text(json.dumps(payload, indent=2, default=str))
        self._log.info("Summary written -> %s", summary_path)
        return summary_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Battery degradation pipeline: parse → ICA → features → model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config",  metavar="YAML",
                     help="Full pipeline config YAML")
    src.add_argument("--dataset", metavar="PATH",
                     help="Dataset directory or single file (no YAML needed)")

    p.add_argument("--output-dir", metavar="DIR", default="./pipeline_output",
                   help="Root output directory (default: ./pipeline_output)")
    p.add_argument("--nominal-capacity", type=float, metavar="AH",
                   help="Nominal cell capacity in Ah (required with --dataset)")

    resume = p.add_mutually_exclusive_group()
    resume.add_argument("--resume",    dest="resume", action="store_true",  default=True,
                        help="Skip stages whose checkpoint already exists (default)")
    resume.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Ignore existing checkpoints, re-run all stages")

    p.add_argument("--stages", nargs="+",
                   choices=["parse", "ica", "features", "model"],
                   default=["parse", "ica", "features", "model"],
                   metavar="STAGE",
                   help="Stages to run (default: all)")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG-level logging")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    log   = _setup_logging(Path(args.output_dir), level=level)

    # Build config
    if args.config:
        cfg = load_config(args.config, output_dir=args.output_dir)
    else:
        if args.nominal_capacity is None:
            log.error("--nominal-capacity is required when using --dataset")
            return 2
        cfg = config_from_cli(args.dataset, args.output_dir, args.nominal_capacity)

    cfg.resume = args.resume
    cfg.stages = args.stages

    log.info("Pipeline starting  |  cells=%d  stages=%s  resume=%s  output=%s",
             len(cfg.cells), cfg.stages, cfg.resume, cfg.output_dir)

    runner = PipelineRunner(cfg)
    result = runner.run()

    log.info("Pipeline complete  |  status=%s  summary=%s",
             result.overall_status, result.summary_path)

    if result.model_metrics:
        m = result.model_metrics
        log.info("Overall model  RMSE=%.5f  MAE=%.5f  R²=%.5f",
                 m["rmse"], m["mae"], m["r2"])

    return 0 if result.overall_status != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
