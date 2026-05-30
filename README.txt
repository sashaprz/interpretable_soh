Dataset: https://www.kaggle.com/datasets/sudhaveerapaneni/oxford-dataset

What this project is

  A battery State-of-Health (SOH) prediction pipeline with a web dashboard.
  It takes raw electrochemical cycling data from lithium-ion batteries, runs
  physics-informed signal processing (dQ/dV incremental capacity analysis),
  and uses an ElasticNet regression model to predict how much capacity a cell
  has retained (SOH = 1.0 = fully healthy, 0.8 = end-of-life threshold).

  The dataset is the Oxford Battery Degradation Dataset
  (Oxford_Battery_Degradation_Dataset_1.mat, 8 graphite/NMC pouch cells).

  There are two primary user flows:

  - Flow 1: Batch training — upload cycle data for multiple cells, train an
    ElasticNet model via leave-one-cell-out cross-validation, get a full SOH
    trajectory and degradation mechanism report for each cell.
  - Flow 2: Single-cell prediction — provide discharge cycle measurements for
    a new cell, run the trained model, and get an SOH estimate + plain-language
    explanation of which degradation mechanism is active.

  ---
  Running the dashboard

  Local:
    pip install -r requirements.txt
    python server.py
    open http://localhost:5000

  Deployed (Render.com or similar):
    Push the repo to GitHub, connect to Render as a Web Service.
    Build command: pip install -r requirements.txt
    Start command: python server.py
    The server reads the PORT environment variable automatically.

  The dashboard (dashboard.html) is a self-contained React app served by
  Flask. It works in demo mode (synthetic data) when opened as a plain file,
  and connects to the pipeline when served through server.py.

  ---
  New files added for the dashboard

  server.py
    Flask web server. Exposes five routes:
      GET  /                    — serve dashboard.html
      GET  /api/models          — list trained models saved to disk
      POST /api/train           — receive uploaded files, run the pipeline in
                                  a background thread, return a job_id
      GET  /api/jobs/<job_id>   — poll training progress (0.0–1.0) and result
      POST /api/predict         — receive one cell's data, run the pipeline
                                  (parse→ICA→features) + trained model, return
                                  the full SOH trajectory and mechanism report

    Training runs in a background thread so the server stays responsive to
    other users. Job state is persisted to pipeline_output/jobs/ so a server
    restart does not lose a completed result.

    The data adapter (_cell_to_json) converts pipeline output into the JSON
    schema the dashboard charts expect: per-cycle SOH true/pred, mechanism
    signals (LLI/LAM/RES) from interpret_cycle, phases, onsets, RUL, and
    peak diagnostics.

  mat_converter.py
    Converts the Oxford Battery Degradation Dataset .mat file into per-cell
    CSVs that step1_data_parsing.py can ingest. Cells are loaded one at a
    time (scipy variable_names) and OCV discharge curves are downsampled to
    1500 points/cycle to keep peak RAM below 50 MB. The slow OCV discharge
    (OCVdc, ~C/25) is preferred over the 1C discharge for ICA quality.

    Output columns: cycle_index, time (s), voltage (V), current (A),
    capacity (Ah), temperature (°C).

    Uploading a .mat file through the dashboard automatically triggers this
    conversion before passing data to the pipeline.

  dashboard.html
    Single-file React app (React 18, Babel standalone, IBM Plex fonts, all
    inlined — no build step). Loads from the server but also works as a
    static file in demo mode (falls back to synthetic data from soh-data.js
    when API calls fail).

    Stages:
      Landing      — model library with hero heading, list of trained models
      Upload/Train — file picker (.mat or .csv), nominal capacity input,
                     demo batch fallback
      Training     — real-time progress bar polling /api/jobs/<id> every
                     700 ms; shows current pipeline stage (e.g. "ica · cell3")
      Upload/Pred  — file picker for a single cell to score
      Predicting   — animation while /api/predict runs synchronously
      Report       — SOH trajectory chart, prediction callout panel (plain-
                     language SOH/RUL/mechanism explanation), mechanism
                     timeline, mechanism report cards (LLI/LAM/RES), dQ/dV
                     explainer guide

  build_dashboard.py
    Regenerates dashboard.html by reading the JSX component files from the
    original design ZIP and injecting the modified soh-landing.jsx and
    soh-stages.jsx sections. Run python build_dashboard.py after editing
    either of those sections.

  ---
  The pipeline stages

  feature_schema.py — The contract layer

    Single source of truth for column names between step3 and step4.
    Defines FEATURE_COLS (10 canonical names), METADATA_COLS, COLUMN_ALIASES
    (legacy rename map), validate_feature_columns(), and save_schema_json().
    Step3 asserts at import time that its internal feature dict matches
    FEATURE_COLS exactly — drift causes an immediate import failure.

  ---
  step1_data_parsing.py — Format-agnostic data ingestion

    Loads raw cycler files (CSV, Excel, Parquet, Feather), standardizes
    column names, detects cycles, and returns a ParsedDataset.

    Key design decisions:
    - Raw data is never mutated; all transforms happen on copies.
    - Missing critical columns (time, voltage, current) raise a hard error.
    - Cycle detection is rest-aware, not naive zero-crossing.

    Pipeline inside parse_file():
    1. load_raw()            — auto-detect delimiter; cache to .parsed.pkl
    2. standardize_columns() — map source names to canonical schema via
                               DEFAULT_ALIASES (case-insensitive)
    3. validate_required()   — hard fail if time/voltage/current absent
    4. coerce_numeric()      — convert to float; detect datetime strings
    5. normalize_units()     — convert to SI (A, Ah, s); CellConfig overrides
                               take priority
    6. clean_rows()          — drop NaN, sort by time, dedup timestamps
    7. detect_cycles()       — state machine (charge/discharge/rest) with
                               carry-forward rest assignment; increments on
                               discharge → charge transition
    8. build_cycle_object()  — per-cycle dict: data, charge, discharge,
                               meta (capacity, C-rate, ICA flag), flags

  ---
  step2_Q(v)_extraction.py — Incremental Capacity Analysis (ICA)

    Computes dQ/dV curves from individual cycle half-cycles. ICA peaks
    reveal electrode phase transitions; how they shift, shrink, or broaden
    over life identifies the degradation mechanism.

    Key decisions:
    - Savitzky-Golay deriv=1 (not smooth-then-gradient) to avoid
      re-injecting noise via finite difference.
    - PCHIP interpolation instead of cubic spline (no overshoot on steep
      Q(V) regions, no phantom peaks).
    - Every ICACurve carries a SHA-256 hash of the ICAConfig used to
      produce it. assert_comparable() refuses cross-config comparisons.

    Pipeline per half-cycle in compute_ica_for_half():
    1. CC segment extraction — keep only constant-current portion
       (|I - median|I|| / median ≤ 5%)
    2. Q(V) construction — coulomb-count via trapezoidal integration
    3. Monotonic voltage enforcement — keep only advancing voltage points
    4. PCHIP interpolation onto uniform grid (dv_mv, default 2 mV)
    5. SG smoothing + differentiation (deriv=1, delta=dv)
    6. Peak detection — scipy.signal.find_peaks with prominence threshold
    7. Validation — smoothing distortion check, peak stability check

    run_ica() auto-selects the lowest C-rate cycle (≤C/10 preferred),
    runs compute_ica_for_half(), and caches results per config hash.

  ---
  step3_deltaQ(V)_feature_extraction.py — ΔQ(V) feature engineering

    Computes ΔQ(V) = Q_cycle(V) − Q_reference(V) for each cycle and
    extracts 10 statistical features. These are the ML model inputs.

    FeatureConfig: shared voltage grid, SG parameters, CC gating, which
    half-cycle to use, reference cycle rank, config hash.

    QVExtractor: raw half-cycle DataFrame → smooth Q(V) curve + dQ/dV.
    Three CC gates: C-rate ≤ c_rate_max, |dI/dt| stability, rolling
    current CV ≤ 2%. Rejects cycles where < 20% of points pass.

    DeltaQVComputer: Q_cycle − Q_reference, NaN propagated.

    FeatureExtractor: 10 features from ΔQ(V) (finite values only, ±5σ
    outliers clipped): variance, log-variance, skewness, kurtosis,
    integral of |ΔQ|, max deviation, min, max, mean, RMS.
    SOH leakage guard blocks proxy names like discharge_capacity_ah.

    FeatureMatrixBuilder: groups by cell, picks reference per cell,
    computes ΔQ(V) and features for every cycle. Outputs metadata +
    feature columns; failed cycles get extraction_ok=False and NaN features.

  ---
  step4_soh_model.py — ElasticNet SOH model

    Trains an ElasticNet to predict SOH from the 10 ΔQ(V) features via
    Leave-One-Cell-Out cross-validation (LOCO-CV).

    SOHModelTrainer:
    - _build_pipeline(): StandardScaler → ElasticNet(α=1e-3, l1_ratio=0.5)
    - _loco_splits(): yields (train_df, test_df, cell_id) — one cell held
      out per fold, training on all others
    - fit(): LOCO CV for honest metrics, then a final model on all data
      for serialization. Metrics come only from held-out predictions.
    - save(): elasticnet_soh.joblib, feature_schema.json, metrics.json,
      predictions.csv, feature_importance.csv

    ModelResults: predictions DataFrame, overall/per-cell metrics,
    feature importance table, artifact paths.

  ---
  step4_interpretation.py — Physics-based degradation interpretation

    Given ICA curves, computes the three canonical degradation mechanisms:

    LLI (Loss of Lithium Inventory): peaks shift in voltage.
    LAM (Loss of Active Material): integrated ICA area shrinks.
    Resistance growth: peaks broaden (higher impedance).

    extract_peaks(): scipy.signal.find_peaks; widths returned in mV
    (width_mV = sample_widths × grid_spacing_mV, grid-invariant).

    compute_lli() / _match_peaks(): Hungarian algorithm
    (scipy.optimize.linear_sum_assignment) for optimal peak matching.
    Pairs > 20 mV apart are treated as disappeared/appeared.
    Returns LLIResult: mean shift, confidence, disappearance penalty.

    compute_lam(): 1 − (curr_area / ref_area) via Simpson's rule.

    compute_resistance_growth(): mean fractional peak width increase.
    Both inputs must be in mV (not sample indices), enforced by assertion.

    interpret_cycle(): calls all three, shares grid spacing, returns flat
    metrics dict + full LLIResult. Used by server.py to produce per-cycle
    mechanism signals (LLI/LAM/RES) for the dashboard report.

    build_physics_features(): interpret_cycle() over all cycles in an ICA
    DataFrame, grouped by cell.

  ---
  predict.py — Inference-time predictor

    SOHPredictor: load a trained model and predict SOH for a single new
    half-cycle plus generate a natural-language explanation.

    - from_model(): load a joblib pipeline artifact.
    - set_reference(): process an early-life half-cycle through QVExtractor.
    - assess(): QVExtract current cycle → ΔQ(V) → features → predict SOH
      → interpret_cycle() for physics → natural-language explanation.
    - assess_trajectory(): run assess() over a list of cycles → DataFrame.

    _explain(): generates a human-readable paragraph from physics metrics
    (LLI confidence + shift magnitude, LAM percentage, resistance growth).

  ---
  run_pipeline.py — End-to-end pipeline orchestrator

    Chains step1 → step2 → step3 → step4 for a batch of cells.
    Supports YAML config files or CLI arguments.
    Checkpoints each stage to disk (joblib) and resumes by default.
    PipelineRunner._timed() wraps each stage with timing, error handling,
    and checkpoint save/load.

    server.py subclasses PipelineRunner as _TrackingRunner, overriding
    _timed() to update the job progress dict after each stage completes
    so the dashboard polling can show real-time progress.

  ---
  How the files connect (data flow)

  Oxford .mat file
         │
    mat_converter.py  ← converts .mat → per-cell CSVs (one at a time,
         │               1500 pts/cycle to cap RAM at ~50 MB)
    step1_data_parsing.py
    ├─ raw → clean → cycles
    └─ ParsedDataset
         │
    step2_Q(v)_extraction.py  ← ICA for slow OCV discharge cycles
    └─ ICACurve list (dQ/dV on uniform voltage grid)
         │
    step3_deltaQ(V)_feature_extraction.py
    ├─ QVExtractor: raw half-cycle → Q(V) smooth
    ├─ DeltaQVComputer: Q_cycle − Q_reference
    └─ FeatureExtractor: 10 statistical features
         │
    feature_schema.py  ← FEATURE_COLS contract (step3 ↔ step4)
         │
    step4_soh_model.py
    ├─ StandardScaler → ElasticNet
    ├─ LOCO-CV → predictions, metrics
    └─ elasticnet_soh.joblib
         │
    step4_interpretation.py
    ├─ LLI: peak shift via Hungarian matching
    ├─ LAM: ICA area loss (Simpson's rule)
    └─ Resistance: peak broadening (mV, grid-invariant)
         │
    server.py  ← wires pipeline to web dashboard
    ├─ _TrackingRunner: real-time progress reporting
    ├─ _cell_to_json: pipeline output → dashboard JSON schema
    └─ _build_batch: averages mechanism signals across cells
         │
    dashboard.html  ← React UI served at /
    ├─ Train flow: upload → progress bar → batch SOH report
    └─ Predict flow: upload → SOH + RUL + mechanism callout

  Every artifact (ICA cache, trained pipeline, feature schema JSON, job
  results) includes a config hash so stale artifacts from a different
  configuration are detected and rejected.

  ---
  Notes on the data

  The Oxford dataset has 8 graphite/NMC pouch cells cycled at 40–50°C.
  Each cell has cycle snapshots every ~100 cycles. Each snapshot includes:
    C1ch / C1dc   — 1C charge/discharge
    OCVch / OCVdc — slow (~C/25) OCV charge/discharge  ← used for ICA

  The OCV discharge is preferred because the slow rate gives sharp, well-
  resolved dQ/dV peaks. The mat_converter.py extracts OCVdc by default,
  falling back to C1dc if OCV is unavailable.

  Nominal capacity for the Oxford cells is approximately 0.74 Ah (740 mAh).
