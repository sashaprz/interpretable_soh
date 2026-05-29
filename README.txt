Dataset: https://www.kaggle.com/datasets/sudhaveerapaneni/oxford-dataset

What this project is

  This is a battery State-of-Health (SOH) prediction pipeline. It takes
  raw electrochemical cycling data from lithium-ion batteries and uses
  physics-informed signal processing + machine learning to predict how
  much capacity a battery has retained (SOH = 1.0 means healthy, 0.7
  means 70% capacity remaining). The dataset is the Oxford Battery
  Degradation Dataset (the .mat file in the root).

  There are two primary user flows:

  - Flow 1: Multiple cells, batch training — upload cycle CSVs → train
  a model → get SOH trajectory + report.
  - Flow 2: Single new cycle — provide one discharge measurement for a
  known cell → get an SOH prediction + a natural-language explanation
  of what's degrading.

  ---
  The pipeline stages

  feature_schema.py — The contract layer

  This tiny but critical file is the single source of truth for column
  names that flow between steps 3 and 4. It defines:

  - FEATURE_COLS: The 10 canonical feature names (dqv_variance,
  dqv_log_variance, dqv_skewness, dqv_kurtosis, dqv_integral_abs,
  dqv_max_deviation, dqv_min, dqv_max, dqv_mean, dqv_rms).
  - METADATA_COLS: Non-feature columns always present (cell_id,
  cycle_number, soh, etc.)
  - COLUMN_ALIASES: A rename map for legacy column names, enabling
  backward compatibility.
  - validate_feature_columns(): Raises a descriptive ValueError if a
  DataFrame is missing required columns — this is called at model
  training and inference time.
  - save_schema_json(): Writes the contract to disk alongside model
  artifacts.

  Step 3 imports this at module load time and asserts that its internal
  _CORE_FEATURES dictionary keys exactly match FEATURE_COLS. If they
  ever drift, the import fails immediately.

  ---
  step1_data_parsing.py — Format-agnostic data ingestion

  Purpose: Load raw cycler files (CSV, Excel, Parquet, Feather) from
  any vendor format, standardize column names, detect cycles, and
  package everything into a ParsedDataset.

  Key design decisions:
  - The codebase never mutates raw data. raw is always preserved
  untouched; all transforms happen on copies.
  - Missing critical columns raise a hard error — no interpolation or
  guessing.
  - Cycle detection is rest-aware, not naïve zero-crossing (which would
  shatter cycles at every noise spike).

  The full pipeline inside parse_file():

  1. load_raw() — Dispatches on file extension via _read_any(). CSV
  uses sep=None with the Python engine to auto-detect delimiters
  (tab/comma/semicolon). Results are cached to disk as .parsed.pkl
  files.
  2. standardize_columns() — Maps source column names to the canonical
  schema (time, voltage, current, capacity, cycle_index, step) using a
  large alias dictionary (DEFAULT_ALIASES). Column name matching is
  case-insensitive and strips whitespace. Unit hints are parsed from
  column names like "Capacity/mAh" or "I/mA" and stored for the next
  step. Non-canonical leftover columns are kept for debugging.
  3. validate_required() — Hard fails if time, voltage, or current are
  absent. No fallbacks.
  4. coerce_numeric() — Converts canonical columns to float. Datetime
  strings in the time column are detected heuristically (>50%
  parseable) and converted to elapsed seconds.
  5. normalize_units() — Converts to SI base units (Amps, Ah, seconds).
  CellConfig unit overrides take priority; otherwise the inferred unit
  from the column name is used. Notably, it refuses to magnitude-guess
  (e.g., if mA vs A is ambiguous, it warns and assumes A rather than
  looking at the numbers).
  6. clean_rows() — Drops rows with NaN in required columns, sorts by
  time (stable mergesort), and removes duplicate timestamps (keep
  first).
  7. detect_cycles() — Tries to use an existing cycle_index column if
  it's monotonically non-decreasing with >1 unique value. Otherwise,
  runs a state machine: assigns +1 (charging), -1 (discharging), or 0
  (rest) based on current sign vs. a small threshold. Rest periods
  inherit the last non-rest sign (carry-forward). A new cycle counter
  increments each time the state transitions from discharge → charge.
  8. build_cycle_object() — For each cycle, builds a dict containing:
    - data: the full cleaned cycle DataFrame
    - charge / discharge: half-cycle DataFrames split by phase
    - meta: capacity (from reported column or coulomb-counted via
  trapezoid integration), voltage range, duration, estimated C-rate,
  ICA suitability flag
    - flags: list of quality issues (too few points, unknown C-rate,
  fast-charge rejection)

  ParsedDataset is the output object. It has raw, clean, cycles (dict
  of cycle objects), and helper methods ica_cycles() and flagged().

  ---
  step2_Q(v)_extraction.py — Incremental Capacity Analysis (ICA)

  Purpose: Compute dQ/dV curves (the "ICA curve") from individual cycle
  half-cycles. These curves reveal the electrochemical phase
  transitions of the electrode materials as sharp peaks. How those
  peaks shift, shrink, or disappear over life tells you the degradation
  mechanism.

  Key design decisions (explicitly stated in the docstring):
  - Uses Savitzky-Golay deriv=1 instead of smooth-then-gradient. This
  takes the derivative analytically from the same local polynomial used
  for smoothing, avoiding re-injecting noise via a separate finite
  difference.
  - Uses PCHIP interpolation (Piecewise Cubic Hermite Interpolating
  Polynomial) instead of cubic spline. Cubic splines overshoot on steep
  Q(V) regions and manufacture phantom peaks. PCHIP cannot overshoot.
  - Every ICACurve carries a SHA-256 hash of the ICAConfig used to
  produce it. assert_comparable() refuses to let curves produced with
  different configs be compared.

  The pipeline per half-cycle in compute_ica_for_half():

  1. CC segment extraction (_extract_cc_segment()): Keeps only the
  constant-current portion. Identifies CC as points where |I -
  median(|I|)| / median(|I|) ≤ cc_current_tol (default 5%). Drops CV
  tails and current ramps.
  2. Q(V) construction (_build_qv()): Coulomb-counts charge using
  trapezoidal integration of |I| dt, converting seconds → Ah. Returns
  time-ordered (V, Q) pairs.
  3. Monotonic voltage enforcement (_enforce_monotonic_v()): Determines
  charge (V increasing) or discharge (V decreasing) from the net
  trend, then keeps only points that advance voltage in that direction.
  Flips V and Q for discharge so downstream code always sees ascending
  voltage.
  4. Interpolation onto uniform grid: PCHIP or linear (configurable).
  The uniform grid is spaced at dv_mv (default 2 mV).
  5. Smoothing + differentiation: Either savgol_filter(deriv=1,
  delta=dv) (default) or smooth-then-gradient. Edge artifacts from the
  SG filter are trimmed.
  6. Peak detection: scipy.signal.find_peaks on the dQ/dV curve with a
  prominence threshold (default 5% of max |dQ/dV|).
  7. Validation:
    - Smoothing distortion: RMSE between smoothed and raw Q divided by
  capacity. If > 2%, flagged.
    - Peak stability: Re-runs with a 1.5× wider window; peaks must not
  shift by more than 10 mV.

  run_ica() is the top-level driver. It auto-selects the lowest C-rate
  cycle (≤C/10 preferred for research grade, ≤C/5 as fallback), runs
  compute_ica_for_half() for each requested half, and caches results
  per (source, cycle, half, config_hash).

  ---
  step3_deltaQ(V)_feature_extraction.py — Delta Q(V) feature
  engineering

  Purpose: For each cycle, compute ΔQ(V) = Q_cycle(V) - Q_reference(V)
  — the difference in capacity curve relative to an early healthy
  cycle. Then extract 10 statistical features from this difference
  curve. These features are the inputs to the ML model.

  The key insight: Rather than predicting SOH directly from raw
  measurements, the features capture how the Q(V) curve has changed
  from the reference. This is physics-motivated — degradation
  mechanisms leave characteristic signatures in how the
  capacity-voltage curve deforms.

  FeatureConfig — configuration dataclass:
  - Defines the shared voltage grid (v_min, v_max, dv — default 2.5V to
  4.2V at 5mV).
  - SG smoothing parameters, CC gating parameters.
  - half_cycle: which half-cycle to use (default "discharge").
  - reference_cycle_rank: which valid cycle to use as reference
  (default: first valid one).
  - A config_hash() method (SHA-256 of all fields) for cache
  invalidation and provenance.

  QVExtractor — extracts a smooth Q(V) curve from a raw half-cycle
  DataFrame:

  1. CC region selection with 3 gates:
    - Gate 1: C-rate ≤ c_rate_max (default ~C/10)
    - Gate 2: |dI/dt| ≤ 5× median |dI/dt| — rejects CV tails and
  current pulses
    - Gate 3: rolling coefficient of variation of |I| ≤ current_cv_max
  (default 2%) — rejects unstable current
    - If less than 20% of points pass, the cycle is rejected.
  2. Coulomb counting via cumsum(|I| × dt) / 3600.
  3. Monotonic voltage filter: For discharge, keeps only V ≤ previous V
  + 1 mV tolerance. Also deduplicates at 0.1 mV resolution.
  4. PCHIP interpolation onto the shared voltage grid (no extrapolation
  → NaN outside measured range).
  5. NaN inpainting: Linear interpolation across interior NaNs; edge
  NaNs → 0 before SG filter.
  6. SG smoothing + dQ/dV (single SG pass with deriv=1, delta=dv).
  7. Edge trimming and NaN restoration (grid positions outside the
  measured voltage range stay NaN).

  DeltaQVComputer — computes Q_cycle - Q_reference:
  - fit_reference() selects the Nth valid Q(V) curve by cycle number.
  - set_reference() accepts an externally supplied reference (e.g.,
  from a cross-cell baseline).
  - compute() simply subtracts arrays, propagating NaN where either
  curve has NaN.

  FeatureExtractor — computes 10 statistical features from ΔQ(V):
  - Operates only on finite values (NaNs excluded).
  - Clips outliers beyond ±5σ before computation.
  - Features: variance, log-variance, skewness, kurtosis, integral of
  |ΔQ(V)| (×dV), max deviation, min, max, mean, RMS.
  - Chemistry-specific extras for LFP/NMC (range, positive/negative
  integrals).
  - SOH leakage guard: a hard-coded blocklist (_SOH_PROXY_NAMES)
  prevents feature names like discharge_capacity_ah from sneaking in.

  FeatureMatrixBuilder — orchestrates everything:
  - Groups records by cell_id.
  - For each cell: extracts all Q(V) curves, selects a reference,
  computes ΔQ(V) for every cycle, extracts features.
  - Output DataFrame has metadata columns + the 10 feature columns.
  - Rows where extraction fails get extraction_ok=False and NaN
  features.

  CellGroupKFold — a custom cross-validation splitter that keeps all
  cycles from one cell in the same fold. This prevents the leakage that
  would occur with random splits (because cycles from the same cell
  are autocorrelated in time — an early and late cycle from the same
  cell look similar enough to inflate generalization estimates).

  make_soh_pipeline() — builds a sklearn Pipeline:
  - NaNFlagImputer: appends binary NaN-indicator columns, then
  median-imputes. Fit on training data only.
  - OutlierClipper: clips to ±n_sigma using training mean/std.
  - RobustScaler (or StandardScaler).
  - Optional final estimator.

  Persistence (save_pipeline / load_pipeline): saves the pipeline +
  FeatureConfig + hash as a joblib bundle. Rejects stale artifacts
  where the recomputed hash doesn't match the stored hash. Also writes
  a feature_schema.json sibling.

  ---
  step4_soh_model.py — ElasticNet SOH model

  Purpose: Train an ElasticNet regression model to predict SOH from the
  10 ΔQ(V) features, using Leave-One-Cell-Out (LOCO) cross-validation
  to ensure predictions generalize to unseen cells.

  SOHModelTrainer:

  - load_features(): Reads the step3 CSV, applies COLUMN_ALIASES
  renames, validates the schema.
  - _build_pipeline(): StandardScaler → ElasticNet(alpha=1e-3,
  l1_ratio=0.5). ElasticNet is a mix of L1 (Lasso, forces some
  coefficients to zero) and L2 (Ridge) regularization. At l1_ratio=0.5,
  it blends both.
  - _loco_splits(): Generator that yields (train_df, test_df, cell_id)
  — each test set is one cell, training uses all other cells.
  - fit(): Runs LOCO CV and collects held-out predictions. Then trains
  a second final model on all data for serialization and feature
  importance. Metrics come only from LOCO predictions, never from the
  final model's in-sample predictions.
  - save(): Writes 5 artifacts: elasticnet_soh.joblib,
  feature_schema.json, metrics.json, predictions.csv,
  feature_importance.csv.
  - run(): Load → fit → save in one call.

  ModelResults dataclass: structured output with predictions DataFrame,
  overall metrics (RMSE/MAE/R²), per-cell metrics, feature importance
  table, and artifact paths.

  ---
  step4_interpretation.py — Physics-based degradation interpretation

  Purpose: Given ICA curves (dQ/dV vs. V), compute physics-motivated
  degradation descriptors that explain why a battery is degrading, not
  just how much. The three mechanisms tracked are:

  - LLI (Loss of Lithium Inventory): lithium is consumed by SEI growth
  or plating. ICA peaks shift in voltage.
  - LAM (Loss of Active Material): electrode particles crack or lose
  contact. The integrated area under the ICA curve shrinks.
  - Resistance growth: impedance increases. ICA peaks broaden.

  extract_peaks(): Uses scipy.signal.find_peaks with prominence and
  minimum width thresholds. Peak widths are returned in millivolts (not
  sample indices) — conversion is widths_mV = raw_sample_widths ×
  grid_spacing_mV. This makes the metric grid-spacing-invariant.

  compute_lli() / _match_peaks(): Matches reference-cycle peaks to
  current-cycle peaks using the Hungarian algorithm
  (scipy.optimize.linear_sum_assignment). This finds the globally
  optimal one-to-one assignment. Pairs further apart than 20 mV are
  treated as disappeared/appeared rather than shifted. Returns a
  LLIResult with mean shift, match confidence, disappearance penalty.

  compute_lam(): 1 - (curr_area / ref_area) using Simpson's rule. Zero
  at the reference cycle, positive and increasing as material is lost.

  compute_resistance_growth(): Mean fractional increase in peak width:
  mean((curr_widths - ref_widths) / ref_widths). Both inputs must be in
  mV (not raw sample indices), enforced by assertions.

  interpret_cycle(): Calls all three, shares the grid spacing
  computation, and returns a flat metrics dict plus the full LLIResult.

  build_physics_features(): Applies interpret_cycle() over all cycles
  in a cell-grouped ICA DataFrame. For each cell, finds the reference
  row (by is_reference flag or cycle_number == reference_cycle).

  compute_shift_trajectory(): Fits a linear trend to peak shifts across
  cycles (np.polyfit). Reports shift_rate_per_cycle and detects when
  confidence drops below 0.5 (onset of severe degradation).

  plot_peak_broadening_evolution(): Matplotlib plot of peak width vs.
  cycle number (optionally normalized to the first cycle). Width values
  are grid-spacing-invariant, so cells processed at different ICA grid
  resolutions are directly comparable.

  ---
  predict.py — Inference-time predictor (Flow 2)

  Purpose: Load a trained model and predict SOH for a single new
  half-cycle, plus generate a natural-language explanation.

  SOHPredictor:
  - Constructed with a trained pipeline, a FeatureConfig, and optional
  nominal capacity.
  - from_model(): class method that loads a joblib pipeline file.
  - set_reference(): processes an early-life reference half-cycle
  through QVExtractor. Must succeed (raises ValueError if CC extraction
  fails).
  - set_reference_from_q(): directly supplies pre-extracted Q(V)
  arrays.
  - assess(): runs QVExtractor on the current cycle, computes ΔQ(V) =
  q_smooth - ref_q_smooth, extracts features, calls pipeline.predict(),
  then calls interpret_cycle() for physics metrics. Returns a dict
  with soh_pred, features, physics, explanation, extraction_info.
  - assess_trajectory(): runs assess() over a list of half-cycles and
  returns a DataFrame.

  _explain(): Generates a human-readable paragraph from physics
  metrics:
  - If LLI confidence > 50% and shift > 3 mV → reports direction and
  magnitude.
  - If peaks disappeared → reports Li plating or SEI growth suspicion.
  - If LAM > 2% → reports active material loss percentage.
  - If resistance growth > 5% → reports impedance increase.

  Lazy module loading: Because step3_deltaQ(V)_feature_extraction.py
  has parentheses in its filename (non-standard), predict.py uses
  importlib.util.spec_from_file_location() to load it by path instead
  of by module name. A _MODULE_CACHE dict prevents reloading.

  ---
  test_e2e_pipeline.py — Integration test suite

  Tests all 7 sections end-to-end using synthetic data only (no real
  .mat file required):

  1. Step1 parsing: cycle detection, discharge half-cycles present,
  capacity decreasing over cycles, correct dataset attributes, error
  handling for bad formats/missing files.
  2. Step3 feature extraction: feature matrix produced, column names
  match schema, finite features for valid cycles, SOH leakage check,
  reference cycle fixed per cell.
  3. Step4 model training: loads features, LOCO CV runs, predictions
  cover all cells, metrics finite, RMSE < 0.5, feature importance has
  correct columns, model saves/reloads, predictions CSV written.
  4. Physics interpretation: DataFrame returned, required columns
  present, LAM increases over cycles, resistance growth non-negative
  for broadening peaks, interpret_cycle returns all keys, LLI shift
  sign correct.
  5. Flow 1 end-to-end: full chain CSV → features → model → physics →
  report.
  6. Flow 2 single cycle: SOH prediction keys present, numeric and
  finite, in plausible range, aged < healthy, explanation is non-empty
  string mentioning "SOH", physics has LLI/LAM/resistance, trajectory
  trends downward.
  7. Dataset loaders: Oxford (.mat) and Severson (.pkl) loader
  registry, extension dispatch, error handling, SOH computation from
  QDischarge.
  8. Predict module unit tests: model loading, reference
  acceptance/rejection, nonexistent model error, physics keys finite.
  9. Schema contract: step3 output accepted by step4 trainer, legacy
  column names work after aliasing.

  ---
  How the files connect (data flow)

  Oxford .mat file
         │
    step1_data_parsing.py
    ├─ raw → clean → cycles (dict)
    └─ ParsedDataset
         │
    step2_Q(v)_extraction.py  ← optional, ICA for low-rate research
  cycles
    └─ ICACurve (dQ/dV on uniform voltage grid)
         │
    step3_deltaQ(V)_feature_extraction.py
    ├─ QVExtractor: raw half-cycle → Q(V) smooth + dQ/dV
    ├─ DeltaQVComputer: Q_cycle - Q_reference
    └─ FeatureExtractor: 10 statistical features
         │
  test_e2e_pipeline.py — Integration test suite

  Tests all 7 sections end-to-end using synthetic data only (no real .mat file required):

  1. Step1 parsing: cycle detection, discharge half-cycles present, capacity decreasing over
  cycles, correct dataset attributes, error handling for bad formats/missing files.
  2. Step3 feature extraction: feature matrix produced, column names match schema, finite
  features for valid cycles, SOH leakage check, reference cycle fixed per cell.
  3. Step4 model training: loads features, LOCO CV runs, predictions cover all cells, metrics
  finite, RMSE < 0.5, feature importance has correct columns, model saves/reloads, predictions
  CSV written.
  4. Physics interpretation: DataFrame returned, required columns present, LAM increases over
  cycles, resistance growth non-negative for broadening peaks, interpret_cycle returns all keys,
  LLI shift sign correct.
  5. Flow 1 end-to-end: full chain CSV → features → model → physics → report.
  6. Flow 2 single cycle: SOH prediction keys present, numeric and finite, in plausible range,
  aged < healthy, explanation is non-empty string mentioning "SOH", physics has
  LLI/LAM/resistance, trajectory trends downward.
  7. Dataset loaders: Oxford (.mat) and Severson (.pkl) loader registry, extension dispatch,
  error handling, SOH computation from QDischarge.
  8. Predict module unit tests: model loading, reference acceptance/rejection, nonexistent model
  error, physics keys finite.
  9. Schema contract: step3 output accepted by step4 trainer, legacy column names work after
  aliasing.

  ---
  How the files connect (data flow)

  Oxford .mat file
         │
    step1_data_parsing.py
    ├─ raw → clean → cycles (dict)
    └─ ParsedDataset
         │
    step2_Q(v)_extraction.py  ← optional, ICA for low-rate research cycles
    └─ ICACurve (dQ/dV on uniform voltage grid)
         │
    step3_deltaQ(V)_feature_extraction.py
    ├─ QVExtractor: raw half-cycle → Q(V) smooth + dQ/dV
    ├─ DeltaQVComputer: Q_cycle - Q_reference
    └─ FeatureExtractor: 10 statistical features
         │
    feature_schema.py ← contract (FEATURE_COLS) shared by step3 and step4
         │
    step4_soh_model.py
    ├─ ElasticNet + StandardScaler
    ├─ Leave-One-Cell-Out CV → predictions, metrics
    └─ elasticnet_soh.joblib artifact
         │
    step4_interpretation.py
    ├─ LLI (peak shift, Hungarian matching)
    ├─ LAM (area loss)
    └─ Resistance growth (peak broadening)
         │
    predict.py (Flow 2)
    └─ SOHPredictor: reference + current cycle → soh_pred + explanation

  Every artifact (parsed cache, ICA cache, trained pipeline, feature schema JSON) includes a
  config hash so stale artifacts from a different configuration are detected and rejected.