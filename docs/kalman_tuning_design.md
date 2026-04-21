# Kalman OLTW Tuning Design

## Summary

This document proposes a user-facing tuning system for `kalman_oltw` that lets a researcher:

- start from a stable preset
- override DP and Kalman parameters without editing code
- run one-off tuned benchmarks from the existing CLI
- save repeatable tuned runs in a config file
- launch controlled sweep jobs for broader exploration

The design preserves the public benchmark method name `kalman_oltw`. Tuning changes the resolved architecture and tracker parameters, not the method registration model.

## Goals

- Make DP and Kalman tuning accessible from the benchmark path.
- Keep presets as stable experiment anchors.
- Record the fully resolved tuning parameters in benchmark outputs.
- Support quick manual iteration, reproducible saved configs, and larger controlled sweeps.
- Add a `medium` benchmark mode that is about 10x larger than `small` while staying fast enough for development.

## Non-Goals

- Replacing the preset system with ad hoc parameter blobs.
- Introducing a new benchmark method name for every tuned variant.
- Adding automated Bayesian optimization or distributed search in the first pass.

## Current State

The repo already has:

- implemented `kalman_oltw` presets in [scripts/kalman_online.py](/home/diego/School/SP26/E207_project/scripts/kalman_online.py)
- benchmark execution in [scripts/run_benchmark.py](/home/diego/School/SP26/E207_project/scripts/run_benchmark.py)
- benchmark selection logic in [scripts/evaluation.py](/home/diego/School/SP26/E207_project/scripts/evaluation.py)
- `small` mode fixed at 3 recordings from one piece, yielding 6 directed pairs

The missing piece is a systematic way for a user to modify DP and Kalman knobs on top of a preset without changing source code each time.

## Core Design

### Configuration Layers

Tuning should use a strict precedence order:

1. Built-in preset defaults
2. Config-file overrides
3. Explicit CLI flag overrides

This lets the same preset be reused across:

- quick terminal experiments
- saved named tuning configs
- scripted sweeps

### Override Model

Add a new dataclass in [scripts/kalman_online.py](/home/diego/School/SP26/E207_project/scripts/kalman_online.py):

```python
@dataclass(slots=True)
class KalmanOLTWOverrides:
    step_pattern_name: str | None = None
    hold_reference_weight: float | None = None
    diagonal_weight: float | None = None
    query_double_weight: float | None = None
    reference_double_weight: float | None = None

    min_search_half_window: int | None = None
    uncertainty_scale: float | None = None
    recovery_search_half_windows: tuple[int, ...] | None = None
    low_confidence_margin_threshold: float | None = None
    proactive_innovation_z_threshold: float | None = None

    position_prior_weight: float | None = None
    downweight_margin_threshold: float | None = None
    skip_margin_threshold: float | None = None
    downweight_innovation_z_threshold: float | None = None
    skip_innovation_z_threshold: float | None = None
    downweight_measurement_variance_scale: float | None = None

    position_variance: float | None = None
    velocity_variance: float | None = None
    process_position_variance: float | None = None
    process_velocity_variance: float | None = None
    measurement_variance: float | None = None
    min_velocity: float | None = None
    max_velocity: float | None = None
```

This object should be optional everywhere. If it is omitted, the current behavior remains unchanged.

### Resolved Architecture Flow

Refactor the architecture builder to:

1. Resolve the selected preset
2. Apply `KalmanFilterConfig`
3. Apply `KalmanOLTWOverrides`
4. Return the final `KalmanOLTWArchitecture`

Recommended API shape:

```python
def build_kalman_oltw_architecture(
    config: KalmanFilterConfig | None = None,
    preset_name: str = "baseline_cv",
    overrides: KalmanOLTWOverrides | None = None,
) -> KalmanOLTWArchitecture:
    ...
```

`run_kalman_oltw(...)` should accept:

```python
def run_kalman_oltw(
    ...,
    preset_name: str = "baseline_cv",
    overrides: KalmanOLTWOverrides | None = None,
) -> AlignmentResult:
```

## Tunable Parameters

### DP / Online Alignment Knobs

- step pattern choice
  - `default_normalized_v1`
  - `hold_diag_only`
- transition weights
  - `hold_reference_weight`
  - `diagonal_weight`
  - `query_double_weight`
  - `reference_double_weight`
- search window
  - `min_search_half_window`
  - `uncertainty_scale`
- staged recovery
  - `recovery_search_half_windows`
  - `low_confidence_margin_threshold`
  - `proactive_innovation_z_threshold`
- coupling
  - `position_prior_weight`
- confidence gating
  - `downweight_margin_threshold`
  - `skip_margin_threshold`
  - `downweight_innovation_z_threshold`
  - `skip_innovation_z_threshold`
  - `downweight_measurement_variance_scale`

### Kalman Knobs

- `position_variance`
- `velocity_variance`
- `process_position_variance`
- `process_velocity_variance`
- `measurement_variance`
- `min_velocity`
- `max_velocity`

## UX Option 1: Direct CLI Flags

### User Experience

The user runs one benchmark with explicit overrides:

```bash
python -m scripts.run_benchmark \
  --method kalman_oltw \
  --mode small \
  --kalman-preset hold_diag_narrow_confidence \
  --measurement-variance 16.0 \
  --process-position-variance 2.0 \
  --process-velocity-variance 0.02 \
  --min-search-half-window 96 \
  --low-confidence-margin-threshold 0.01 \
  --skip-innovation-z-threshold 4.5
```

### Implementation

Add optional CLI arguments in [scripts/run_benchmark.py](/home/diego/School/SP26/E207_project/scripts/run_benchmark.py) for the highest-value knobs:

- preset selection
- step pattern choice
- search width
- recovery thresholds
- coupling thresholds
- Kalman process noise
- Kalman measurement noise
- velocity bounds

Convert the parsed flags into `KalmanOLTWOverrides`, pass it through `runner_kwargs`, and leave all unset fields as `None`.

### Pros

- Fastest iteration loop
- Minimal ceremony
- Works well for one-off experiments

### Cons

- Long commands become hard to manage
- Harder to reuse a tuning setup exactly

## UX Option 2: Config File

### User Experience

The user stores a named tuning configuration:

```yaml
preset_name: hold_diag_narrow_confidence
overrides:
  min_search_half_window: 96
  uncertainty_scale: 2.0
  low_confidence_margin_threshold: 0.01
  process_position_variance: 2.0
  process_velocity_variance: 0.02
  measurement_variance: 16.0
```

and runs:

```bash
python -m scripts.run_benchmark \
  --method kalman_oltw \
  --mode medium \
  --kalman-config configs/kalman/hold_diag_tune_01.yaml
```

### File Format

Support JSON first because it requires no new dependency. Support YAML second if `PyYAML` is available in the environment.

Recommended schema:

```yaml
preset_name: hold_diag_narrow_confidence
description: Narrow-window confidence-gated trial
feature_name: chroma_stft
sample_rate: 22050
benchmark_mode: medium
overrides:
  min_search_half_window: 96
  uncertainty_scale: 2.0
  recovery_search_half_windows: [192]
  low_confidence_margin_threshold: 0.01
  proactive_innovation_z_threshold: 3.0
  position_prior_weight: 0.05
  downweight_margin_threshold: 0.02
  skip_margin_threshold: 0.005
  process_position_variance: 2.0
  process_velocity_variance: 0.02
  measurement_variance: 16.0
```

### Implementation

Add a small loader module, for example:

- [scripts/kalman_tuning.py](/home/diego/School/SP26/E207_project/scripts/kalman_tuning.py)

Responsibilities:

- load JSON or YAML
- validate field names and types
- build `KalmanOLTWOverrides`
- return a normalized config object

CLI precedence should remain:

- preset defaults
- config file values
- explicit CLI flags

### Pros

- Reproducible
- Easy to version in git
- Better for collaborative comparison

### Cons

- Slightly more setup than pure CLI tuning

## UX Option 3: Sweep Runner

### User Experience

The user writes a sweep config:

```yaml
benchmark_mode: medium
trials:
  - name: hdn_conf_r9
    preset_name: hold_diag_narrow_confidence
    overrides:
      measurement_variance: 9.0
      process_position_variance: 2.0
  - name: hdn_conf_r16
    preset_name: hold_diag_narrow_confidence
    overrides:
      measurement_variance: 16.0
      process_position_variance: 2.0
```

and runs:

```bash
python -m scripts.run_kalman_sweep --config configs/kalman/sweep_01.yaml
```

### Implementation

Add a new script:

- [scripts/run_kalman_sweep.py](/home/diego/School/SP26/E207_project/scripts/run_kalman_sweep.py)

Responsibilities:

- load the sweep config
- iterate over named trials
- call `evaluation.run_alignment_benchmark(...)`
- collect all metrics rows into one table
- write a summary CSV with one row per trial
- optionally write per-pair CSVs for the best trial only

The sweep script should not invent a new execution path. It should call the same benchmark runner used by the CLI.

### Sweep Output

For each trial, record:

- `trial_name`
- `architecture_preset`
- flattened override values
- aggregate benchmark metrics
- runtime summary

Recommended outputs:

- `*_trial_summary.csv`
- `*_pair_metrics.csv`
- `*_best_trial.json`

### Pros

- Best workflow for structured tuning
- Easy to compare a grid of DP and Kalman settings
- Keeps comparisons consistent

### Cons

- More code than the first two options

## Validation Rules

Validation should happen before the benchmark starts.

Examples:

- `min_search_half_window > 0`
- `uncertainty_scale > 0`
- `recovery_search_half_windows` must be positive ints
- `skip_margin_threshold <= downweight_margin_threshold`
- `skip_innovation_z_threshold >= downweight_innovation_z_threshold`
- `measurement_variance > 0`
- `process_position_variance >= 0`
- `process_velocity_variance >= 0`
- `min_velocity > 0`
- `max_velocity >= min_velocity`
- double-step weights may only be set if the chosen step pattern includes them

Invalid combinations should raise a clear `ValueError` with the parameter name.

## Benchmark Output Requirements

Every tuned run should record:

- `architecture_preset`
- `tuning_source`
  - `preset_only`
  - `config_file`
  - `cli_overrides`
  - `config_plus_cli`
- scalar resolved parameters as flat metadata columns
- one serialized JSON blob with the full resolved tuning state

Recommended metadata fields:

- `resolved_step_pattern_name`
- `resolved_min_search_half_window`
- `resolved_uncertainty_scale`
- `resolved_low_confidence_margin_threshold`
- `resolved_process_position_variance`
- `resolved_process_velocity_variance`
- `resolved_measurement_variance`
- `resolved_min_velocity`
- `resolved_max_velocity`
- `resolved_tuning_json`

Flat scalar columns keep CSV analysis easy. The JSON blob preserves the full run configuration.

## Medium Benchmark Design

### Goal

Add a benchmark mode that is about 10x larger than `small` while remaining development-friendly.

### Size Target

`small` uses 3 recordings from one piece:

- 3 recordings
- 6 directed pairs

The proposed `medium` mode uses 8 recordings from one piece:

- 8 recordings
- 56 directed pairs

This is about 9.3x `small`, which is close enough to the stated target and keeps the selection logic simple and deterministic.

### Selection Policy

Add a new selection mode:

- `medium`

Add a constant in [scripts/evaluation.py](/home/diego/School/SP26/E207_project/scripts/evaluation.py):

```python
MEDIUM_BENCHMARK_RECORDING_COUNT = 8
```

Implement `_select_medium_benchmark_pairs(...)` using the same policy shape as `small`:

1. Group recordings by piece
2. Keep only pieces with at least 8 annotated recordings
3. Sort recordings within each piece by duration
4. Choose the 8 shortest recordings from the fastest eligible piece
5. Return all directed pairs among those 8 recordings
6. Sort selected pairs by `_pair_duration_key`

This keeps `medium` comparable to `small`:

- same single-piece development focus
- same duration-based determinism
- higher case count

### CLI and API Changes

Update `BenchmarkSelectionMode` to include `medium`.

Update [scripts/run_benchmark.py](/home/diego/School/SP26/E207_project/scripts/run_benchmark.py) choices to include `medium`.

Update default experiment naming:

- `kalman_oltw_medium`
- `kalman_oltw_hold_diag_only_medium`
- `kalman_oltw_hold_diag_narrow_confidence_medium`

Update README examples and benchmark descriptions.

### Error Handling

If no piece has at least 8 recordings, raise:

```text
Medium benchmark mode requires at least eight annotated recordings for one piece.
```

### Why Single-Piece Medium Instead of Multi-Piece Medium

Single-piece medium is preferred for the first implementation because it:

- keeps comparisons closer to `small`
- limits cross-piece variability during tuning
- makes regressions easier to interpret
- reuses the current selection philosophy

If a broader development set is needed later, that should be a separate mode rather than overloading `medium`.

## Code Changes

### [scripts/kalman_online.py](/home/diego/School/SP26/E207_project/scripts/kalman_online.py)

- add `KalmanOLTWOverrides`
- add helper to apply overrides onto a resolved architecture
- add validation for override combinations
- update `run_kalman_oltw(...)` to accept overrides
- flatten resolved tuning metadata into `AlignmentResult.metadata`

### [scripts/run_benchmark.py](/home/diego/School/SP26/E207_project/scripts/run_benchmark.py)

- add explicit tuning flags
- add `--kalman-config`
- add `medium` to `--mode`
- build `KalmanOLTWOverrides` from CLI arguments
- merge config-file and CLI overrides

### [scripts/evaluation.py](/home/diego/School/SP26/E207_project/scripts/evaluation.py)

- add `medium` to `BenchmarkSelectionMode`
- add `MEDIUM_BENCHMARK_RECORDING_COUNT`
- implement `_select_medium_benchmark_pairs(...)`
- route `selection_mode="medium"` through `select_recording_pairs(...)`

### New Module: [scripts/kalman_tuning.py](/home/diego/School/SP26/E207_project/scripts/kalman_tuning.py)

- config-file loader
- override parser
- validation helpers
- metadata flattening helpers

### New Module: [scripts/run_kalman_sweep.py](/home/diego/School/SP26/E207_project/scripts/run_kalman_sweep.py)

- trial runner
- summary writer
- best-trial promotion helper

## Testing Plan

Add tests for:

- override parsing from CLI
- override parsing from config file
- precedence: preset < config file < CLI flags
- validation of bad combinations
- `run_kalman_oltw(...)` metadata includes resolved tuning parameters
- `medium` selection returns 56 directed pairs on a dataset with at least 8 recordings
- `medium` mode errors cleanly when the dataset is too small
- sweep runner writes deterministic trial summaries

Recommended concrete test additions:

- extend [tests/test_milestone1_pipeline.py](/home/diego/School/SP26/E207_project/tests/test_milestone1_pipeline.py) for benchmark selection and CLI naming
- add a focused test file such as [tests/test_kalman_tuning.py](/home/diego/School/SP26/E207_project/tests/test_kalman_tuning.py)

## Suggested Rollout Order

1. Add `KalmanOLTWOverrides` and resolved metadata flattening
2. Add CLI tuning flags
3. Add config-file support
4. Add `medium` benchmark mode
5. Add sweep runner

This order gives immediate value after step 2 while keeping the later workflow additions incremental.
