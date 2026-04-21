# Milestone 2 Iteration Plan

This document records the current `kalman_oltw` baseline, the scaffolding added for future iterations, and the controlled experiment matrix for improving the method without turning the next phase into an unstructured parameter sweep.

## Current Baseline

`kalman_oltw` is currently a native Python streaming normalized online-DTW runner with:

- a constant-velocity Kalman filter in reference-frame units
- an uncertainty-scaled search window around the predicted reference position
- a position prior added inside the DP row update
- a normalized causal measurement extracted as the row argmin
- a default step pattern with hold, diagonal, query-double, and reference-double transitions

The current implementation is intentionally the control system for future comparisons, not the presumed final Milestone 2 solution.

## Scaffolding Added

The code now exposes explicit architecture components in `scripts.kalman_online`:

- `SearchPolicyConfig`
- `StepPatternConfig`
- `TrackerConfig`
- `CouplingConfig`
- `KalmanOLTWArchitecture`
- `KalmanExperimentPreset`
- `KalmanOnlineTrace`

The compatibility entry point remains `KalmanFilterConfig`, but it now builds the current architecture bundle through `to_architecture()`. This preserves the existing benchmark surface while making future changes local to one subsystem at a time.

The runner also emits summary diagnostics in `AlignmentResult.metadata`, including:

- search-window widths
- innovation statistics
- recovery-window activations
- row-confidence margins
- aggregate transition usage

## Planned Experiment Presets

The preset registry currently documents the following comparison targets:

- `baseline_cv`: the current production baseline
- `adaptive_noise_cv`: constant-velocity with adaptive process and measurement noise
- `constant_acceleration`: linear acceleration-aware Kalman variant
- `narrow_window`: tighter gating stress test
- `wide_window`: wider gating and more aggressive recovery
- `hold_diag_only`: minimal transition set without the double-step moves
- `symmetric_doubles`: retuned double-step transition experiment

Only `baseline_cv` is implemented today. The others exist as named planned presets so future benchmark results can be reported against stable experiment labels.

## Experiment Matrix

### Search Policy

Compare:

- fixed-width windows
- uncertainty-scaled windows
- forward-biased asymmetric windows
- recovery expansion triggered by flat rows, large innovations, or unreachable windows

Primary diagnostics:

- average and maximum search width
- recovery activation count
- row finite-candidate counts

### Transition Set

Compare:

- the current four-transition baseline
- hold-plus-diagonal only
- retuned symmetric double-step variants
- slope-constrained variants if the double steps remain unstable

Primary diagnostics:

- aggregate transition usage
- path smoothness
- benchmark accuracy on small and paper-test subsets

### Kalman Coupling

Compare:

- search gating only
- position prior only
- gate plus prior
- confidence-aware coupling with row-derived noise adaptation
- soft measurements derived from a local weighted neighborhood instead of a hard row argmin

Primary diagnostics:

- innovation magnitude
- measurement-score sharpness
- recovery frequency

## Tracker Decision Criteria

The tracker decision should remain evidence-based:

- Keep constant-velocity only if it remains competitive after the search and coupling improvements and does not show systematic lag during tempo changes.
- Prefer adaptive-noise constant-velocity if failures are driven mostly by unreliable measurements rather than by genuine state-model mismatch.
- Prefer constant-acceleration only if accelerando and ritardando segments consistently improve without destabilizing easy same-performer pairs.

Do not jump to more complex tracker families until the structured comparison among these linear Kalman variants is complete.

## Next Implementation Sequence

1. Keep `baseline_cv` fixed as the comparison anchor.
2. Implement one search-policy variant and one diagnostic notebook or report path.
3. Implement one transition-set variant and compare it against the unchanged baseline.
4. Implement adaptive-noise constant-velocity before adding acceleration state.
5. Introduce constant-acceleration only if the diagnostics still indicate systematic tempo-change lag.

Every future change should preserve the public benchmark method name `kalman_oltw` unless the objective function itself changes.
