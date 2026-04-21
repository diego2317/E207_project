# Next Steps

## Goal
Improve `kalman_oltw` by fixing the main source of error first: unstable DP-generated measurements. The current evidence says the biggest gains will come from changing the online DP and measurement coupling before spending time on Kalman parameter sweeps.

## What The Ablation Showed
- The baseline `kalman_oltw` is failing mostly because the DP produces bad measurements, not because the Kalman filter is slightly mis-tuned.
- Removing aggressive reference-advance behavior helps a lot.
  `hold_diag_only` reduced average `small`-benchmark MAE from about `11.7s` to about `1.0s`.
- Raising measurement noise globally does not help.
  Larger `R` made results worse, even after the step pattern improved.
- A narrower search window is promising once the step pattern is stabilized.
  It did not improve average MAE much, but it improved `within_100ms` and `within_200ms`.

## Recommended Order
1. Fix the DP transition set.
2. Improve the search policy and recovery behavior.
3. Improve measurement coupling.
4. Tune the Kalman filter.
5. Re-evaluate whether constant-velocity is still the right tracker model.

## Concrete Next Steps

### 1. Make the safer step pattern the default experimental path
- Remove `reference_double` from the active experiment path.
- Keep only `hold_reference` and `diagonal` for the next round.
- Treat this as the control variant for future comparisons.

### 2. Add an explicit preset for the current best variant
- Add a real implemented preset such as `hold_diag_only`.
- Make sure it is selectable through the same benchmark path as `kalman_oltw`.
- Record the preset name in benchmark outputs and metadata.

### 3. Implement narrower search with recovery expansion
- Start from a narrower default search window than the current baseline.
- Add recovery expansion when the row becomes low-confidence or the innovation spikes.
- Keep recovery explicit in metadata so it is measurable.

### 4. Improve measurement handling
- Stop treating every row argmin as equally trustworthy.
- Add confidence-aware update logic using row sharpness or margin.
- Skip or down-weight updates when the row is flat or when innovation is implausibly large.
- Prefer selective measurement trust over globally increasing `R`.

### 5. Run a second ablation after those changes
- Compare:
  `baseline_cv`
- Compare:
  `hold_diag_only`
- Compare:
  `hold_diag_only + narrow window`
- Compare:
  `hold_diag_only + narrow window + confidence gating`
- Evaluate on `small` first, then promote the best candidate to `paper_test`.

### 6. Tune the Kalman filter only after the measurement stream is stable
- Tune `measurement_variance`
- Tune `process_position_variance`
- Tune `process_velocity_variance`
- Tune velocity bounds
- Check whether the filter is lagging, overshooting, or freezing on tempo changes

### 7. Revisit the tracker model after the above
- Keep constant-velocity as the control model.
- Only try modified Kalman variants after the DP and coupling are improved.
- Candidate follow-ups:
  adaptive-noise constant velocity
- Candidate follow-ups:
  reset-on-persistent-innovation logic
- Candidate follow-ups:
  constant-acceleration state

## Practical Benchmark Plan
- Use `small` for fast iteration and pairwise diagnostics.
- For every run, inspect:
  `mean_abs_error_s`
- For every run, inspect:
  `rmse_s`
- For every run, inspect:
  `within_100ms`
- For every run, inspect:
  `within_200ms`
- For every run, inspect:
  `mean_abs_innovation`
- For every run, inspect:
  `mean_search_width`
- For every run, inspect:
  `mean_measurement_score`
- For every run, inspect:
  transition usage
- Once one variant is clearly better on `small`, run `paper_test`.

## Command To Re-Run The Current Baseline
```bash
python -m scripts.run_benchmark --method kalman_oltw --mode small
```

## Main Principle
Tune bias before variance.

In this system:
- The DP mostly determines whether the measurements are biased.
- The Kalman filter mostly determines how noisy or smooth the final track is.

If the DP keeps producing bad measurements, no reasonable Kalman tuning will rescue the method.
