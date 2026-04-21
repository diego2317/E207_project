# E207 Project

Milestone 1 implementation for the Real-Time Alignment & Tracking project.

## Layout

- `scripts/`: reusable modules for benchmark discovery, feature extraction, offline/online baseline adapters, evaluation, and plotting
- `notebooks/`: exploratory notebook for running the alignment benchmark pipeline
- `data/`: local benchmark assets and derived datasets
- `outputs/`: figures, metrics, and logs produced during experiments
- `tests/`: smoke and pipeline tests for the Milestone 1 workflow

## Data Layout

The benchmark loader supports either:

1. A manifest file at `data/raw/mazurka_manifest.csv` or `data/raw/manifest.csv`
2. The Mazurka split layout used in this repo:
   `data/raw/wav_22050_mono/<piece>/*.wav` and `data/raw/annotations_beat/<piece>/*.beat`
3. An inferred directory layout with one audio file and one beat file per recording directory

Recommended manifest columns:

- `piece`
- `recording_id`
- `audio_path`
- `beats_path`

Example:

```csv
piece,recording_id,audio_path,beats_path
mazurka_op24_no2,performance_a,mazurka_op24_no2/performance_a/audio.wav,mazurka_op24_no2/performance_a/beats.csv
mazurka_op24_no2,performance_b,mazurka_op24_no2/performance_b/audio.wav,mazurka_op24_no2/performance_b/beats.csv
```

## Getting Started

Use a dedicated Miniconda environment for this project. Do not install the
project dependencies into the conda `base` environment.

1. Create the project environment:

   ```bash
   conda env create -f environment.yml
   ```

2. Activate it:

   ```bash
   conda activate e207-bench
   ```

3. Verify the interpreter version:

   ```bash
   python --version
   ```

   Expected: `Python 3.10.10`

4. Place benchmark data under `data/raw/`.
5. Run `pytest` to verify the pipeline.
6. Use `notebooks/01_alignment_sandbox.ipynb` for exploratory testing.

If you need a notebook kernel, register the conda environment explicitly:

```bash
python -m ipykernel install --user --name e207-bench --display-name "Python 3.10 (e207-bench)"
```

## Benchmark Entry Points

- `scripts.evaluation.evaluate_recording_pair(pair, method_name=...)` runs one alignment method on one benchmark case and computes beat-based query-to-reference tracking metrics.
- `scripts.evaluation.run_alignment_benchmark(...)` discovers recordings, builds directed same-piece benchmark cases, selects a run mode, and writes metrics summaries to `outputs/metrics/`.
- `scripts.evaluation.run_offline_benchmark(...)` remains as a compatibility wrapper for the offline DTW baseline.
- `python -m scripts.run_benchmark --method offline_dtw --mode single --pair-id <reference__query>` runs one directed benchmark case.
- `python -m scripts.run_benchmark --method offline_dtw --mode small` runs the fixed 3-recording preview benchmark (6 directed cases from the shortest eligible piece).
- `python -m scripts.run_benchmark --method offline_dtw --mode full` runs the full directed benchmark set.
- `python -m scripts.run_benchmark --method offline_dtw --mode paper_test --max-pairs 1000` runs the held-out `paper_test` subset but stops after the first 1000 selected directed cases.
- `python -m scripts.run_benchmark --method oltw --mode small` runs the default `PerformanceMatcher.jar`-backed OLTW baseline.
- `python -m scripts.run_benchmark --method kalman_oltw --mode small` runs the first-pass Kalman-smoothed online prototype using `PerformanceMatcher.jar` as its measurement source.
- Benchmark selection now excludes pairs with average warp factor above `2.0` by default to match the paper-style constraint. Override with `--exclude-warp-factor-above`.
- `--max-pairs` is only supported with `--mode paper_test`.
- `scripts.online_baselines.register_online_baseline("oltw", runner)` and `register_online_baseline("oltw_global", runner)` override the default Java-backed hooks with external implementations when needed.
- `oltw` and `oltw_global` both default to invoking `PerformanceMatcher.jar`; any future `OnlineAlignment` integration should be registered explicitly through `scripts.online_baselines`.
- `kalman_oltw` is intentionally a prototype: it smooths the JAR's online alignment frontier with a constant-velocity Kalman filter, and it does not yet implement the normalized streaming subsequence-DTW objective described for milestone 2.
- `scripts.visualization.plot_alignment_path(...)` and `scripts.visualization.plot_error_summary(...)` generate diagnostic plots for inspection.

## Server Notes

On older servers, prefer `conda run -n e207-bench ...` for benchmark and test
commands so they work reliably in `tmux` and other non-interactive shells.
`oltw` and `oltw_global` also require Java on `PATH` plus `PerformanceMatcher.jar`
in the repo root or passed explicitly via `--jar-path`.
