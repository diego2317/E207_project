# E207 Project

Milestone 1 implementation for the Real-Time Alignment & Tracking project.

## Layout

- `scripts/`: reusable modules for benchmark discovery, feature extraction, offline DTW, evaluation, and plotting
- `notebooks/`: exploratory notebook for running the offline benchmark pipeline
- `data/`: local benchmark assets and derived datasets
- `outputs/`: figures, metrics, and logs produced during experiments
- `tests/`: smoke and pipeline tests for the Milestone 1 workflow

## Data Layout

The benchmark loader supports either:

1. A manifest file at `data/raw/mazurka_manifest.csv` or `data/raw/manifest.csv`
2. An inferred directory layout with one audio file and one beat file per recording directory

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

1. Create a virtual environment.
2. Install `requirements.txt`.
3. Place benchmark data under `data/raw/`.
4. Run `pytest` to verify the pipeline.
5. Use `notebooks/01_alignment_sandbox.ipynb` for exploratory testing.

## Offline Benchmark Entry Points

- `scripts.evaluation.evaluate_recording_pair(pair)` runs one offline-DTW alignment and computes beat-based metrics.
- `scripts.evaluation.run_offline_benchmark(...)` discovers recordings, builds same-piece pairs, and writes metrics summaries to `outputs/metrics/`.
- `scripts.visualization.plot_alignment_path(...)` and `scripts.visualization.plot_error_summary(...)` generate diagnostic plots for inspection.
