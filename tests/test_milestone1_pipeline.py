"""Tests for the milestone 1 offline-DTW benchmark pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from scripts import data_io, evaluation, features, metrics, offline_dtw, visualization
from scripts.models import AlignmentResult, FeatureSequence


def test_discover_recordings_from_manifest(tmp_path: Path) -> None:
    dataset_root = _build_synthetic_dataset(tmp_path)

    recordings = data_io.discover_recordings(dataset_root)
    pairs = data_io.build_recording_pairs(recordings)

    assert len(recordings) == 2
    assert recordings[0].piece == "mazurka_op24_no2"
    assert len(pairs) == 1
    assert pairs[0].pair_id == "performance_a__performance_b"


def test_compute_features_is_deterministic() -> None:
    sr = 22050
    signal = _sine_wave(frequency=440.0, duration_s=1.0, sample_rate=sr)

    first = features.compute_features(signal, sr=sr)
    second = features.compute_features(signal, sr=sr)

    assert first.values.shape[1] == 12
    np.testing.assert_allclose(first.values, second.values)
    np.testing.assert_allclose(first.frame_times, second.frame_times)


def test_offline_dtw_returns_monotonic_path() -> None:
    reference = FeatureSequence(
        values=np.array([[0.0], [1.0], [2.0]], dtype=np.float64),
        frame_times=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        sample_rate=1,
        hop_length=1,
        feature_name="toy",
    )
    query = FeatureSequence(
        values=np.array([[0.0], [1.0], [2.0]], dtype=np.float64),
        frame_times=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        sample_rate=1,
        hop_length=1,
        feature_name="toy",
    )

    result = offline_dtw.run_offline_dtw(reference, query, metric="euclidean")

    assert tuple(result.path[0]) == (0, 0)
    assert tuple(result.path[-1]) == (2, 2)
    assert np.all(np.diff(result.path[:, 0]) >= 0)
    assert np.all(np.diff(result.path[:, 1]) >= 0)


def test_compute_alignment_metrics_gives_zero_error_for_identity() -> None:
    alignment = AlignmentResult(
        method_name="offline_dtw",
        reference_id="ref",
        query_id="query",
        reference_times=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        query_times=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        path=np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64),
    )

    result = metrics.compute_alignment_metrics(
        alignment,
        reference_beats=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        query_beats=np.array([0.0, 0.5, 1.0], dtype=np.float64),
    )

    assert result["mean_abs_error_s"] == 0.0
    assert result["rmse_s"] == 0.0
    assert result["within_100ms"] == 1.0


def test_end_to_end_offline_benchmark_writes_outputs(tmp_path: Path) -> None:
    dataset_root = _build_synthetic_dataset(tmp_path)
    output_dir = tmp_path / "metrics"

    metrics_frame = evaluation.run_offline_benchmark(
        dataset_root=dataset_root,
        output_dir=output_dir,
        experiment_name="synthetic_benchmark",
        save_outputs=True,
        show_progress=False,
    )

    assert not metrics_frame.empty
    assert metrics_frame.loc[0, "pair_id"] == "performance_a__performance_b"
    assert metrics_frame.loc[0, "mean_abs_error_s"] < 0.05
    assert (output_dir / "synthetic_benchmark_pairs.csv").exists()
    assert (output_dir / "synthetic_benchmark_summary.csv").exists()


def test_visualization_helpers_render(tmp_path: Path) -> None:
    alignment = AlignmentResult(
        method_name="offline_dtw",
        reference_id="performance_a",
        query_id="performance_b",
        reference_times=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        query_times=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        path=np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64),
    )
    metrics_frame = pd.DataFrame(
        [{"pair_id": "performance_a__performance_b", "mean_abs_error_s": 0.01}]
    )

    alignment_path = tmp_path / "alignment.png"
    summary_path = tmp_path / "summary.png"
    figure_one, _ = visualization.plot_alignment_path(alignment, output_path=alignment_path)
    figure_two, _ = visualization.plot_error_summary(
        metrics_frame,
        output_path=summary_path,
    )

    assert alignment_path.exists()
    assert summary_path.exists()
    figure_one.clf()
    figure_two.clf()


def _build_synthetic_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "raw"
    piece_dir = dataset_root / "mazurka_op24_no2"
    recording_a_dir = piece_dir / "performance_a"
    recording_b_dir = piece_dir / "performance_b"
    recording_a_dir.mkdir(parents=True)
    recording_b_dir.mkdir(parents=True)

    sample_rate = 22050
    signal = _sine_wave(frequency=440.0, duration_s=2.0, sample_rate=sample_rate)
    beat_times = np.array([0.25, 0.75, 1.25, 1.75], dtype=np.float64)

    audio_a = recording_a_dir / "audio.wav"
    audio_b = recording_b_dir / "audio.wav"
    beats_a = recording_a_dir / "beats.csv"
    beats_b = recording_b_dir / "beats.csv"

    sf.write(audio_a, signal, sample_rate)
    sf.write(audio_b, signal, sample_rate)
    np.savetxt(beats_a, beat_times, delimiter=",")
    np.savetxt(beats_b, beat_times, delimiter=",")

    manifest_path = dataset_root / "mazurka_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["piece", "recording_id", "audio_path", "beats_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "piece": "mazurka_op24_no2",
                "recording_id": "performance_a",
                "audio_path": audio_a.relative_to(dataset_root),
                "beats_path": beats_a.relative_to(dataset_root),
            }
        )
        writer.writerow(
            {
                "piece": "mazurka_op24_no2",
                "recording_id": "performance_b",
                "audio_path": audio_b.relative_to(dataset_root),
                "beats_path": beats_b.relative_to(dataset_root),
            }
        )

    return dataset_root


def _sine_wave(frequency: float, duration_s: float, sample_rate: int) -> np.ndarray:
    time = np.linspace(0.0, duration_s, int(duration_s * sample_rate), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * frequency * time)
