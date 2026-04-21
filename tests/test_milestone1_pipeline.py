"""Tests for the milestone 1 offline-DTW benchmark pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from scripts import (
    aggregate_benchmark,
    data_io,
    evaluation,
    features,
    metrics,
    offline_dtw,
    oltw,
    online_baselines,
    run_benchmark,
    visualization,
)
from scripts.models import AlignmentResult, FeatureSequence


def test_discover_recordings_from_manifest(tmp_path: Path) -> None:
    dataset_root = _build_synthetic_dataset(tmp_path)

    recordings = data_io.discover_recordings(dataset_root)
    pairs = data_io.build_recording_pairs(recordings)

    assert len(recordings) == 2
    assert recordings[0].piece == "mazurka_op24_no2"
    assert len(pairs) == 2
    assert {pair.pair_id for pair in pairs} == {
        "performance_a__performance_b",
        "performance_b__performance_a",
    }


def test_discover_recordings_from_split_layout(tmp_path: Path) -> None:
    dataset_root = _build_split_layout_dataset(tmp_path, recording_count=2)

    recordings = data_io.discover_recordings(dataset_root)
    pairs = data_io.build_recording_pairs(recordings)

    assert len(recordings) == 2
    assert recordings[0].piece == "Chopin_Op024No2"
    assert recordings[0].beats_path is not None
    assert recordings[0].beats_path.suffix == ".beat"
    assert len(pairs) == 2
    assert {pair.pair_id for pair in pairs} == {
        "Chopin_Op024No2_performance_00__Chopin_Op024No2_performance_01",
        "Chopin_Op024No2_performance_01__Chopin_Op024No2_performance_00",
    }


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
    assert result.metadata["total_cost"] == pytest.approx(0.0)


def test_offline_dtw_accumulation_matches_reference_kernel() -> None:
    local_cost = np.array(
        [
            [0.5, 1.0, 0.5],
            [0.5, 0.25, 1.5],
            [1.0, 0.25, 0.25],
        ],
        dtype=np.float64,
    )

    expected_cost, expected_backpointers = offline_dtw._accumulate_cost_reference(local_cost)
    actual_cost, actual_backpointers = offline_dtw._accumulate_cost(local_cost)

    assert actual_cost == pytest.approx(expected_cost)
    np.testing.assert_array_equal(actual_backpointers, expected_backpointers)


def test_offline_dtw_prefers_diagonal_on_zero_cost_ties() -> None:
    reference = FeatureSequence(
        values=np.zeros((3, 1), dtype=np.float64),
        frame_times=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        sample_rate=1,
        hop_length=1,
        feature_name="toy",
    )
    query = FeatureSequence(
        values=np.zeros((3, 1), dtype=np.float64),
        frame_times=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        sample_rate=1,
        hop_length=1,
        feature_name="toy",
    )

    result = offline_dtw.run_offline_dtw(reference, query, metric="euclidean")

    np.testing.assert_array_equal(
        result.path,
        np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64),
    )


def test_offline_dtw_accepts_noncontiguous_float32_features() -> None:
    reference_values = np.arange(6, dtype=np.float32).reshape(3, 2)[:, :1]
    query_values = np.arange(6, dtype=np.float32).reshape(3, 2)[:, :1]
    assert not reference_values.flags["C_CONTIGUOUS"]
    assert not query_values.flags["C_CONTIGUOUS"]

    reference = FeatureSequence(
        values=reference_values,
        frame_times=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        sample_rate=1,
        hop_length=1,
        feature_name="toy",
    )
    query = FeatureSequence(
        values=query_values,
        frame_times=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        sample_rate=1,
        hop_length=1,
        feature_name="toy",
    )

    result = offline_dtw.run_offline_dtw(reference, query, metric="euclidean")

    np.testing.assert_array_equal(
        result.path,
        np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64),
    )
    assert result.metadata["total_cost"] == pytest.approx(0.0)


def test_oltw_parses_performance_matcher_output(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = FeatureSequence(
        values=np.zeros((3, 1), dtype=np.float64),
        frame_times=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        sample_rate=1,
        hop_length=1,
        feature_name="toy",
        metadata={"recording_id": "reference", "audio_path": "reference.wav"},
    )
    query = FeatureSequence(
        values=np.zeros((3, 1), dtype=np.float64),
        frame_times=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        sample_rate=1,
        hop_length=1,
        feature_name="toy",
        metadata={"recording_id": "query", "audio_path": "query.wav"},
    )

    def fake_run(*args: object, **kwargs: object) -> object:
        class Completed:
            returncode = 0
            stdout = "\n".join(["ALIGNMENT: 1, 1", "ALIGNMENT: 2, 2"])
            stderr = ""

        return Completed()

    monkeypatch.setattr(oltw.subprocess, "run", fake_run)
    result = oltw.run_oltw(reference, query)

    assert result.method_name == "oltw"
    assert tuple(result.path[0]) == (0, 0)
    assert tuple(result.path[-1]) == (2, 2)
    assert np.all(np.diff(result.path[:, 0]) >= 0)
    assert np.all(np.diff(result.path[:, 1]) >= 0)
    assert result.metadata["backend"] == "PerformanceMatcher.jar"
    assert result.metadata["use_global_constraint"] is False


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


def test_compute_alignment_metrics_tracks_reference_time_from_query_time() -> None:
    alignment = AlignmentResult(
        method_name="offline_dtw",
        reference_id="ref",
        query_id="query",
        reference_times=np.array([0.0, 1.0, 1.0], dtype=np.float64),
        query_times=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        path=np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64),
    )

    result = metrics.compute_alignment_metrics(
        alignment,
        reference_beats=np.array([0.5, 1.0], dtype=np.float64),
        query_beats=np.array([0.0, 1.0], dtype=np.float64),
    )

    assert result["mean_abs_error_s"] == 0.0
    assert result["rmse_s"] == 0.0


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

    assert len(metrics_frame) == 2
    assert set(metrics_frame["pair_id"]) == {
        "performance_a__performance_b",
        "performance_b__performance_a",
    }
    assert (metrics_frame["mean_abs_error_s"] < 0.05).all()
    assert (output_dir / "synthetic_benchmark_pairs.csv").exists()
    assert (output_dir / "synthetic_benchmark_summary.csv").exists()
    assert (output_dir / "synthetic_benchmark_beat_errors.csv").exists()
    assert (output_dir / "synthetic_benchmark_tolerance_curve.csv").exists()


def test_run_alignment_benchmark_supports_performance_matcher_oltw(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_root = _build_synthetic_dataset(tmp_path)

    def fake_run(*args: object, **kwargs: object) -> object:
        class Completed:
            returncode = 0
            stdout = "\n".join(
                f"ALIGNMENT: {index}, {index}"
                for index in range(1, 90)
            )
            stderr = ""

        return Completed()

    monkeypatch.setattr(oltw.subprocess, "run", fake_run)

    metrics_frame = evaluation.run_alignment_benchmark(
        dataset_root=dataset_root,
        method_name="oltw",
        save_outputs=False,
        show_progress=False,
    )

    assert len(metrics_frame) == 2
    assert set(metrics_frame["method_name"]) == {"oltw"}
    assert (metrics_frame["mean_abs_error_s"] < 0.05).all()


def test_run_alignment_benchmark_supports_registered_method(tmp_path: Path) -> None:
    dataset_root = _build_synthetic_dataset(tmp_path)

    def fake_online_runner(
        reference_features: FeatureSequence,
        query_features: FeatureSequence,
    ) -> AlignmentResult:
        length = min(len(reference_features.frame_times), len(query_features.frame_times))
        indices = np.arange(length, dtype=np.int64)
        return AlignmentResult(
            method_name="toy_online",
            reference_id=reference_features.metadata["recording_id"],
            query_id=query_features.metadata["recording_id"],
            reference_times=reference_features.frame_times[:length],
            query_times=query_features.frame_times[:length],
            path=np.column_stack([indices, indices]),
            metadata={"backend": "synthetic"},
        )

    evaluation.register_alignment_runner("toy_online", fake_online_runner)
    try:
        metrics_frame = evaluation.run_alignment_benchmark(
            dataset_root=dataset_root,
            method_name="toy_online",
            save_outputs=False,
            show_progress=False,
        )
    finally:
        evaluation.unregister_alignment_runner("toy_online")

    assert len(metrics_frame) == 2
    assert set(metrics_frame["method_name"]) == {"toy_online"}
    assert set(metrics_frame["pair_id"]) == {
        "performance_a__performance_b",
        "performance_b__performance_a",
    }


def test_get_alignment_runner_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unsupported method_name"):
        evaluation.get_alignment_runner("not_a_method")


def test_oltw_global_passes_global_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = FeatureSequence(
        values=np.zeros((2, 1), dtype=np.float64),
        frame_times=np.array([0.0, 0.5], dtype=np.float64),
        sample_rate=1,
        hop_length=1,
        feature_name="toy",
        metadata={"audio_path": "reference.wav"},
    )
    query = FeatureSequence(
        values=np.zeros((2, 1), dtype=np.float64),
        frame_times=np.array([0.0, 0.5], dtype=np.float64),
        sample_rate=1,
        hop_length=1,
        feature_name="toy",
        metadata={"audio_path": "query.wav"},
    )

    commands: list[list[str]] = []

    def fake_run(command: list[str], *args: object, **kwargs: object) -> object:
        commands.append(command)

        class Completed:
            returncode = 0
            stdout = "ALIGNMENT: 1, 1"
            stderr = ""

        return Completed()

    monkeypatch.setattr(oltw.subprocess, "run", fake_run)
    result = online_baselines.run_oltw_global(reference, query)

    assert result.method_name == "oltw_global"
    assert commands
    assert "-G" in commands[0]


def test_select_recording_pairs_supports_single_small_all_pairs_and_paper_test(
    tmp_path: Path,
) -> None:
    dataset_root = _build_split_layout_dataset(
        tmp_path,
        piece_specs={
            "Chopin_Op024No2": [0.8, 0.9, 1.0, 1.1],
            "Chopin_Op030No2": [0.3, 0.4, 0.5, 1.2],
        },
    )
    recordings = data_io.discover_recordings(dataset_root)
    pairs = data_io.build_recording_pairs(recordings)

    full_pairs = evaluation.select_recording_pairs(pairs, selection_mode="all_pairs")
    small_pairs = evaluation.select_recording_pairs(pairs, selection_mode="small")
    paper_pairs = evaluation.select_recording_pairs(
        pairs,
        selection_mode="paper_test",
        development_piece="Chopin_Op030No2",
    )
    single_pair = evaluation.select_recording_pairs(
        pairs,
        selection_mode="single",
        pair_id=full_pairs[0].pair_id,
    )

    assert len(full_pairs) == 24
    assert len(paper_pairs) == 12
    assert {pair.piece for pair in paper_pairs} == {"Chopin_Op024No2"}
    assert len(small_pairs) == 6
    assert {pair.pair_id for pair in small_pairs} == {
        "Chopin_Op030No2_performance_00__Chopin_Op030No2_performance_01",
        "Chopin_Op030No2_performance_00__Chopin_Op030No2_performance_02",
        "Chopin_Op030No2_performance_01__Chopin_Op030No2_performance_00",
        "Chopin_Op030No2_performance_01__Chopin_Op030No2_performance_02",
        "Chopin_Op030No2_performance_02__Chopin_Op030No2_performance_00",
        "Chopin_Op030No2_performance_02__Chopin_Op030No2_performance_01",
    }
    assert len(single_pair) == 1
    assert single_pair[0].pair_id == full_pairs[0].pair_id

    with pytest.raises(ValueError, match="pair_id is required"):
        evaluation.select_recording_pairs(pairs, selection_mode="single")


def test_select_recording_pairs_can_filter_large_warp_factors(tmp_path: Path) -> None:
    dataset_root = _build_split_layout_dataset(
        tmp_path,
        piece_specs={"Chopin_Op030No2": [0.5, 0.6, 1.4]},
    )
    pairs = data_io.build_recording_pairs(data_io.discover_recordings(dataset_root))

    filtered_pairs = evaluation.select_recording_pairs(
        pairs,
        selection_mode="all_pairs",
        max_warp_factor=2.0,
    )

    assert len(filtered_pairs) == 2
    assert {pair.pair_id for pair in filtered_pairs} == {
        "Chopin_Op030No2_performance_00__Chopin_Op030No2_performance_01",
        "Chopin_Op030No2_performance_01__Chopin_Op030No2_performance_00",
    }


def test_run_offline_benchmark_supports_selection_modes(tmp_path: Path) -> None:
    dataset_root = _build_split_layout_dataset(
        tmp_path,
        piece_specs={
            "Chopin_Op024No2": [0.7, 0.8, 0.9],
            "Chopin_Op030No2": [0.3, 0.4, 0.5],
        },
    )
    recordings = data_io.discover_recordings(dataset_root)
    pairs = evaluation.select_recording_pairs(
        data_io.build_recording_pairs(recordings),
        selection_mode="all_pairs",
    )
    target_pair_id = pairs[0].pair_id

    single_frame = evaluation.run_offline_benchmark(
        dataset_root=dataset_root,
        selection_mode="single",
        pair_id=target_pair_id,
        save_outputs=False,
        show_progress=False,
    )
    small_frame = evaluation.run_offline_benchmark(
        dataset_root=dataset_root,
        selection_mode="small",
        save_outputs=False,
        show_progress=False,
    )
    full_frame = evaluation.run_offline_benchmark(
        dataset_root=dataset_root,
        selection_mode="all_pairs",
        save_outputs=False,
        show_progress=False,
    )
    paper_frame = evaluation.run_offline_benchmark(
        dataset_root=dataset_root,
        selection_mode="paper_test",
        development_piece="Chopin_Op030No2",
        save_outputs=False,
        show_progress=False,
    )

    assert len(single_frame) == 1
    assert single_frame.loc[0, "pair_id"] == target_pair_id
    assert len(small_frame) == 6
    assert len(full_frame) == 12
    assert len(paper_frame) == 6


def test_select_preview_recording_pair_returns_fastest_small_case(tmp_path: Path) -> None:
    dataset_root = _build_split_layout_dataset(
        tmp_path,
        piece_specs={"Chopin_Op030No2": [0.3, 0.4, 0.5, 1.2]},
    )
    pairs = data_io.build_recording_pairs(data_io.discover_recordings(dataset_root))

    preview_pair = evaluation.select_preview_recording_pair(pairs)

    assert preview_pair.piece == "Chopin_Op030No2"
    assert preview_pair.pair_id == "Chopin_Op030No2_performance_00__Chopin_Op030No2_performance_01"


def test_run_benchmark_cli_passes_selection_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_run_alignment_benchmark(**kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame([{"pair_id": "demo_pair"}])

    monkeypatch.setattr(
        run_benchmark.evaluation,
        "run_alignment_benchmark",
        fake_run_alignment_benchmark,
    )

    exit_code = run_benchmark.main(
        [
            "--method",
            "offline_dtw",
            "--mode",
            "small",
            "--dataset-root",
            str(tmp_path / "raw"),
            "--output-dir",
            str(tmp_path / "metrics"),
            "--experiment-name",
            "cli_benchmark",
            "--no-save",
        ]
    )

    assert exit_code == 0
    assert captured["method_name"] == "offline_dtw"
    assert captured["selection_mode"] == "small"
    assert captured["subset_size"] == 10
    assert captured["save_outputs"] is False
    assert captured["development_piece"] == evaluation.DEFAULT_DEVELOPMENT_PIECE
    assert len(captured["tolerance_grid"]) > 10
    assert (
        "Completed offline_dtw small benchmark run with 1 benchmark case(s)."
        in capsys.readouterr().out
    )


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
    tolerance_frame = pd.DataFrame(
        [
            {"method_name": "offline_dtw", "tolerance_s": 0.0, "error_rate": 1.0},
            {"method_name": "offline_dtw", "tolerance_s": 0.1, "error_rate": 0.2},
        ]
    )

    alignment_path = tmp_path / "alignment.png"
    summary_path = tmp_path / "summary.png"
    tolerance_path = tmp_path / "tolerance.png"
    figure_one, _ = visualization.plot_alignment_path(alignment, output_path=alignment_path)
    figure_two, _ = visualization.plot_error_summary(
        metrics_frame,
        output_path=summary_path,
    )
    figure_three, _ = visualization.plot_tolerance_curve(
        tolerance_frame,
        output_path=tolerance_path,
    )

    assert alignment_path.exists()
    assert summary_path.exists()
    assert tolerance_path.exists()
    figure_one.clf()
    figure_two.clf()
    figure_three.clf()


def test_aggregate_benchmark_cli_writes_curve_and_figure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    beat_errors_path = tmp_path / "oltw_beat_errors.csv"
    pd.DataFrame(
        [
            {"method_name": "oltw", "abs_error_s": 0.05},
            {"method_name": "oltw", "abs_error_s": 0.10},
            {"method_name": "offline_dtw", "abs_error_s": 0.02},
        ]
    ).to_csv(beat_errors_path, index=False)

    metrics_dir = tmp_path / "metrics"
    figures_dir = tmp_path / "figures"
    exit_code = aggregate_benchmark.main(
        [
            "--beat-errors",
            str(beat_errors_path),
            "--metrics-output-dir",
            str(metrics_dir),
            "--figures-output-dir",
            str(figures_dir),
            "--experiment-name",
            "combined",
        ]
    )

    assert exit_code == 0
    assert (metrics_dir / "combined_tolerance_curve.csv").exists()
    assert (figures_dir / "combined_error_rate_vs_tolerance.png").exists()
    assert "Aggregated 3 beat-level predictions" in capsys.readouterr().out


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


def _build_split_layout_dataset(
    tmp_path: Path,
    piece: str = "Chopin_Op024No2",
    recording_count: int = 2,
    piece_specs: dict[str, list[float]] | None = None,
) -> Path:
    dataset_root = tmp_path / "raw"
    specs = piece_specs or {piece: [0.8] * recording_count}

    sample_rate = 22050
    beat_times = np.array([0.1, 0.3, 0.5, 0.7], dtype=np.float64)

    for current_piece, durations_s in specs.items():
        audio_dir = dataset_root / "wav_22050_mono" / current_piece
        beats_dir = dataset_root / "annotations_beat" / current_piece
        audio_dir.mkdir(parents=True)
        beats_dir.mkdir(parents=True)

        for index, duration_s in enumerate(durations_s):
            recording_id = f"{current_piece}_performance_{index:02d}"
            audio_path = audio_dir / f"{recording_id}.wav"
            beats_path = beats_dir / f"{recording_id}.beat"
            signal = _sine_wave(
                frequency=440.0 + 10.0 * index,
                duration_s=duration_s,
                sample_rate=sample_rate,
            )

            sf.write(audio_path, signal * (1.0 - 0.02 * index), sample_rate)
            _write_beat_annotation(beats_path, beat_times)

    return dataset_root


def _write_beat_annotation(path: Path, beat_times: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("% synthetic beat annotation\n")
        handle.write("% start_time[sec]\tend_time[sec]\tlabel\n")
        for label, start_time in enumerate(beat_times, start=1):
            handle.write(f"{start_time:.6f}\t{start_time + 0.05:.6f}\t{label}\n")


def _sine_wave(frequency: float, duration_s: float, sample_rate: int) -> np.ndarray:
    time = np.linspace(0.0, duration_s, int(duration_s * sample_rate), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * frequency * time)
