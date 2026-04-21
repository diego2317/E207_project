"""Benchmark orchestration helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from scripts import data_io, features, metrics, offline_dtw, online_baselines
from scripts.config import DEFAULT_HOP_LENGTH, DEFAULT_SAMPLE_RATE, METRICS_DIR
from scripts.models import AlignmentResult, FeatureSequence, Recording, RecordingPair


AlignmentRunner = Callable[..., AlignmentResult]
BenchmarkSelectionMode = Literal[
    "single",
    "small",
    "full",
    "all_pairs",
    "paper_test",
]
SMALL_BENCHMARK_RECORDING_COUNT = 3
DEFAULT_METHOD_NAME = "offline_dtw"
DEFAULT_DEVELOPMENT_PIECE = "Chopin_Op030No2"
DEFAULT_MAX_WARP_FACTOR = 2.0
_ALIGNMENT_RUNNERS: dict[str, AlignmentRunner] = {
    DEFAULT_METHOD_NAME: offline_dtw.run_offline_dtw,
    "oltw": online_baselines.run_oltw,
    "oltw_global": online_baselines.run_oltw_global,
    "kalman_oltw": online_baselines.run_kalman_oltw,
}


def register_alignment_runner(method_name: str, runner: AlignmentRunner) -> None:
    """Register a benchmark method under a stable method name."""

    normalized_name = method_name.strip().lower()
    if not normalized_name:
        raise ValueError("method_name must not be empty.")
    _ALIGNMENT_RUNNERS[normalized_name] = runner


def unregister_alignment_runner(method_name: str) -> None:
    """Remove a previously registered benchmark method."""

    _ALIGNMENT_RUNNERS.pop(method_name.strip().lower(), None)


def list_alignment_methods() -> tuple[str, ...]:
    """Return the registered benchmark method names in sorted order."""

    return tuple(sorted(_ALIGNMENT_RUNNERS))


def get_alignment_runner(method_name: str = DEFAULT_METHOD_NAME) -> AlignmentRunner:
    """Return the alignment runner registered for a benchmark method."""

    normalized_name = method_name.strip().lower()
    try:
        return _ALIGNMENT_RUNNERS[normalized_name]
    except KeyError as error:
        supported_methods = ", ".join(list_alignment_methods())
        raise ValueError(
            f"Unsupported method_name={method_name!r}. Supported methods: {supported_methods}."
        ) from error


def evaluate_recording_pair(
    pair: RecordingPair,
    method_name: str = DEFAULT_METHOD_NAME,
    runner: AlignmentRunner | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    feature_name: str = "chroma_stft",
    runner_kwargs: dict[str, object] | None = None,
    metric_tolerances: Iterable[float] = metrics.DEFAULT_TOLERANCES,
) -> tuple[AlignmentResult, dict[str, object], pd.DataFrame]:
    """Run one alignment method on one benchmark pair."""

    selected_runner = runner or get_alignment_runner(method_name)
    if _runner_uses_audio_paths(selected_runner):
        reference_features = _build_timeline_features(
            pair.reference,
            sample_rate=sample_rate,
            feature_name=feature_name,
        )
        query_features = _build_timeline_features(
            pair.query,
            sample_rate=sample_rate,
            feature_name=feature_name,
        )
    else:
        (reference_audio, ref_sr), (query_audio, query_sr) = data_io.load_pair_audio(
            pair,
            sample_rate=sample_rate,
        )
        reference_features = features.compute_features(
            reference_audio,
            sr=ref_sr,
            feature_name=feature_name,
        )
        query_features = features.compute_features(
            query_audio,
            sr=query_sr,
            feature_name=feature_name,
        )
        reference_features.metadata["audio_path"] = str(pair.reference.audio_path)
        query_features.metadata["audio_path"] = str(pair.query.audio_path)

    reference_features.metadata["recording_id"] = pair.reference.recording_id
    query_features.metadata["recording_id"] = pair.query.recording_id

    alignment_result = selected_runner(
        reference_features,
        query_features,
        **(runner_kwargs or {}),
    )
    reference_beats = data_io.load_beat_timestamps(pair.reference.beats_path)
    query_beats = data_io.load_beat_timestamps(pair.query.beats_path)
    metric_row = metrics.compute_alignment_metrics(
        alignment_result,
        reference_beats=reference_beats,
        query_beats=query_beats,
        tolerances=metric_tolerances,
    )
    metric_row.update(alignment_result.metadata)
    error_frame = metrics.compute_alignment_error_trace(
        alignment_result,
        reference_beats=reference_beats,
        query_beats=query_beats,
    )
    metric_row.update(
        {
            "piece": pair.piece,
            "pair_id": pair.pair_id,
            "feature_name": feature_name,
        }
    )
    error_frame["piece"] = pair.piece
    error_frame["pair_id"] = pair.pair_id
    error_frame["feature_name"] = feature_name
    return alignment_result, metric_row, error_frame


def benchmark_recording_pairs(
    pairs: list[RecordingPair],
    method_name: str = DEFAULT_METHOD_NAME,
    runner: AlignmentRunner | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    feature_name: str = "chroma_stft",
    output_dir: Path | str = METRICS_DIR,
    experiment_name: str = "alignment_benchmark",
    runner_kwargs: dict[str, object] | None = None,
    metric_tolerances: Iterable[float] = metrics.DEFAULT_TOLERANCES,
    tolerance_grid: Iterable[float] | None = None,
    save_outputs: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Evaluate an alignment method over a list of directed benchmark cases."""

    if not pairs:
        raise ValueError("No benchmark cases were provided for benchmarking.")

    metric_rows: list[dict[str, object]] = []
    error_frames: list[pd.DataFrame] = []
    iterator = tqdm(pairs, desc=experiment_name) if show_progress else pairs
    for pair in iterator:
        _, metric_row, error_frame = evaluate_recording_pair(
            pair,
            method_name=method_name,
            runner=runner,
            sample_rate=sample_rate,
            feature_name=feature_name,
            runner_kwargs=runner_kwargs,
            metric_tolerances=metric_tolerances,
        )
        metric_rows.append(metric_row)
        error_frames.append(error_frame)

    metrics_frame = pd.DataFrame(metric_rows)
    error_rows = pd.concat(error_frames, ignore_index=True)
    if save_outputs:
        save_benchmark_outputs(
            metrics_frame,
            error_rows,
            output_dir=output_dir,
            experiment_name=experiment_name,
            tolerance_grid=tolerance_grid,
        )
    return metrics_frame


def select_recording_pairs(
    pairs: list[RecordingPair],
    selection_mode: BenchmarkSelectionMode = "all_pairs",
    pair_id: str | None = None,
    subset_size: int = 10,
    max_pairs: int | None = None,
    development_piece: str = DEFAULT_DEVELOPMENT_PIECE,
    max_warp_factor: float | None = DEFAULT_MAX_WARP_FACTOR,
) -> list[RecordingPair]:
    """Select benchmark cases for a given benchmark mode."""

    if max_pairs is not None:
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive when provided.")
        if selection_mode != "paper_test":
            raise ValueError("max_pairs is only supported when selection_mode='paper_test'.")

    ordered_pairs = sorted(pairs, key=lambda item: (item.piece, item.pair_id))
    if selection_mode in {"full", "all_pairs"}:
        selected_pairs = ordered_pairs
    elif selection_mode == "paper_test":
        selected_pairs = _select_paper_test_benchmark_pairs(
            ordered_pairs,
            development_piece=development_piece,
        )
    elif selection_mode == "small":
        selected_pairs = _select_small_benchmark_pairs(ordered_pairs)
    elif selection_mode == "single":
        if not pair_id:
            raise ValueError("pair_id is required when selection_mode='single'.")
        for pair in ordered_pairs:
            if pair.pair_id == pair_id:
                selected_pairs = [pair]
                break
        else:
            raise ValueError(f"Pair '{pair_id}' was not found in the discovered benchmark pairs.")
    else:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    if max_warp_factor is not None:
        selected_pairs = [
            pair
            for pair in selected_pairs
            if estimate_average_warping_factor(pair) <= max_warp_factor
        ]
    if max_pairs is not None:
        selected_pairs = selected_pairs[:max_pairs]
    if not selected_pairs:
        raise ValueError(
            f"No benchmark pairs remained after applying selection_mode={selection_mode!r}."
        )
    return selected_pairs


def select_preview_recording_pair(pairs: list[RecordingPair]) -> RecordingPair:
    """Return the fastest preview case from the fixed small benchmark set."""

    small_pairs = select_recording_pairs(pairs, selection_mode="small")
    return min(small_pairs, key=_pair_duration_key)


def run_alignment_benchmark(
    dataset_root: Path | str | None = None,
    method_name: str = DEFAULT_METHOD_NAME,
    runner: AlignmentRunner | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    feature_name: str = "chroma_stft",
    output_dir: Path | str = METRICS_DIR,
    experiment_name: str = "alignment_benchmark",
    selection_mode: BenchmarkSelectionMode = "all_pairs",
    pair_id: str | None = None,
    subset_size: int = 10,
    max_pairs: int | None = None,
    runner_kwargs: dict[str, object] | None = None,
    metric_tolerances: Iterable[float] = metrics.DEFAULT_TOLERANCES,
    tolerance_grid: Iterable[float] | None = None,
    development_piece: str = DEFAULT_DEVELOPMENT_PIECE,
    max_warp_factor: float | None = DEFAULT_MAX_WARP_FACTOR,
    save_outputs: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Discover data, select benchmark cases, and run one alignment benchmark."""

    recordings = data_io.discover_recordings(dataset_root)
    pairs = data_io.build_recording_pairs(recordings)
    selected_pairs = select_recording_pairs(
        pairs,
        selection_mode=selection_mode,
        pair_id=pair_id,
        subset_size=subset_size,
        max_pairs=max_pairs,
        development_piece=development_piece,
        max_warp_factor=max_warp_factor,
    )
    return benchmark_recording_pairs(
        selected_pairs,
        method_name=method_name,
        runner=runner,
        sample_rate=sample_rate,
        feature_name=feature_name,
        output_dir=output_dir,
        experiment_name=experiment_name,
        runner_kwargs=runner_kwargs,
        metric_tolerances=metric_tolerances,
        tolerance_grid=tolerance_grid,
        save_outputs=save_outputs,
        show_progress=show_progress,
    )


def run_offline_benchmark(
    dataset_root: Path | str | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    feature_name: str = "chroma_stft",
    output_dir: Path | str = METRICS_DIR,
    experiment_name: str = "offline_dtw_benchmark",
    selection_mode: BenchmarkSelectionMode = "all_pairs",
    pair_id: str | None = None,
    subset_size: int = 10,
    max_pairs: int | None = None,
    metric_tolerances: Iterable[float] = metrics.DEFAULT_TOLERANCES,
    tolerance_grid: Iterable[float] | None = None,
    development_piece: str = DEFAULT_DEVELOPMENT_PIECE,
    max_warp_factor: float | None = DEFAULT_MAX_WARP_FACTOR,
    save_outputs: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compatibility wrapper for running the offline DTW benchmark."""

    return run_alignment_benchmark(
        dataset_root=dataset_root,
        method_name=DEFAULT_METHOD_NAME,
        sample_rate=sample_rate,
        feature_name=feature_name,
        output_dir=output_dir,
        experiment_name=experiment_name,
        selection_mode=selection_mode,
        pair_id=pair_id,
        subset_size=subset_size,
        max_pairs=max_pairs,
        metric_tolerances=metric_tolerances,
        tolerance_grid=tolerance_grid,
        development_piece=development_piece,
        max_warp_factor=max_warp_factor,
        save_outputs=save_outputs,
        show_progress=show_progress,
    )


def save_benchmark_outputs(
    metrics_frame: pd.DataFrame,
    error_rows: pd.DataFrame,
    output_dir: Path | str = METRICS_DIR,
    experiment_name: str = "offline_dtw_benchmark",
    tolerance_grid: Iterable[float] | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Persist per-pair metrics and aggregated summaries."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    pair_metrics_path = destination / f"{experiment_name}_pairs.csv"
    summary_path = destination / f"{experiment_name}_summary.csv"
    beat_errors_path = destination / f"{experiment_name}_beat_errors.csv"
    tolerance_curve_path = destination / f"{experiment_name}_tolerance_curve.csv"

    metrics_frame.to_csv(pair_metrics_path, index=False)
    metrics.summarize_metrics(metrics_frame).to_csv(summary_path, index=False)
    error_rows.to_csv(beat_errors_path, index=False)
    metrics.compute_tolerance_curve(
        error_rows,
        tolerances=tolerance_grid,
    ).to_csv(tolerance_curve_path, index=False)
    return pair_metrics_path, summary_path, beat_errors_path, tolerance_curve_path


def _select_small_benchmark_pairs(pairs: list[RecordingPair]) -> list[RecordingPair]:
    piece_recordings = _collect_recordings_by_piece(pairs)
    candidate_groups: list[tuple[float, str, list[Recording]]] = []
    for piece, recordings in piece_recordings.items():
        ordered_recordings = sorted(recordings, key=_recording_duration_key)
        if len(ordered_recordings) < SMALL_BENCHMARK_RECORDING_COUNT:
            continue
        chosen = ordered_recordings[:SMALL_BENCHMARK_RECORDING_COUNT]
        duration_sum = sum(_recording_duration_seconds(recording) for recording in chosen)
        candidate_groups.append((duration_sum, piece, chosen))

    if not candidate_groups:
        raise ValueError(
            "Small benchmark mode requires at least three annotated recordings for one piece."
        )

    _, chosen_piece, chosen_recordings = min(
        candidate_groups,
        key=lambda item: (item[0], item[1]),
    )
    chosen_ids = {recording.recording_id for recording in chosen_recordings}
    selected_pairs = [
        pair
        for pair in pairs
        if pair.piece == chosen_piece
        and pair.reference.recording_id in chosen_ids
        and pair.query.recording_id in chosen_ids
    ]
    return sorted(selected_pairs, key=_pair_duration_key)


def _select_paper_test_benchmark_pairs(
    pairs: list[RecordingPair],
    development_piece: str,
) -> list[RecordingPair]:
    return [pair for pair in pairs if pair.piece != development_piece]


def _collect_recordings_by_piece(pairs: list[RecordingPair]) -> dict[str, list[Recording]]:
    grouped: dict[str, dict[str, Recording]] = {}
    for pair in pairs:
        piece_group = grouped.setdefault(pair.piece, {})
        piece_group[pair.reference.recording_id] = pair.reference
        piece_group[pair.query.recording_id] = pair.query
    return {
        piece: [recordings[recording_id] for recording_id in sorted(recordings)]
        for piece, recordings in grouped.items()
    }


def _runner_uses_audio_paths(runner: AlignmentRunner) -> bool:
    return bool(getattr(runner, "uses_audio_paths", False))


def _build_timeline_features(
    recording: Recording,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    feature_name: str = "chroma_stft",
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> FeatureSequence:
    duration_s = get_recording_duration_s(recording)
    frame_count = max(1, int(np.floor((duration_s * sample_rate) / hop_length)) + 1)
    frame_times = np.arange(frame_count, dtype=np.float64) * (hop_length / sample_rate)
    return FeatureSequence(
        values=np.zeros((frame_count, 1), dtype=np.float64),
        frame_times=frame_times,
        sample_rate=sample_rate,
        hop_length=hop_length,
        feature_name=feature_name,
        metadata={
            "audio_path": str(recording.audio_path),
            "recording_id": recording.recording_id,
            "duration_s": duration_s,
        },
    )


def get_recording_duration_s(recording: Recording) -> float:
    """Return one recording duration, caching the metadata lookup."""

    return _recording_duration_seconds(recording)


def estimate_average_warping_factor(pair: RecordingPair) -> float:
    """Estimate the pairwise duration ratio used for paper-style filtering."""

    reference_duration = get_recording_duration_s(pair.reference)
    query_duration = get_recording_duration_s(pair.query)
    shorter = min(reference_duration, query_duration)
    longer = max(reference_duration, query_duration)
    if shorter <= 0:
        raise ValueError(f"Non-positive recording duration encountered for pair {pair.pair_id}.")
    return longer / shorter


def _recording_duration_seconds(recording: Recording) -> float:
    cached_duration = recording.metadata.get("duration_s")
    if cached_duration is not None:
        return float(cached_duration)

    info = sf.info(recording.audio_path)
    duration_s = float(info.frames / info.samplerate)
    recording.metadata["duration_s"] = duration_s
    return duration_s


def _recording_duration_key(recording: Recording) -> tuple[float, str]:
    return (_recording_duration_seconds(recording), recording.recording_id)


def _pair_duration_key(pair: RecordingPair) -> tuple[float, float, str]:
    reference_duration = _recording_duration_seconds(pair.reference)
    query_duration = _recording_duration_seconds(pair.query)
    return (
        reference_duration * query_duration,
        reference_duration + query_duration,
        pair.pair_id,
    )
