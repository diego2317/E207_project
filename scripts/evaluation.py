"""Benchmark orchestration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from typing import Literal

import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from scripts import data_io, features, metrics, offline_dtw
from scripts.config import DEFAULT_SAMPLE_RATE, METRICS_DIR
from scripts.models import AlignmentResult, Recording, RecordingPair


AlignmentRunner = Callable[..., AlignmentResult]
BenchmarkSelectionMode = Literal["single", "small", "full"]
SMALL_BENCHMARK_RECORDING_COUNT = 3


def evaluate_recording_pair(
    pair: RecordingPair,
    runner: AlignmentRunner = offline_dtw.run_offline_dtw,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    feature_name: str = "chroma_stft",
) -> tuple[AlignmentResult, dict[str, object]]:
    """Run one alignment method on one benchmark pair."""

    (reference_audio, ref_sr), (query_audio, query_sr) = data_io.load_pair_audio(
        pair,
        sample_rate=sample_rate,
    )
    reference_features = features.compute_features(
        reference_audio,
        sr=ref_sr,
        feature_name=feature_name,
    )
    reference_features.metadata["recording_id"] = pair.reference.recording_id

    query_features = features.compute_features(
        query_audio,
        sr=query_sr,
        feature_name=feature_name,
    )
    query_features.metadata["recording_id"] = pair.query.recording_id

    alignment_result = runner(reference_features, query_features)
    reference_beats = data_io.load_beat_timestamps(pair.reference.beats_path)
    query_beats = data_io.load_beat_timestamps(pair.query.beats_path)
    metric_row = metrics.compute_alignment_metrics(
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
    return alignment_result, metric_row


def benchmark_recording_pairs(
    pairs: list[RecordingPair],
    runner: AlignmentRunner = offline_dtw.run_offline_dtw,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    feature_name: str = "chroma_stft",
    output_dir: Path | str = METRICS_DIR,
    experiment_name: str = "offline_dtw_benchmark",
    save_outputs: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Evaluate an alignment method over a list of directed benchmark cases."""

    if not pairs:
        raise ValueError("No benchmark cases were provided for benchmarking.")

    metric_rows: list[dict[str, object]] = []
    iterator = tqdm(pairs, desc=experiment_name) if show_progress else pairs
    for pair in iterator:
        _, metric_row = evaluate_recording_pair(
            pair,
            runner=runner,
            sample_rate=sample_rate,
            feature_name=feature_name,
        )
        metric_rows.append(metric_row)

    metrics_frame = pd.DataFrame(metric_rows)
    if save_outputs:
        save_benchmark_outputs(metrics_frame, output_dir=output_dir, experiment_name=experiment_name)
    return metrics_frame


def select_recording_pairs(
    pairs: list[RecordingPair],
    selection_mode: BenchmarkSelectionMode = "full",
    pair_id: str | None = None,
    subset_size: int = 10,
) -> list[RecordingPair]:
    """Select benchmark cases for a given benchmark mode."""

    ordered_pairs = sorted(pairs, key=lambda item: (item.piece, item.pair_id))
    if selection_mode == "full":
        return ordered_pairs

    if selection_mode == "small":
        return _select_small_benchmark_pairs(ordered_pairs)

    if selection_mode == "single":
        if not pair_id:
            raise ValueError("pair_id is required when selection_mode='single'.")
        for pair in ordered_pairs:
            if pair.pair_id == pair_id:
                return [pair]
        raise ValueError(f"Pair '{pair_id}' was not found in the discovered benchmark pairs.")

    raise ValueError(f"Unsupported selection_mode: {selection_mode}")


def select_preview_recording_pair(pairs: list[RecordingPair]) -> RecordingPair:
    """Return the fastest preview case from the fixed small benchmark set."""

    small_pairs = select_recording_pairs(pairs, selection_mode="small")
    return min(small_pairs, key=_pair_duration_key)


def run_offline_benchmark(
    dataset_root: Path | str | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    feature_name: str = "chroma_stft",
    output_dir: Path | str = METRICS_DIR,
    experiment_name: str = "offline_dtw_benchmark",
    selection_mode: BenchmarkSelectionMode = "full",
    pair_id: str | None = None,
    subset_size: int = 10,
    save_outputs: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Discover data, select benchmark cases, and run the offline DTW benchmark."""

    recordings = data_io.discover_recordings(dataset_root)
    pairs = data_io.build_recording_pairs(recordings)
    selected_pairs = select_recording_pairs(
        pairs,
        selection_mode=selection_mode,
        pair_id=pair_id,
        subset_size=subset_size,
    )
    return benchmark_recording_pairs(
        selected_pairs,
        sample_rate=sample_rate,
        feature_name=feature_name,
        output_dir=output_dir,
        experiment_name=experiment_name,
        save_outputs=save_outputs,
        show_progress=show_progress,
    )


def save_benchmark_outputs(
    metrics_frame: pd.DataFrame,
    output_dir: Path | str = METRICS_DIR,
    experiment_name: str = "offline_dtw_benchmark",
) -> tuple[Path, Path]:
    """Persist per-pair metrics and aggregated summaries."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    pair_metrics_path = destination / f"{experiment_name}_pairs.csv"
    summary_path = destination / f"{experiment_name}_summary.csv"

    metrics_frame.to_csv(pair_metrics_path, index=False)
    metrics.summarize_metrics(metrics_frame).to_csv(summary_path, index=False)
    return pair_metrics_path, summary_path


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
