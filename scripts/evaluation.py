"""Benchmark orchestration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm.auto import tqdm

from scripts import data_io, features, metrics, offline_dtw
from scripts.config import DEFAULT_SAMPLE_RATE, METRICS_DIR
from scripts.models import AlignmentResult, RecordingPair


AlignmentRunner = Callable[..., AlignmentResult]


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
    """Evaluate an alignment method over a list of benchmark pairs."""

    if not pairs:
        raise ValueError("No recording pairs were provided for benchmarking.")

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


def run_offline_benchmark(
    dataset_root: Path | str | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    feature_name: str = "chroma_stft",
    output_dir: Path | str = METRICS_DIR,
    experiment_name: str = "offline_dtw_benchmark",
    save_outputs: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Discover data, build same-piece pairs, and run the offline DTW benchmark."""

    recordings = data_io.discover_recordings(dataset_root)
    pairs = data_io.build_recording_pairs(recordings)
    return benchmark_recording_pairs(
        pairs,
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
