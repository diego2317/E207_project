"""Evaluation metrics for alignment quality."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from scripts.models import AlignmentResult


DEFAULT_TOLERANCES = (0.05, 0.1, 0.2)


def estimate_query_times(
    alignment_result: AlignmentResult,
    reference_times: np.ndarray,
) -> np.ndarray:
    """Interpolate query timestamps for arbitrary reference timestamps."""

    ref_path, query_path = _collapse_duplicates(
        alignment_result.reference_times,
        alignment_result.query_times,
    )
    return np.interp(
        np.asarray(reference_times, dtype=np.float64),
        ref_path,
        query_path,
        left=query_path[0],
        right=query_path[-1],
    )


def estimate_reference_times(
    alignment_result: AlignmentResult,
    query_times: np.ndarray,
) -> np.ndarray:
    """Interpolate reference timestamps for arbitrary query timestamps."""

    query_path, reference_path = _collapse_duplicates(
        alignment_result.query_times,
        alignment_result.reference_times,
    )
    return np.interp(
        np.asarray(query_times, dtype=np.float64),
        query_path,
        reference_path,
        left=reference_path[0],
        right=reference_path[-1],
    )


def compute_alignment_metrics(
    alignment_result: AlignmentResult,
    reference_beats: np.ndarray,
    query_beats: np.ndarray,
    tolerances: Iterable[float] = DEFAULT_TOLERANCES,
) -> dict[str, float | int | str]:
    """Compare an estimated query-to-reference alignment against beat correspondence."""

    reference_beats = np.asarray(reference_beats, dtype=np.float64)
    query_beats = np.asarray(query_beats, dtype=np.float64)
    num_beats = min(reference_beats.size, query_beats.size)
    if num_beats == 0:
        raise ValueError("Both recordings must contain at least one beat timestamp.")

    query_eval = query_beats[:num_beats]
    reference_eval = reference_beats[:num_beats]
    estimated_reference = estimate_reference_times(alignment_result, query_eval)
    errors = estimated_reference - reference_eval
    abs_errors = np.abs(errors)

    metrics: dict[str, float | int | str] = {
        "method_name": alignment_result.method_name,
        "reference_id": alignment_result.reference_id,
        "query_id": alignment_result.query_id,
        "num_beats_used": int(num_beats),
        "mean_error_s": float(np.mean(errors)),
        "mean_abs_error_s": float(np.mean(abs_errors)),
        "median_abs_error_s": float(np.median(abs_errors)),
        "rmse_s": float(np.sqrt(np.mean(errors**2))),
        "max_abs_error_s": float(np.max(abs_errors)),
        "p95_abs_error_s": float(np.percentile(abs_errors, 95)),
    }

    for tolerance in tolerances:
        milliseconds = int(round(tolerance * 1000))
        metrics[f"within_{milliseconds}ms"] = float(np.mean(abs_errors <= tolerance))

    return metrics


def summarize_metrics(metric_rows: pd.DataFrame | list[dict[str, object]]) -> pd.DataFrame:
    """Aggregate per-pair metrics into a compact summary table."""

    frame = metric_rows if isinstance(metric_rows, pd.DataFrame) else pd.DataFrame(metric_rows)
    if frame.empty:
        return pd.DataFrame()

    summary = (
        frame.groupby("method_name", dropna=False)
        .agg(
            num_pairs=("method_name", "size"),
            mean_mae_s=("mean_abs_error_s", "mean"),
            median_mae_s=("mean_abs_error_s", "median"),
            mean_rmse_s=("rmse_s", "mean"),
            mean_p95_abs_error_s=("p95_abs_error_s", "mean"),
            mean_within_100ms=("within_100ms", "mean"),
            mean_within_200ms=("within_200ms", "mean"),
        )
        .reset_index()
    )
    return summary


def _collapse_duplicates(
    reference_times: np.ndarray,
    query_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference_times = np.asarray(reference_times, dtype=np.float64)
    query_times = np.asarray(query_times, dtype=np.float64)
    unique_reference, inverse = np.unique(reference_times, return_inverse=True)

    query_sums = np.zeros_like(unique_reference, dtype=np.float64)
    counts = np.zeros_like(unique_reference, dtype=np.int64)
    np.add.at(query_sums, inverse, query_times)
    np.add.at(counts, inverse, 1)

    averaged_query = query_sums / counts
    return unique_reference, averaged_query
