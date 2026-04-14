"""Plotting helpers for diagnostics and result summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.models import AlignmentResult


def plot_alignment_path(
    alignment_result: AlignmentResult,
    output_path: Path | str | None = None,
):
    """Plot the estimated alignment path in time coordinates."""

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        alignment_result.reference_times,
        alignment_result.query_times,
        linewidth=1.5,
    )
    axis.set_title(
        f"{alignment_result.method_name}: {alignment_result.reference_id} vs {alignment_result.query_id}"
    )
    axis.set_xlabel("Reference time (s)")
    axis.set_ylabel("Query time (s)")
    axis.grid(alpha=0.3)
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    return figure, axis


def plot_error_summary(
    metrics_frame: pd.DataFrame,
    metric_column: str = "mean_abs_error_s",
    output_path: Path | str | None = None,
):
    """Plot a bar chart of per-pair alignment error."""

    if metrics_frame.empty:
        raise ValueError("metrics_frame must not be empty.")

    figure, axis = plt.subplots(figsize=(8, 4))
    plot_frame = metrics_frame.sort_values(metric_column)
    axis.bar(plot_frame["pair_id"], plot_frame[metric_column])
    axis.set_title(f"Alignment summary by pair: {metric_column}")
    axis.set_ylabel(metric_column)
    axis.set_xlabel("Pair")
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    return figure, axis
