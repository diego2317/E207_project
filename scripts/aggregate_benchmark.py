"""Aggregate per-beat benchmark outputs into an error-rate-vs-tolerance figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts import metrics, visualization
from scripts.config import FIGURES_DIR, METRICS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beat-errors",
        nargs="+",
        type=Path,
        required=True,
        help="One or more *_beat_errors.csv files produced by scripts.run_benchmark.",
    )
    parser.add_argument(
        "--metrics-output-dir",
        type=Path,
        default=METRICS_DIR,
        help="Directory where the aggregated tolerance-curve CSV will be written.",
    )
    parser.add_argument(
        "--figures-output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory where the comparison figure will be written.",
    )
    parser.add_argument(
        "--experiment-name",
        default="alignment_tolerance_comparison",
        help="Output filename prefix for the aggregated CSV and figure.",
    )
    parser.add_argument(
        "--tolerance-max-seconds",
        type=float,
        default=metrics.DEFAULT_MAX_TOLERANCE_S,
        help="Maximum tolerance value included in the aggregated sweep.",
    )
    parser.add_argument(
        "--tolerance-step-seconds",
        type=float,
        default=metrics.DEFAULT_TOLERANCE_STEP_S,
        help="Tolerance step size used in the aggregated sweep.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    beat_error_frames = [pd.read_csv(path) for path in args.beat_errors]
    combined_errors = pd.concat(beat_error_frames, ignore_index=True)
    tolerance_grid = metrics.build_tolerance_grid(
        max_tolerance_s=args.tolerance_max_seconds,
        step_s=args.tolerance_step_seconds,
    )
    tolerance_curve = metrics.compute_tolerance_curve(
        combined_errors,
        tolerances=tolerance_grid,
    )

    args.metrics_output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.metrics_output_dir / f"{args.experiment_name}_tolerance_curve.csv"
    figure_path = args.figures_output_dir / f"{args.experiment_name}_error_rate_vs_tolerance.png"

    tolerance_curve.to_csv(curve_path, index=False)
    figure, _ = visualization.plot_tolerance_curve(
        tolerance_curve,
        output_path=figure_path,
    )
    figure.clf()

    print(
        f"Aggregated {len(combined_errors)} beat-level predictions across "
        f"{combined_errors['method_name'].nunique()} method(s). "
        f"Curve: {curve_path} Figure: {figure_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
