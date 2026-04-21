"""Command-line entry point for alignment benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts import evaluation, metrics, oltw as oltw_backend
from scripts.config import DEFAULT_SAMPLE_RATE, METRICS_DIR, RAW_DATA_DIR


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=evaluation.list_alignment_methods(),
        default=evaluation.DEFAULT_METHOD_NAME,
        help="Alignment method to benchmark.",
    )
    parser.add_argument(
        "--mode",
        choices=("single", "small", "full", "all_pairs", "paper_test"),
        default="all_pairs",
        help=(
            "Select one directed benchmark case, a fixed 3-recording preview set, "
            "all directed pairs, or the paper-style held-out test subset."
        ),
    )
    parser.add_argument(
        "--pair-id",
        default=None,
        help="Pair identifier to run when --mode=single, e.g. ref_recording__query_recording.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=10,
        help="Legacy option retained for compatibility; small mode now runs a fixed 3-recording preview set.",
    )
    parser.add_argument(
        "--max-pairs",
        type=_positive_int,
        default=None,
        help="When --mode=paper_test, optionally limit the number of held-out directed cases.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=RAW_DATA_DIR,
        help="Dataset root containing either a manifest or the Mazurka split layout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=METRICS_DIR,
        help="Directory where benchmark metrics CSV files will be written.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Output filename prefix. Defaults to a method- and mode-specific name.",
    )
    parser.add_argument(
        "--feature-name",
        default="chroma_stft",
        help="Feature representation passed to the benchmark pipeline.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Audio sample rate used when loading recordings.",
    )
    parser.add_argument(
        "--development-piece",
        default=evaluation.DEFAULT_DEVELOPMENT_PIECE,
        help="Piece reserved for development when --mode=paper_test.",
    )
    parser.add_argument(
        "--exclude-warp-factor-above",
        type=float,
        default=evaluation.DEFAULT_MAX_WARP_FACTOR,
        help=(
            "Exclude pairs whose duration ratio exceeds this threshold. "
            f"Defaults to {evaluation.DEFAULT_MAX_WARP_FACTOR}."
        ),
    )
    parser.add_argument(
        "--jar-path",
        type=Path,
        default=oltw_backend.DEFAULT_JAR_PATH,
        help="Path to PerformanceMatcher.jar for the oltw baselines.",
    )
    parser.add_argument(
        "--tolerance-max-seconds",
        type=float,
        default=metrics.DEFAULT_MAX_TOLERANCE_S,
        help="Maximum tolerance value included in the saved error-rate sweep CSV.",
    )
    parser.add_argument(
        "--tolerance-step-seconds",
        type=float,
        default=metrics.DEFAULT_TOLERANCE_STEP_S,
        help="Tolerance step size used for the saved error-rate sweep CSV.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display a progress bar while processing benchmark pairs.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing metrics CSV outputs to disk.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    runner_kwargs: dict[str, object] | None = None
    if args.method in {"oltw", "oltw_global"}:
        runner_kwargs = {"jar_path": args.jar_path}

    tolerance_grid = metrics.build_tolerance_grid(
        max_tolerance_s=args.tolerance_max_seconds,
        step_s=args.tolerance_step_seconds,
    )
    experiment_name = args.experiment_name or _default_experiment_name(
        method_name=args.method,
        mode=args.mode,
        pair_id=args.pair_id,
    )
    metrics_frame = evaluation.run_alignment_benchmark(
        dataset_root=args.dataset_root,
        method_name=args.method,
        sample_rate=args.sample_rate,
        feature_name=args.feature_name,
        output_dir=args.output_dir,
        experiment_name=experiment_name,
        selection_mode=args.mode,
        pair_id=args.pair_id,
        subset_size=args.subset_size,
        max_pairs=args.max_pairs,
        runner_kwargs=runner_kwargs,
        tolerance_grid=tolerance_grid,
        development_piece=args.development_piece,
        max_warp_factor=args.exclude_warp_factor_above,
        save_outputs=not args.no_save,
        show_progress=args.show_progress,
    )
    print(
        f"Completed {args.method} {args.mode} benchmark run with {len(metrics_frame)} "
        f"benchmark case(s). Experiment: {experiment_name}"
    )
    return 0


def _default_experiment_name(method_name: str, mode: str, pair_id: str | None) -> str:
    if mode == "single" and pair_id:
        return f"{method_name}_single_{pair_id}"
    if mode == "small":
        return f"{method_name}_small"
    if mode in {"full", "all_pairs"}:
        return f"{method_name}_all_pairs"
    if mode == "paper_test":
        return f"{method_name}_paper_test"
    return f"{method_name}_benchmark"


if __name__ == "__main__":
    raise SystemExit(main())
