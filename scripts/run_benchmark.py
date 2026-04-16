"""Command-line entry point for offline benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts import evaluation
from scripts.config import DEFAULT_SAMPLE_RATE, METRICS_DIR, RAW_DATA_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("single", "small", "full"),
        default="full",
        help="Select one directed benchmark case, a fixed 3-recording preview set, or the full dataset.",
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
        help="Output filename prefix. Defaults to a mode-specific offline DTW name.",
    )
    parser.add_argument(
        "--feature-name",
        default="chroma_stft",
        help="Feature representation passed to the offline benchmark pipeline.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Audio sample rate used when loading recordings.",
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

    experiment_name = args.experiment_name or _default_experiment_name(
        mode=args.mode,
        pair_id=args.pair_id,
    )
    metrics_frame = evaluation.run_offline_benchmark(
        dataset_root=args.dataset_root,
        sample_rate=args.sample_rate,
        feature_name=args.feature_name,
        output_dir=args.output_dir,
        experiment_name=experiment_name,
        selection_mode=args.mode,
        pair_id=args.pair_id,
        subset_size=args.subset_size,
        save_outputs=not args.no_save,
        show_progress=args.show_progress,
    )
    print(
        f"Completed {args.mode} benchmark run with {len(metrics_frame)} benchmark case(s). "
        f"Experiment: {experiment_name}"
    )
    return 0


def _default_experiment_name(mode: str, pair_id: str | None) -> str:
    if mode == "single" and pair_id:
        return f"offline_dtw_single_{pair_id}"
    if mode == "small":
        return "offline_dtw_small"
    if mode == "full":
        return "offline_dtw_full"
    return "offline_dtw_benchmark"


if __name__ == "__main__":
    raise SystemExit(main())
