"""PerformanceMatcher-backed OLTW baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

import numpy as np

from scripts.config import PROJECT_ROOT
from scripts.models import AlignmentResult, FeatureSequence


ALIGNMENT_PATTERN = re.compile(r"ALIGNMENT:\s*(\d+)\s*,\s*(\d+)")
DEFAULT_JAR_PATH = PROJECT_ROOT / "PerformanceMatcher.jar"
DEFAULT_MATCHER_FLAGS = ("-b", "-q", "-D", "--use-chroma-map")


@dataclass(slots=True)
class PerformanceMatcherConfig:
    """Configuration for invoking the Java PerformanceMatcher baseline."""

    method_name: str
    jar_path: Path = DEFAULT_JAR_PATH
    java_command: str = "java"
    matcher_flags: tuple[str, ...] = DEFAULT_MATCHER_FLAGS
    use_global_constraint: bool = False


def build_performance_matcher_command(
    query_audio_path: Path | str,
    reference_audio_path: Path | str,
    config: PerformanceMatcherConfig,
) -> list[str]:
    """Build the Java command for one PerformanceMatcher run."""

    command = [
        config.java_command,
        "-jar",
        str(config.jar_path),
        *config.matcher_flags,
    ]
    if config.use_global_constraint:
        command.append("-G")
    command.extend([str(query_audio_path), str(reference_audio_path)])
    return command


def parse_performance_matcher_alignment(stdout: str) -> np.ndarray:
    """Extract query/reference frame pairs from PerformanceMatcher stdout."""

    matches = [
        (int(match.group(1)), int(match.group(2)))
        for match in ALIGNMENT_PATTERN.finditer(stdout)
    ]
    if not matches:
        raise ValueError("PerformanceMatcher stdout did not contain any ALIGNMENT lines.")
    return np.asarray(matches, dtype=np.int64)


def run_oltw(
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    jar_path: Path | str = DEFAULT_JAR_PATH,
    java_command: str = "java",
    matcher_flags: tuple[str, ...] = DEFAULT_MATCHER_FLAGS,
) -> AlignmentResult:
    """Run the PerformanceMatcher OLTW baseline."""

    return _run_performance_matcher(
        reference_features,
        query_features,
        config=PerformanceMatcherConfig(
            method_name="oltw",
            jar_path=Path(jar_path),
            java_command=java_command,
            matcher_flags=tuple(matcher_flags),
            use_global_constraint=False,
        ),
    )


def run_oltw_global(
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    jar_path: Path | str = DEFAULT_JAR_PATH,
    java_command: str = "java",
    matcher_flags: tuple[str, ...] = DEFAULT_MATCHER_FLAGS,
) -> AlignmentResult:
    """Run the PerformanceMatcher OLTW-global baseline."""

    return _run_performance_matcher(
        reference_features,
        query_features,
        config=PerformanceMatcherConfig(
            method_name="oltw_global",
            jar_path=Path(jar_path),
            java_command=java_command,
            matcher_flags=tuple(matcher_flags),
            use_global_constraint=True,
        ),
    )


def _run_performance_matcher(
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    config: PerformanceMatcherConfig,
) -> AlignmentResult:
    reference_audio_path = _resolve_audio_path(reference_features, expected_role="reference")
    query_audio_path = _resolve_audio_path(query_features, expected_role="query")
    command = build_performance_matcher_command(
        query_audio_path=query_audio_path,
        reference_audio_path=reference_audio_path,
        config=config,
    )

    if not config.jar_path.exists():
        raise FileNotFoundError(
            f"PerformanceMatcher.jar was not found at {config.jar_path}. "
            "Pass --jar-path to scripts.run_benchmark or place the JAR at the project root."
        )

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Could not execute {config.java_command!r}. Make sure Java is installed and on PATH."
        ) from error

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout_tail = completed.stdout.strip()[-500:]
        raise RuntimeError(
            f"PerformanceMatcher exited with status {completed.returncode} for "
            f"{config.method_name}. stderr={stderr!r} stdout_tail={stdout_tail!r}"
        )

    raw_alignment = parse_performance_matcher_alignment(completed.stdout)
    path = _normalize_alignment_path(
        raw_alignment=raw_alignment,
        num_reference_frames=reference_features.frame_times.size,
        num_query_frames=query_features.frame_times.size,
    )
    reference_indices = path[:, 0]
    query_indices = path[:, 1]

    return AlignmentResult(
        method_name=config.method_name,
        reference_id=reference_features.metadata.get("recording_id", "reference"),
        query_id=query_features.metadata.get("recording_id", "query"),
        reference_times=reference_features.frame_times[reference_indices],
        query_times=query_features.frame_times[query_indices],
        path=path,
        metadata={
            "backend": "PerformanceMatcher.jar",
            "jar_path": str(config.jar_path),
            "java_command": config.java_command,
            "matcher_flags": list(config.matcher_flags),
            "use_global_constraint": config.use_global_constraint,
            "command": command,
            "raw_alignment_points": int(raw_alignment.shape[0]),
            "stdout_line_count": int(len(completed.stdout.splitlines())),
        },
    )


def _normalize_alignment_path(
    raw_alignment: np.ndarray,
    num_reference_frames: int,
    num_query_frames: int,
) -> np.ndarray:
    if raw_alignment.ndim != 2 or raw_alignment.shape[1] != 2:
        raise ValueError("raw_alignment must have shape (N, 2).")
    if num_reference_frames <= 0 or num_query_frames <= 0:
        raise ValueError("Alignment timelines must contain at least one frame.")

    query_indices = np.clip(raw_alignment[:, 0], 0, num_query_frames - 1)
    reference_indices = np.clip(raw_alignment[:, 1], 0, num_reference_frames - 1)
    path = np.column_stack([reference_indices, query_indices])

    if path[0, 0] != 0 or path[0, 1] != 0:
        path = np.vstack([np.array([[0, 0]], dtype=np.int64), path])

    keep_mask = np.ones(path.shape[0], dtype=bool)
    if path.shape[0] > 1:
        keep_mask[1:] = np.any(np.diff(path, axis=0) != 0, axis=1)
    normalized = path[keep_mask]

    if np.any(np.diff(normalized[:, 0]) < 0) or np.any(np.diff(normalized[:, 1]) < 0):
        raise ValueError("PerformanceMatcher returned a non-monotonic alignment path.")
    return normalized


def _resolve_audio_path(
    feature_sequence: FeatureSequence,
    expected_role: str,
) -> Path:
    audio_path = feature_sequence.metadata.get("audio_path")
    if audio_path is None:
        raise ValueError(
            f"{expected_role} audio_path is missing from FeatureSequence.metadata. "
            "Use scripts.evaluation to build benchmark inputs or provide the metadata explicitly."
        )
    return Path(audio_path)


run_oltw.uses_audio_paths = True
run_oltw_global.uses_audio_paths = True
