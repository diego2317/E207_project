"""Kalman-smoothed online alignment prototype built on PerformanceMatcher.jar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np

from scripts import oltw as oltw_backend
from scripts.models import AlignmentResult, FeatureSequence


DEFAULT_METHOD_NAME = "kalman_oltw"


@dataclass(slots=True)
class KalmanFilterConfig:
    """Frame-domain constant-velocity filter parameters for the prototype."""

    position_variance: float = 16.0
    velocity_variance: float = 0.25
    process_position_variance: float = 1.0
    process_velocity_variance: float = 1.0e-3
    measurement_variance: float = 16.0
    min_velocity: float = 0.25
    max_velocity: float = 4.0


def run_kalman_oltw(
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    jar_path: Path | str = oltw_backend.DEFAULT_JAR_PATH,
    java_command: str = "java",
    matcher_flags: tuple[str, ...] = oltw_backend.DEFAULT_MATCHER_FLAGS,
    kalman_config: KalmanFilterConfig | None = None,
) -> AlignmentResult:
    """Run the phase-1 Kalman-smoothed online alignment prototype."""

    if reference_features.frame_times.size == 0 or query_features.frame_times.size == 0:
        raise ValueError("kalman_oltw requires non-empty reference and query timelines.")

    config = kalman_config or KalmanFilterConfig()
    reference_audio_path = _resolve_audio_path(reference_features, expected_role="reference")
    query_audio_path = _resolve_audio_path(query_features, expected_role="query")
    command = oltw_backend.build_performance_matcher_command(
        query_audio_path=query_audio_path,
        reference_audio_path=reference_audio_path,
        config=oltw_backend.PerformanceMatcherConfig(
            method_name=DEFAULT_METHOD_NAME,
            jar_path=Path(jar_path),
            java_command=java_command,
            matcher_flags=tuple(matcher_flags),
            use_global_constraint=False,
        ),
    )

    completed = _run_performance_matcher_command(
        command=command,
        jar_path=Path(jar_path),
        java_command=java_command,
    )
    raw_alignment = oltw_backend.parse_performance_matcher_alignment(completed.stdout)
    measurements = _build_measurement_track(
        raw_alignment=raw_alignment,
        num_reference_frames=reference_features.frame_times.size,
        num_query_frames=query_features.frame_times.size,
    )
    initial_velocity = _estimate_initial_velocity(
        num_reference_frames=reference_features.frame_times.size,
        num_query_frames=query_features.frame_times.size,
    )
    filtered_states, observed_mask = _run_constant_velocity_kalman(
        measurements=measurements,
        num_reference_frames=reference_features.frame_times.size,
        initial_velocity=initial_velocity,
        config=config,
    )

    reference_indices = np.rint(filtered_states[:, 0]).astype(np.int64)
    reference_indices = np.clip(reference_indices, 0, reference_features.frame_times.size - 1)
    reference_indices = np.maximum.accumulate(reference_indices)
    query_indices = np.arange(query_features.frame_times.size, dtype=np.int64)
    path = np.column_stack([reference_indices, query_indices])

    return AlignmentResult(
        method_name=DEFAULT_METHOD_NAME,
        reference_id=reference_features.metadata.get("recording_id", "reference"),
        query_id=query_features.metadata.get("recording_id", "query"),
        reference_times=reference_features.frame_times[reference_indices],
        query_times=query_features.frame_times[query_indices],
        path=path,
        metadata={
            "backend": "PerformanceMatcher.jar",
            "measurement_source": "alignment_frontier",
            "spec_faithful": False,
            "kalman_state_units": "reference_frames",
            "jar_path": str(jar_path),
            "java_command": java_command,
            "matcher_flags": list(matcher_flags),
            "command": command,
            "raw_alignment_points": int(raw_alignment.shape[0]),
            "measurement_count": int(np.count_nonzero(observed_mask)),
            "stdout_line_count": int(len(completed.stdout.splitlines())),
            "initial_velocity_frames_per_query_frame": float(initial_velocity),
            "kalman_position_variance": float(config.position_variance),
            "kalman_velocity_variance": float(config.velocity_variance),
            "kalman_process_position_variance": float(config.process_position_variance),
            "kalman_process_velocity_variance": float(config.process_velocity_variance),
            "kalman_measurement_variance": float(config.measurement_variance),
            "kalman_min_velocity": float(config.min_velocity),
            "kalman_max_velocity": float(config.max_velocity),
        },
    )


def _build_measurement_track(
    raw_alignment: np.ndarray,
    num_reference_frames: int,
    num_query_frames: int,
) -> np.ndarray:
    """Convert PerformanceMatcher ALIGNMENT output into one measurement per query frame."""

    if raw_alignment.ndim != 2 or raw_alignment.shape[1] != 2:
        raise ValueError("raw_alignment must have shape (N, 2).")
    if num_reference_frames <= 0 or num_query_frames <= 0:
        raise ValueError("Alignment timelines must contain at least one frame.")

    measurements = np.full(num_query_frames, np.nan, dtype=np.float64)
    measurements[0] = 0.0
    last_reference_index = 0.0

    for query_index, reference_index in raw_alignment:
        clipped_query = int(np.clip(query_index, 0, num_query_frames - 1))
        clipped_reference = float(np.clip(reference_index, 0, num_reference_frames - 1))
        last_reference_index = max(last_reference_index, clipped_reference)
        existing = measurements[clipped_query]
        if np.isnan(existing):
            measurements[clipped_query] = last_reference_index
        else:
            measurements[clipped_query] = max(float(existing), last_reference_index)

    return measurements


def _run_constant_velocity_kalman(
    measurements: np.ndarray,
    num_reference_frames: int,
    initial_velocity: float,
    config: KalmanFilterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter one measurement stream with a constant-velocity Kalman model."""

    if measurements.ndim != 1:
        raise ValueError("measurements must be a 1D array.")
    if measurements.size == 0:
        raise ValueError("measurements must not be empty.")
    if num_reference_frames <= 0:
        raise ValueError("num_reference_frames must be positive.")

    transition = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    observation = np.array([[1.0, 0.0]], dtype=np.float64)
    identity = np.eye(2, dtype=np.float64)
    process_noise = np.diag(
        [config.process_position_variance, config.process_velocity_variance],
    ).astype(np.float64)
    measurement_noise = np.array([[config.measurement_variance]], dtype=np.float64)

    state = np.array([0.0, initial_velocity], dtype=np.float64)
    covariance = np.diag([config.position_variance, config.velocity_variance]).astype(np.float64)

    filtered_states = np.zeros((measurements.size, 2), dtype=np.float64)
    observed_mask = np.isfinite(measurements)
    previous_position = 0.0

    for query_index in range(measurements.size):
        if query_index > 0:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_noise

        measurement = measurements[query_index]
        if np.isfinite(measurement):
            innovation = np.array([measurement], dtype=np.float64) - (observation @ state)
            innovation_covariance = observation @ covariance @ observation.T + measurement_noise
            kalman_gain = covariance @ observation.T @ np.linalg.inv(innovation_covariance)
            state = state + (kalman_gain @ innovation).reshape(-1)
            covariance = (identity - kalman_gain @ observation) @ covariance

        state[1] = float(np.clip(state[1], config.min_velocity, config.max_velocity))
        state[0] = float(np.clip(state[0], previous_position, num_reference_frames - 1))
        covariance = (covariance + covariance.T) * 0.5

        filtered_states[query_index] = state
        previous_position = state[0]

    return filtered_states, observed_mask


def _estimate_initial_velocity(
    num_reference_frames: int,
    num_query_frames: int,
) -> float:
    return float(num_reference_frames / max(num_query_frames, 1))


def _run_performance_matcher_command(
    command: list[str],
    jar_path: Path,
    java_command: str,
) -> subprocess.CompletedProcess[str]:
    if not jar_path.exists():
        raise FileNotFoundError(
            f"PerformanceMatcher.jar was not found at {jar_path}. "
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
            f"Could not execute {java_command!r}. Make sure Java is installed and on PATH."
        ) from error

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout_tail = completed.stdout.strip()[-500:]
        raise RuntimeError(
            f"PerformanceMatcher exited with status {completed.returncode} for "
            f"{DEFAULT_METHOD_NAME}. stderr={stderr!r} stdout_tail={stdout_tail!r}"
        )
    return completed


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


run_kalman_oltw.uses_audio_paths = True
