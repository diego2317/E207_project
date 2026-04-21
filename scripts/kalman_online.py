"""Kalman-guided normalized online alignment implemented in native Python."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.models import AlignmentResult, FeatureSequence


DEFAULT_METHOD_NAME = "kalman_oltw"

DIAGONAL_WEIGHT = 2.0
HOLD_REFERENCE_WEIGHT = 1.0
QUERY_DOUBLE_WEIGHT = 3.0
REFERENCE_DOUBLE_WEIGHT = 3.0


@dataclass(slots=True)
class KalmanFilterConfig:
    """Frame-domain constant-velocity filter and search parameters."""

    position_variance: float = 64.0
    velocity_variance: float = 0.25
    process_position_variance: float = 4.0
    process_velocity_variance: float = 1.0e-2
    measurement_variance: float = 9.0
    min_velocity: float = 0.25
    max_velocity: float = 4.0
    min_search_half_window: int = 192
    uncertainty_scale: float = 3.0
    position_prior_weight: float = 0.05
    start_anywhere: bool = False


def run_kalman_oltw(
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    metric: str = "cosine",
    kalman_config: KalmanFilterConfig | None = None,
) -> AlignmentResult:
    """Run the native Kalman-guided normalized online DTW prototype."""

    reference_values = _prepare_feature_values(reference_features.values)
    query_values = _prepare_feature_values(query_features.values)
    if reference_values.shape[0] == 0 or query_values.shape[0] == 0:
        raise ValueError("kalman_oltw requires non-empty reference and query feature sequences.")

    config = kalman_config or KalmanFilterConfig()
    measurement_indices, measurement_scores, filtered_states = _run_kalman_guided_online_dtw(
        reference_values=reference_values,
        query_values=query_values,
        config=config,
        metric=metric,
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
            "backend": "python_streaming_normalized_dtw",
            "measurement_source": "normalized_row_argmin",
            "spec_faithful": True,
            "distance_metric": metric,
            "kalman_state_units": "reference_frames",
            "measurement_count": int(measurement_indices.size),
            "mean_measurement_score": float(np.mean(measurement_scores)),
            "max_measurement_score": float(np.max(measurement_scores)),
            "initial_velocity_frames_per_query_frame": float(
                _estimate_initial_velocity(
                    num_reference_frames=reference_features.frame_times.size,
                    num_query_frames=query_features.frame_times.size,
                )
            ),
            "step_pattern": {
                "transitions": ((0, 1), (1, 1), (1, 2), (2, 1)),
                "weights": (
                    HOLD_REFERENCE_WEIGHT,
                    DIAGONAL_WEIGHT,
                    QUERY_DOUBLE_WEIGHT,
                    REFERENCE_DOUBLE_WEIGHT,
                ),
            },
            "kalman_position_variance": float(config.position_variance),
            "kalman_velocity_variance": float(config.velocity_variance),
            "kalman_process_position_variance": float(config.process_position_variance),
            "kalman_process_velocity_variance": float(config.process_velocity_variance),
            "kalman_measurement_variance": float(config.measurement_variance),
            "kalman_min_velocity": float(config.min_velocity),
            "kalman_max_velocity": float(config.max_velocity),
            "min_search_half_window": int(config.min_search_half_window),
            "uncertainty_scale": float(config.uncertainty_scale),
            "position_prior_weight": float(config.position_prior_weight),
            "start_anywhere": bool(config.start_anywhere),
        },
    )


def _run_kalman_guided_online_dtw(
    reference_values: np.ndarray,
    query_values: np.ndarray,
    config: KalmanFilterConfig,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run streaming normalized DP and Kalman updates one query frame at a time."""

    num_reference_frames = reference_values.shape[0]
    num_query_frames = query_values.shape[0]
    metric_name = metric.strip().lower()
    if metric_name != "cosine":
        raise ValueError(f"Unsupported metric {metric!r}; kalman_oltw currently supports 'cosine'.")

    reference_normalized, reference_valid = _normalize_feature_rows(reference_values)
    query_normalized, query_valid = _normalize_feature_rows(query_values)

    initial_velocity = _estimate_initial_velocity(
        num_reference_frames=num_reference_frames,
        num_query_frames=num_query_frames,
    )
    state = np.array([0.0, initial_velocity], dtype=np.float64)
    covariance = np.diag([config.position_variance, config.velocity_variance]).astype(np.float64)

    previous_totals = np.full(num_reference_frames, np.inf, dtype=np.float64)
    previous_lengths = np.zeros(num_reference_frames, dtype=np.float64)
    previous_previous_totals = np.full(num_reference_frames, np.inf, dtype=np.float64)
    previous_previous_lengths = np.zeros(num_reference_frames, dtype=np.float64)

    measurement_indices = np.zeros(num_query_frames, dtype=np.int64)
    measurement_scores = np.zeros(num_query_frames, dtype=np.float64)
    filtered_states = np.zeros((num_query_frames, 2), dtype=np.float64)

    minimum_reference_index = 0

    for query_index in range(num_query_frames):
        if query_index > 0:
            state, covariance = _kalman_predict(state, covariance, config)

        predicted_position = float(state[0])
        predicted_variance = float(covariance[0, 0])
        local_cost_row = _compute_cosine_local_cost_row(
            reference_normalized=reference_normalized,
            reference_valid=reference_valid,
            query_vector=query_normalized[query_index],
            query_is_valid=bool(query_valid[query_index]),
        )

        left, right, radius = _compute_search_window(
            predicted_position=predicted_position,
            predicted_variance=predicted_variance,
            minimum_reference_index=minimum_reference_index,
            num_reference_frames=num_reference_frames,
            config=config,
            query_index=query_index,
        )

        current_totals, current_lengths = _update_streaming_row(
            local_cost_row=local_cost_row,
            previous_totals=previous_totals,
            previous_lengths=previous_lengths,
            previous_previous_totals=previous_previous_totals,
            previous_previous_lengths=previous_previous_lengths,
            left=left,
            right=right,
            query_index=query_index,
            predicted_position=predicted_position,
            radius=radius,
            config=config,
        )

        normalized_scores = _compute_normalized_scores(current_totals, current_lengths)
        measurement_index = _select_measurement_index(
            normalized_scores=normalized_scores,
            minimum_reference_index=minimum_reference_index,
        )
        measurement_value = float(measurement_index)
        measurement_score = float(normalized_scores[measurement_index])

        state, covariance = _kalman_update(
            state=state,
            covariance=covariance,
            measurement=measurement_value,
            num_reference_frames=num_reference_frames,
            previous_position=float(filtered_states[query_index - 1, 0]) if query_index > 0 else 0.0,
            config=config,
        )

        measurement_indices[query_index] = measurement_index
        measurement_scores[query_index] = measurement_score
        filtered_states[query_index] = state
        minimum_reference_index = int(max(minimum_reference_index, round(state[0])))

        previous_previous_totals = previous_totals
        previous_previous_lengths = previous_lengths
        previous_totals = current_totals
        previous_lengths = current_lengths

    return measurement_indices, measurement_scores, filtered_states


def _prepare_feature_values(values: np.ndarray) -> np.ndarray:
    prepared = np.asarray(values, dtype=np.float64)
    if prepared.ndim != 2:
        raise ValueError("kalman_oltw inputs must be 2D feature matrices.")
    return np.ascontiguousarray(prepared)


def _normalize_feature_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    valid = norms[:, 0] > 0.0
    normalized = np.zeros_like(values, dtype=np.float64)
    normalized[valid] = values[valid] / norms[valid]
    return normalized, valid


def _compute_cosine_local_cost_row(
    reference_normalized: np.ndarray,
    reference_valid: np.ndarray,
    query_vector: np.ndarray,
    query_is_valid: bool,
) -> np.ndarray:
    if not query_is_valid:
        return np.ones(reference_normalized.shape[0], dtype=np.float64)

    local_cost = np.ascontiguousarray(1.0 - (reference_normalized @ query_vector), dtype=np.float64)
    local_cost[~reference_valid] = 1.0
    return local_cost


def _compute_search_window(
    predicted_position: float,
    predicted_variance: float,
    minimum_reference_index: int,
    num_reference_frames: int,
    config: KalmanFilterConfig,
    query_index: int,
) -> tuple[int, int, float]:
    radius = float(
        max(
            config.min_search_half_window,
            int(np.ceil(config.uncertainty_scale * np.sqrt(max(predicted_variance, 0.0)))),
        )
    )
    if query_index == 0 and config.start_anywhere:
        left = 0
        right = num_reference_frames - 1
        radius = float(num_reference_frames)
    else:
        left = max(minimum_reference_index, int(np.floor(predicted_position - radius)))
        right = min(num_reference_frames - 1, int(np.ceil(predicted_position + radius)))

    if left > right:
        left = minimum_reference_index
        right = minimum_reference_index
    return left, right, max(radius, 1.0)


def _update_streaming_row(
    local_cost_row: np.ndarray,
    previous_totals: np.ndarray,
    previous_lengths: np.ndarray,
    previous_previous_totals: np.ndarray,
    previous_previous_lengths: np.ndarray,
    left: int,
    right: int,
    query_index: int,
    predicted_position: float,
    radius: float,
    config: KalmanFilterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    num_reference_frames = local_cost_row.size
    current_totals = np.full(num_reference_frames, np.inf, dtype=np.float64)
    current_lengths = np.zeros(num_reference_frames, dtype=np.float64)

    for reference_index in range(left, right + 1):
        local_cost = float(local_cost_row[reference_index])
        local_cost += config.position_prior_weight * (
            ((reference_index - predicted_position) / radius) ** 2
        )

        if query_index == 0:
            current_totals[reference_index] = local_cost
            current_lengths[reference_index] = 1.0
            continue

        best_total = np.inf
        best_length = 0.0
        best_normalized = np.inf

        if np.isfinite(previous_totals[reference_index]):
            best_total, best_length, best_normalized = _consider_candidate(
                best_total=best_total,
                best_length=best_length,
                best_normalized=best_normalized,
                candidate_total=previous_totals[reference_index] + HOLD_REFERENCE_WEIGHT * local_cost,
                candidate_length=previous_lengths[reference_index] + HOLD_REFERENCE_WEIGHT,
            )

        if reference_index >= 1 and np.isfinite(previous_totals[reference_index - 1]):
            best_total, best_length, best_normalized = _consider_candidate(
                best_total=best_total,
                best_length=best_length,
                best_normalized=best_normalized,
                candidate_total=previous_totals[reference_index - 1] + DIAGONAL_WEIGHT * local_cost,
                candidate_length=previous_lengths[reference_index - 1] + DIAGONAL_WEIGHT,
            )

        if reference_index >= 1 and query_index >= 2 and np.isfinite(previous_previous_totals[reference_index - 1]):
            best_total, best_length, best_normalized = _consider_candidate(
                best_total=best_total,
                best_length=best_length,
                best_normalized=best_normalized,
                candidate_total=previous_previous_totals[reference_index - 1]
                + QUERY_DOUBLE_WEIGHT * local_cost,
                candidate_length=previous_previous_lengths[reference_index - 1] + QUERY_DOUBLE_WEIGHT,
            )

        if reference_index >= 2 and np.isfinite(previous_totals[reference_index - 2]):
            best_total, best_length, best_normalized = _consider_candidate(
                best_total=best_total,
                best_length=best_length,
                best_normalized=best_normalized,
                candidate_total=previous_totals[reference_index - 2]
                + REFERENCE_DOUBLE_WEIGHT * local_cost,
                candidate_length=previous_lengths[reference_index - 2] + REFERENCE_DOUBLE_WEIGHT,
            )

        current_totals[reference_index] = best_total
        current_lengths[reference_index] = best_length

    if not np.isfinite(current_totals[left : right + 1]).any():
        if left == 0 and right == num_reference_frames - 1:
            reachable_previous = int(np.count_nonzero(np.isfinite(previous_totals)))
            reachable_previous_previous = int(np.count_nonzero(np.isfinite(previous_previous_totals)))
            raise RuntimeError(
                "Streaming online DTW produced no reachable cells for the current row. "
                f"query_index={query_index} reachable_previous={reachable_previous} "
                f"reachable_previous_previous={reachable_previous_previous}"
            )
        return _update_streaming_row(
            local_cost_row=local_cost_row,
            previous_totals=previous_totals,
            previous_lengths=previous_lengths,
            previous_previous_totals=previous_previous_totals,
            previous_previous_lengths=previous_previous_lengths,
            left=0,
            right=num_reference_frames - 1,
            query_index=query_index,
            predicted_position=predicted_position,
            radius=max(float(num_reference_frames), 1.0),
            config=config,
        )

    return current_totals, current_lengths


def _consider_candidate(
    best_total: float,
    best_length: float,
    best_normalized: float,
    candidate_total: float,
    candidate_length: float,
) -> tuple[float, float, float]:
    candidate_normalized = candidate_total / candidate_length
    if candidate_normalized < best_normalized:
        return candidate_total, candidate_length, candidate_normalized
    if np.isclose(candidate_normalized, best_normalized) and candidate_total < best_total:
        return candidate_total, candidate_length, candidate_normalized
    return best_total, best_length, best_normalized


def _compute_normalized_scores(
    totals: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    scores = np.full(totals.shape, np.inf, dtype=np.float64)
    valid = np.isfinite(totals) & (lengths > 0.0)
    scores[valid] = totals[valid] / lengths[valid]
    return scores


def _select_measurement_index(
    normalized_scores: np.ndarray,
    minimum_reference_index: int,
) -> int:
    finite_indices = np.flatnonzero(np.isfinite(normalized_scores))
    if finite_indices.size == 0:
        raise RuntimeError("Streaming online DTW produced no finite candidates for the current row.")

    valid_indices = finite_indices[finite_indices >= minimum_reference_index]
    if valid_indices.size == 0:
        valid_indices = finite_indices

    best_offset = int(np.argmin(normalized_scores[valid_indices]))
    return int(valid_indices[best_offset])


def _kalman_predict(
    state: np.ndarray,
    covariance: np.ndarray,
    config: KalmanFilterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    transition = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    process_noise = np.diag(
        [config.process_position_variance, config.process_velocity_variance],
    ).astype(np.float64)
    predicted_state = transition @ state
    predicted_covariance = transition @ covariance @ transition.T + process_noise
    return predicted_state, predicted_covariance


def _kalman_update(
    state: np.ndarray,
    covariance: np.ndarray,
    measurement: float,
    num_reference_frames: int,
    previous_position: float,
    config: KalmanFilterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    observation = np.array([[1.0, 0.0]], dtype=np.float64)
    identity = np.eye(2, dtype=np.float64)
    measurement_noise = np.array([[config.measurement_variance]], dtype=np.float64)

    innovation = np.array([measurement], dtype=np.float64) - (observation @ state)
    innovation_covariance = observation @ covariance @ observation.T + measurement_noise
    kalman_gain = covariance @ observation.T @ np.linalg.inv(innovation_covariance)
    updated_state = state + (kalman_gain @ innovation).reshape(-1)
    updated_covariance = (identity - kalman_gain @ observation) @ covariance

    updated_state[1] = float(np.clip(updated_state[1], config.min_velocity, config.max_velocity))
    updated_state[0] = float(np.clip(updated_state[0], previous_position, num_reference_frames - 1))
    updated_covariance = (updated_covariance + updated_covariance.T) * 0.5
    return updated_state, updated_covariance


def _run_constant_velocity_kalman(
    measurements: np.ndarray,
    num_reference_frames: int,
    initial_velocity: float,
    config: KalmanFilterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter a measurement track with the same constant-velocity model."""

    if measurements.ndim != 1:
        raise ValueError("measurements must be a 1D array.")
    if measurements.size == 0:
        raise ValueError("measurements must not be empty.")
    if num_reference_frames <= 0:
        raise ValueError("num_reference_frames must be positive.")

    state = np.array([measurements[0], initial_velocity], dtype=np.float64)
    covariance = np.diag([config.position_variance, config.velocity_variance]).astype(np.float64)
    filtered_states = np.zeros((measurements.size, 2), dtype=np.float64)
    observed_mask = np.isfinite(measurements)
    previous_position = max(0.0, float(measurements[0]))

    for query_index in range(measurements.size):
        if query_index > 0:
            state, covariance = _kalman_predict(state, covariance, config)

        measurement = float(measurements[query_index]) if np.isfinite(measurements[query_index]) else state[0]
        state, covariance = _kalman_update(
            state=state,
            covariance=covariance,
            measurement=measurement,
            num_reference_frames=num_reference_frames,
            previous_position=previous_position,
            config=config,
        )
        filtered_states[query_index] = state
        previous_position = state[0]

    return filtered_states, observed_mask


def _estimate_initial_velocity(
    num_reference_frames: int,
    num_query_frames: int,
) -> float:
    return float(num_reference_frames / max(num_query_frames, 1))
