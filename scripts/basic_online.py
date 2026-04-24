"""Reset baselines for causal alignment experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts import kalman_online
from scripts.models import AlignmentResult, FeatureSequence


NAIVE_METHOD_NAME = "naive_online_dtw"
BASIC_KALMAN_METHOD_NAME = "basic_kalman_online_dtw"


@dataclass(slots=True)
class BasicKalmanConfig:
    """Minimal constant-velocity Kalman settings for the reset baseline."""

    position_variance: float = 64.0
    velocity_variance: float = 0.25
    process_position_variance: float = 4.0
    process_velocity_variance: float = 1.0e-2
    measurement_variance: float = 9.0
    min_velocity: float = 0.25
    max_velocity: float = 4.0

    def to_legacy_config(self) -> kalman_online.KalmanFilterConfig:
        """Build the existing tracker config without enabling advanced coupling."""

        return kalman_online.KalmanFilterConfig(
            position_variance=self.position_variance,
            velocity_variance=self.velocity_variance,
            process_position_variance=self.process_position_variance,
            process_velocity_variance=self.process_velocity_variance,
            measurement_variance=self.measurement_variance,
            min_velocity=self.min_velocity,
            max_velocity=self.max_velocity,
            min_search_half_window=0,
            uncertainty_scale=0.0,
            position_prior_weight=0.0,
            start_anywhere=False,
        )


@dataclass(frozen=True, slots=True)
class NaiveOnlineTrace:
    """Lightweight diagnostics for the reset baselines."""

    measurement_indices: np.ndarray
    measurement_scores: np.ndarray
    row_best_score_margins: np.ndarray
    row_finite_counts: np.ndarray
    search_widths: np.ndarray
    transition_usage: dict[str, int]


def run_naive_online_dtw(
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    metric: str = "cosine",
) -> AlignmentResult:
    """Run the reset baseline: full-width causal normalized online DTW."""

    reference_values = kalman_online._prepare_feature_values(reference_features.values)
    query_values = kalman_online._prepare_feature_values(query_features.values)
    if reference_values.shape[0] == 0 or query_values.shape[0] == 0:
        raise ValueError("naive_online_dtw requires non-empty reference and query feature sequences.")

    measurement_indices, trace = _run_naive_measurement_track(
        reference_values=reference_values,
        query_values=query_values,
        metric=metric,
    )
    reference_indices = np.maximum.accumulate(
        np.clip(measurement_indices, 0, reference_features.frame_times.size - 1)
    )
    return _build_alignment_result(
        method_name=NAIVE_METHOD_NAME,
        reference_features=reference_features,
        query_features=query_features,
        reference_indices=reference_indices,
        trace=trace,
        metric=metric,
        metadata={
            "tracker_model": "none",
            "reset_baseline": True,
        },
    )


def run_basic_kalman_online_dtw(
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    metric: str = "cosine",
    kalman_config: BasicKalmanConfig | None = None,
) -> AlignmentResult:
    """Run the reset baseline with a plain constant-velocity Kalman filter."""

    reference_values = kalman_online._prepare_feature_values(reference_features.values)
    query_values = kalman_online._prepare_feature_values(query_features.values)
    if reference_values.shape[0] == 0 or query_values.shape[0] == 0:
        raise ValueError(
            "basic_kalman_online_dtw requires non-empty reference and query feature sequences."
        )

    measurement_indices, trace = _run_naive_measurement_track(
        reference_values=reference_values,
        query_values=query_values,
        metric=metric,
    )
    config = kalman_config or BasicKalmanConfig()
    legacy_config = config.to_legacy_config()
    filtered_states, _ = kalman_online._run_constant_velocity_kalman(
        measurements=measurement_indices.astype(np.float64),
        num_reference_frames=reference_features.frame_times.size,
        initial_velocity=kalman_online._estimate_initial_velocity(
            num_reference_frames=reference_features.frame_times.size,
            num_query_frames=query_features.frame_times.size,
        ),
        config=legacy_config,
    )
    reference_indices = np.rint(filtered_states[:, 0]).astype(np.int64)
    reference_indices = np.maximum.accumulate(
        np.clip(reference_indices, 0, reference_features.frame_times.size - 1)
    )
    return _build_alignment_result(
        method_name=BASIC_KALMAN_METHOD_NAME,
        reference_features=reference_features,
        query_features=query_features,
        reference_indices=reference_indices,
        trace=trace,
        metric=metric,
        metadata={
            "tracker_model": "basic_constant_velocity",
            "reset_baseline": True,
            "kalman_position_variance": float(config.position_variance),
            "kalman_velocity_variance": float(config.velocity_variance),
            "kalman_process_position_variance": float(config.process_position_variance),
            "kalman_process_velocity_variance": float(config.process_velocity_variance),
            "kalman_measurement_variance": float(config.measurement_variance),
            "kalman_min_velocity": float(config.min_velocity),
            "kalman_max_velocity": float(config.max_velocity),
        },
    )


def _build_alignment_result(
    method_name: str,
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    reference_indices: np.ndarray,
    trace: NaiveOnlineTrace,
    metric: str,
    metadata: dict[str, object],
) -> AlignmentResult:
    query_indices = np.arange(query_features.frame_times.size, dtype=np.int64)
    path = np.column_stack([reference_indices, query_indices])
    trace_summary = _summarize_trace(trace)
    return AlignmentResult(
        method_name=method_name,
        reference_id=reference_features.metadata.get("recording_id", "reference"),
        query_id=query_features.metadata.get("recording_id", "query"),
        reference_times=reference_features.frame_times[reference_indices],
        query_times=query_features.frame_times[query_indices],
        path=path,
        metadata={
            "backend": "python_full_width_causal_dtw",
            "distance_metric": metric,
            "measurement_source": "normalized_row_argmin",
            "start_at_reference_origin": True,
            "search_policy_name": "full_width_causal",
            "step_pattern_name": "default_normalized_v1",
            "step_pattern": {
                "transitions": tuple(
                    (transition.reference_advance, transition.query_advance)
                    for transition in _reset_step_pattern().transitions
                ),
                "labels": tuple(
                    transition.label for transition in _reset_step_pattern().transitions
                ),
                "weights": tuple(
                    transition.weight for transition in _reset_step_pattern().transitions
                ),
            },
            **trace_summary,
            **metadata,
        },
    )


def _run_naive_measurement_track(
    reference_values: np.ndarray,
    query_values: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, NaiveOnlineTrace]:
    metric_name = metric.strip().lower()
    if metric_name != "cosine":
        raise ValueError(
            f"Unsupported metric {metric!r}; reset baselines currently support 'cosine'."
        )

    num_reference_frames = reference_values.shape[0]
    num_query_frames = query_values.shape[0]
    step_pattern = _reset_step_pattern()
    coupling_config = kalman_online.CouplingConfig(
        name="measurement_only_v1",
        mode="measurement_only",
        position_prior_weight=0.0,
        measurement_source="normalized_row_argmin",
        confidence_source="row_best_score_margin",
    )
    reference_normalized, reference_valid = kalman_online._normalize_feature_rows(reference_values)
    query_normalized, query_valid = kalman_online._normalize_feature_rows(query_values)

    max_query_advance = max(transition.query_advance for transition in step_pattern.transitions)
    row_history: dict[int, tuple[np.ndarray, np.ndarray]] = {
        lag: (
            np.full(num_reference_frames, np.inf, dtype=np.float64),
            np.zeros(num_reference_frames, dtype=np.float64),
        )
        for lag in range(1, max_query_advance + 1)
    }
    measurement_indices = np.zeros(num_query_frames, dtype=np.int64)
    measurement_scores = np.zeros(num_query_frames, dtype=np.float64)
    row_best_score_margins = np.zeros(num_query_frames, dtype=np.float64)
    row_finite_counts = np.zeros(num_query_frames, dtype=np.int64)
    search_widths = np.zeros(num_query_frames, dtype=np.int64)
    transition_usage = {transition.label: 0 for transition in step_pattern.transitions}
    minimum_reference_index = 0

    for query_index in range(num_query_frames):
        local_cost_row = kalman_online._compute_cosine_local_cost_row(
            reference_normalized=reference_normalized,
            reference_valid=reference_valid,
            query_vector=query_normalized[query_index],
            query_is_valid=bool(query_valid[query_index]),
        )

        if query_index == 0:
            current_totals = np.full(num_reference_frames, np.inf, dtype=np.float64)
            current_lengths = np.zeros(num_reference_frames, dtype=np.float64)
            current_totals[0] = float(local_cost_row[0])
            current_lengths[0] = 1.0
            row_diagnostics = kalman_online.RowUpdateDiagnostics(
                finite_candidate_count=1,
                best_score=float(local_cost_row[0]),
                second_best_score=np.inf,
                best_score_margin=0.0,
                used_recovery_window=False,
                recovery_reason="origin_anchor",
                recovery_stage=0,
                transition_usage={label: 0 for label in transition_usage},
            )
            measurement_index = 0
        else:
            search_window = kalman_online.SearchWindow(
                left=minimum_reference_index,
                right=num_reference_frames - 1,
                radius=max(float(num_reference_frames - minimum_reference_index), 1.0),
                recovery_mode="full_width",
                expansion_stage=0,
            )
            current_totals, current_lengths, row_diagnostics = kalman_online._update_streaming_row(
                local_cost_row=local_cost_row,
                row_history=row_history,
                search_window=search_window,
                query_index=query_index,
                predicted_position=float(minimum_reference_index),
                coupling_config=coupling_config,
                step_pattern=step_pattern,
            )
            normalized_scores = kalman_online._compute_normalized_scores(
                current_totals,
                current_lengths,
            )
            measurement_index = kalman_online._select_measurement_index(
                normalized_scores=normalized_scores,
                minimum_reference_index=minimum_reference_index,
            )
            search_widths[query_index] = search_window.right - search_window.left + 1

        measurement_indices[query_index] = measurement_index
        normalized_scores = kalman_online._compute_normalized_scores(current_totals, current_lengths)
        measurement_scores[query_index] = float(normalized_scores[measurement_index])
        row_best_score_margins[query_index] = row_diagnostics.best_score_margin
        row_finite_counts[query_index] = row_diagnostics.finite_candidate_count
        if query_index == 0:
            search_widths[query_index] = num_reference_frames
        minimum_reference_index = max(minimum_reference_index, measurement_index)

        for label, count in row_diagnostics.transition_usage.items():
            transition_usage[label] = transition_usage.get(label, 0) + count
        for lag in range(max_query_advance, 1, -1):
            row_history[lag] = row_history[lag - 1]
        row_history[1] = (current_totals, current_lengths)

    return (
        measurement_indices,
        NaiveOnlineTrace(
            measurement_indices=measurement_indices,
            measurement_scores=measurement_scores,
            row_best_score_margins=row_best_score_margins,
            row_finite_counts=row_finite_counts,
            search_widths=search_widths,
            transition_usage=transition_usage,
        ),
    )


def _summarize_trace(trace: NaiveOnlineTrace) -> dict[str, object]:
    finite_margins = trace.row_best_score_margins[np.isfinite(trace.row_best_score_margins)]
    mean_margin = float(np.mean(finite_margins)) if finite_margins.size else 0.0
    return {
        "measurement_count": int(trace.measurement_indices.size),
        "mean_measurement_score": float(np.mean(trace.measurement_scores)),
        "max_measurement_score": float(np.max(trace.measurement_scores)),
        "mean_search_width": float(np.mean(trace.search_widths)),
        "max_search_width": int(np.max(trace.search_widths)),
        "mean_row_finite_count": float(np.mean(trace.row_finite_counts)),
        "mean_row_best_score_margin": mean_margin,
        "transition_usage": dict(trace.transition_usage),
    }


def _reset_step_pattern() -> kalman_online.StepPatternConfig:
    return kalman_online.StepPatternConfig(
        name="default_normalized_v1",
        transitions=kalman_online._default_step_transitions(),
    )
