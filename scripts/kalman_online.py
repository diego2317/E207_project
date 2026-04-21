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


@dataclass(frozen=True, slots=True)
class StepTransition:
    """One causal transition in the streaming normalized-DTW step pattern."""

    label: str
    reference_advance: int
    query_advance: int
    weight: float


@dataclass(frozen=True, slots=True)
class StepPatternConfig:
    """Transition-set scaffold for future online-DTW experiments."""

    name: str
    transitions: tuple[StepTransition, ...]
    tie_break_policy: str = "lowest_total_cost"


@dataclass(frozen=True, slots=True)
class SearchPolicyConfig:
    """Search-window policy scaffold for future gating experiments."""

    name: str
    min_search_half_window: int
    uncertainty_scale: float
    start_anywhere: bool
    recovery_strategy: str = "full_width"


@dataclass(frozen=True, slots=True)
class CouplingConfig:
    """How tracker state influences the DP and how measurements are extracted."""

    name: str
    mode: str
    position_prior_weight: float
    measurement_source: str
    confidence_source: str


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Tracker-model scaffold for current and future linear Kalman variants."""

    name: str
    state_model: str
    position_variance: float
    velocity_variance: float
    process_position_variance: float
    process_velocity_variance: float
    measurement_variance: float
    min_velocity: float
    max_velocity: float
    adaptive_process_noise: bool = False
    adaptive_measurement_noise: bool = False
    reset_on_persistent_innovation: bool = False


@dataclass(frozen=True, slots=True)
class KalmanOLTWArchitecture:
    """Named bundle of the current runner's interchangeable algorithm pieces."""

    preset_name: str
    search_policy: SearchPolicyConfig
    step_pattern: StepPatternConfig
    coupling: CouplingConfig
    tracker: TrackerConfig


@dataclass(frozen=True, slots=True)
class SearchWindow:
    """Concrete search bounds chosen for one query row."""

    left: int
    right: int
    radius: float
    recovery_mode: str


@dataclass(slots=True)
class RowUpdateDiagnostics:
    """Lightweight diagnostics for one streaming-DTW row update."""

    finite_candidate_count: int
    best_score: float
    second_best_score: float
    best_score_margin: float
    used_recovery_window: bool
    transition_usage: dict[str, int]


@dataclass(frozen=True, slots=True)
class KalmanUpdateDiagnostics:
    """Innovation statistics from one Kalman measurement update."""

    innovation: float
    innovation_variance: float


@dataclass(frozen=True, slots=True)
class KalmanOnlineTrace:
    """Per-frame trace used to summarize and compare future variants."""

    predicted_positions: np.ndarray
    predicted_velocities: np.ndarray
    predicted_position_variances: np.ndarray
    filtered_positions: np.ndarray
    filtered_velocities: np.ndarray
    measurement_indices: np.ndarray
    measurement_scores: np.ndarray
    innovations: np.ndarray
    innovation_variances: np.ndarray
    search_left_bounds: np.ndarray
    search_right_bounds: np.ndarray
    search_radii: np.ndarray
    used_recovery_window: np.ndarray
    row_finite_counts: np.ndarray
    row_best_score_margins: np.ndarray
    transition_usage: dict[str, int]


@dataclass(frozen=True, slots=True)
class KalmanExperimentPreset:
    """Documented preset scaffold for future controlled comparisons."""

    name: str
    description: str
    tracker_model: str
    search_policy: str
    step_pattern: str
    coupling: str
    implemented: bool
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class KalmanFilterConfig:
    """Compatibility config for the current constant-velocity baseline."""

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

    def to_architecture(self, preset_name: str = "baseline_cv") -> KalmanOLTWArchitecture:
        """Build the current architecture bundle from the compatibility config."""

        return build_default_kalman_oltw_architecture(self, preset_name=preset_name)


def _default_step_transitions() -> tuple[StepTransition, ...]:
    return (
        StepTransition(
            label="hold_reference",
            reference_advance=0,
            query_advance=1,
            weight=HOLD_REFERENCE_WEIGHT,
        ),
        StepTransition(
            label="diagonal",
            reference_advance=1,
            query_advance=1,
            weight=DIAGONAL_WEIGHT,
        ),
        StepTransition(
            label="query_double",
            reference_advance=1,
            query_advance=2,
            weight=QUERY_DOUBLE_WEIGHT,
        ),
        StepTransition(
            label="reference_double",
            reference_advance=2,
            query_advance=1,
            weight=REFERENCE_DOUBLE_WEIGHT,
        ),
    )


def build_default_kalman_oltw_architecture(
    config: KalmanFilterConfig | None = None,
    preset_name: str = "baseline_cv",
) -> KalmanOLTWArchitecture:
    """Return the current production architecture without changing behavior."""

    selected_config = config or KalmanFilterConfig()
    return KalmanOLTWArchitecture(
        preset_name=preset_name,
        search_policy=SearchPolicyConfig(
            name="uncertainty_window_v1",
            min_search_half_window=selected_config.min_search_half_window,
            uncertainty_scale=selected_config.uncertainty_scale,
            start_anywhere=selected_config.start_anywhere,
            recovery_strategy="full_width",
        ),
        step_pattern=StepPatternConfig(
            name="default_normalized_v1",
            transitions=_default_step_transitions(),
        ),
        coupling=CouplingConfig(
            name="gate_and_prior_v1",
            mode="gate_and_prior",
            position_prior_weight=selected_config.position_prior_weight,
            measurement_source="normalized_row_argmin",
            confidence_source="row_best_score_margin",
        ),
        tracker=TrackerConfig(
            name="constant_velocity_v1",
            state_model="constant_velocity",
            position_variance=selected_config.position_variance,
            velocity_variance=selected_config.velocity_variance,
            process_position_variance=selected_config.process_position_variance,
            process_velocity_variance=selected_config.process_velocity_variance,
            measurement_variance=selected_config.measurement_variance,
            min_velocity=selected_config.min_velocity,
            max_velocity=selected_config.max_velocity,
        ),
    )


_KALMAN_EXPERIMENT_PRESETS: dict[str, KalmanExperimentPreset] = {
    "baseline_cv": KalmanExperimentPreset(
        name="baseline_cv",
        description="Current constant-velocity Kalman baseline with uncertainty-scaled search, position prior, and the default normalized step pattern.",
        tracker_model="constant_velocity",
        search_policy="uncertainty_window_v1",
        step_pattern="default_normalized_v1",
        coupling="gate_and_prior_v1",
        implemented=True,
    ),
    "adaptive_noise_cv": KalmanExperimentPreset(
        name="adaptive_noise_cv",
        description="Planned constant-velocity variant with adaptive process and measurement noise driven by row confidence and innovation statistics.",
        tracker_model="constant_velocity_adaptive_noise",
        search_policy="uncertainty_window_v1",
        step_pattern="default_normalized_v1",
        coupling="confidence_aware_gate_and_prior",
        implemented=False,
    ),
    "constant_acceleration": KalmanExperimentPreset(
        name="constant_acceleration",
        description="Planned linear constant-acceleration tracker for testing whether explicit acceleration reduces lag during tempo changes.",
        tracker_model="constant_acceleration",
        search_policy="uncertainty_window_v1",
        step_pattern="default_normalized_v1",
        coupling="gate_and_prior_v1",
        implemented=False,
    ),
    "narrow_window": KalmanExperimentPreset(
        name="narrow_window",
        description="Planned search-policy stress test with tighter gating around the predicted position.",
        tracker_model="constant_velocity",
        search_policy="narrow_uncertainty_window",
        step_pattern="default_normalized_v1",
        coupling="gate_and_prior_v1",
        implemented=False,
    ),
    "wide_window": KalmanExperimentPreset(
        name="wide_window",
        description="Planned search-policy stress test with wider gating and more aggressive recovery expansion.",
        tracker_model="constant_velocity",
        search_policy="wide_uncertainty_window",
        step_pattern="default_normalized_v1",
        coupling="gate_and_prior_v1",
        implemented=False,
    ),
    "hold_diag_only": KalmanExperimentPreset(
        name="hold_diag_only",
        description="Planned minimal transition set that removes the double-step transitions to test whether they are destabilizing the tracker.",
        tracker_model="constant_velocity",
        search_policy="uncertainty_window_v1",
        step_pattern="hold_diag_only",
        coupling="gate_and_prior_v1",
        implemented=False,
    ),
    "symmetric_doubles": KalmanExperimentPreset(
        name="symmetric_doubles",
        description="Planned retuned transition set that keeps the double-step moves but treats them as a controlled, symmetric experiment.",
        tracker_model="constant_velocity",
        search_policy="uncertainty_window_v1",
        step_pattern="symmetric_doubles",
        coupling="gate_and_prior_v1",
        implemented=False,
    ),
}


def list_kalman_oltw_presets(include_planned: bool = True) -> tuple[str, ...]:
    """List the current and planned preset names for future comparisons."""

    presets = (
        preset_name
        for preset_name, preset in sorted(_KALMAN_EXPERIMENT_PRESETS.items())
        if include_planned or preset.implemented
    )
    return tuple(presets)


def get_kalman_oltw_preset(name: str) -> KalmanExperimentPreset:
    """Return one preset scaffold by name."""

    normalized_name = name.strip().lower()
    try:
        return _KALMAN_EXPERIMENT_PRESETS[normalized_name]
    except KeyError as error:
        supported = ", ".join(list_kalman_oltw_presets(include_planned=True))
        raise ValueError(f"Unknown kalman_oltw preset {name!r}. Supported presets: {supported}.") from error


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
    architecture = config.to_architecture()
    measurement_indices, measurement_scores, filtered_states, trace = _run_kalman_guided_online_dtw(
        reference_values=reference_values,
        query_values=query_values,
        config=config,
        architecture=architecture,
        metric=metric,
        return_trace=True,
    )

    reference_indices = np.rint(filtered_states[:, 0]).astype(np.int64)
    reference_indices = np.clip(reference_indices, 0, reference_features.frame_times.size - 1)
    reference_indices = np.maximum.accumulate(reference_indices)
    query_indices = np.arange(query_features.frame_times.size, dtype=np.int64)
    path = np.column_stack([reference_indices, query_indices])
    trace_summary = _summarize_trace(trace)

    return AlignmentResult(
        method_name=DEFAULT_METHOD_NAME,
        reference_id=reference_features.metadata.get("recording_id", "reference"),
        query_id=query_features.metadata.get("recording_id", "query"),
        reference_times=reference_features.frame_times[reference_indices],
        query_times=query_features.frame_times[query_indices],
        path=path,
        metadata={
            "backend": "python_streaming_normalized_dtw",
            "measurement_source": architecture.coupling.measurement_source,
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
            "architecture_preset": architecture.preset_name,
            "tracker_model": architecture.tracker.state_model,
            "tracker_config_name": architecture.tracker.name,
            "search_policy_name": architecture.search_policy.name,
            "coupling_name": architecture.coupling.name,
            "coupling_mode": architecture.coupling.mode,
            "step_pattern_name": architecture.step_pattern.name,
            "step_pattern": {
                "transitions": tuple(
                    (transition.reference_advance, transition.query_advance)
                    for transition in architecture.step_pattern.transitions
                ),
                "labels": tuple(transition.label for transition in architecture.step_pattern.transitions),
                "weights": tuple(transition.weight for transition in architecture.step_pattern.transitions),
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
            **trace_summary,
        },
    )


def _run_kalman_guided_online_dtw(
    reference_values: np.ndarray,
    query_values: np.ndarray,
    config: KalmanFilterConfig,
    metric: str,
    architecture: KalmanOLTWArchitecture | None = None,
    return_trace: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, KalmanOnlineTrace]:
    """Run streaming normalized DP and Kalman updates one query frame at a time."""

    num_reference_frames = reference_values.shape[0]
    num_query_frames = query_values.shape[0]
    metric_name = metric.strip().lower()
    if metric_name != "cosine":
        raise ValueError(f"Unsupported metric {metric!r}; kalman_oltw currently supports 'cosine'.")

    resolved_architecture = architecture or config.to_architecture()
    reference_normalized, reference_valid = _normalize_feature_rows(reference_values)
    query_normalized, query_valid = _normalize_feature_rows(query_values)

    initial_velocity = _estimate_initial_velocity(
        num_reference_frames=num_reference_frames,
        num_query_frames=num_query_frames,
    )
    state = np.array([0.0, initial_velocity], dtype=np.float64)
    covariance = np.diag(
        [
            resolved_architecture.tracker.position_variance,
            resolved_architecture.tracker.velocity_variance,
        ],
    ).astype(np.float64)

    max_query_advance = max(
        transition.query_advance for transition in resolved_architecture.step_pattern.transitions
    )
    row_history: dict[int, tuple[np.ndarray, np.ndarray]] = {
        lag: (
            np.full(num_reference_frames, np.inf, dtype=np.float64),
            np.zeros(num_reference_frames, dtype=np.float64),
        )
        for lag in range(1, max_query_advance + 1)
    }

    measurement_indices = np.zeros(num_query_frames, dtype=np.int64)
    measurement_scores = np.zeros(num_query_frames, dtype=np.float64)
    filtered_states = np.zeros((num_query_frames, 2), dtype=np.float64)

    trace = _initialize_trace(
        num_query_frames=num_query_frames,
        transition_labels=tuple(
            transition.label for transition in resolved_architecture.step_pattern.transitions
        ),
    )
    minimum_reference_index = 0

    for query_index in range(num_query_frames):
        if query_index > 0:
            state, covariance = _kalman_predict(
                state,
                covariance,
                tracker_config=resolved_architecture.tracker,
            )

        predicted_state = state.copy()
        predicted_position = float(predicted_state[0])
        predicted_variance = float(covariance[0, 0])
        local_cost_row = _compute_cosine_local_cost_row(
            reference_normalized=reference_normalized,
            reference_valid=reference_valid,
            query_vector=query_normalized[query_index],
            query_is_valid=bool(query_valid[query_index]),
        )

        search_window = _compute_search_window(
            predicted_position=predicted_position,
            predicted_variance=predicted_variance,
            minimum_reference_index=minimum_reference_index,
            num_reference_frames=num_reference_frames,
            search_policy=resolved_architecture.search_policy,
            query_index=query_index,
        )

        current_totals, current_lengths, row_diagnostics = _update_streaming_row(
            local_cost_row=local_cost_row,
            row_history=row_history,
            search_window=search_window,
            query_index=query_index,
            predicted_position=predicted_position,
            coupling_config=resolved_architecture.coupling,
            step_pattern=resolved_architecture.step_pattern,
        )

        normalized_scores = _compute_normalized_scores(current_totals, current_lengths)
        measurement_index = _select_measurement_index(
            normalized_scores=normalized_scores,
            minimum_reference_index=minimum_reference_index,
        )
        measurement_value = float(measurement_index)
        measurement_score = float(normalized_scores[measurement_index])

        state, covariance, kalman_diagnostics = _kalman_update(
            state=state,
            covariance=covariance,
            measurement=measurement_value,
            num_reference_frames=num_reference_frames,
            previous_position=float(filtered_states[query_index - 1, 0]) if query_index > 0 else 0.0,
            tracker_config=resolved_architecture.tracker,
        )

        measurement_indices[query_index] = measurement_index
        measurement_scores[query_index] = measurement_score
        filtered_states[query_index] = state
        minimum_reference_index = int(max(minimum_reference_index, round(state[0])))
        _record_trace_step(
            trace=trace,
            query_index=query_index,
            predicted_state=predicted_state,
            predicted_variance=predicted_variance,
            search_window=search_window,
            filtered_state=state,
            measurement_index=measurement_index,
            measurement_score=measurement_score,
            row_diagnostics=row_diagnostics,
            kalman_diagnostics=kalman_diagnostics,
        )

        for transition_label, count in row_diagnostics.transition_usage.items():
            trace.transition_usage[transition_label] = trace.transition_usage.get(transition_label, 0) + count
        for lag in range(max_query_advance, 1, -1):
            row_history[lag] = row_history[lag - 1]
        row_history[1] = (current_totals, current_lengths)

    if return_trace:
        return measurement_indices, measurement_scores, filtered_states, trace
    return measurement_indices, measurement_scores, filtered_states


def _initialize_trace(
    num_query_frames: int,
    transition_labels: tuple[str, ...],
) -> KalmanOnlineTrace:
    return KalmanOnlineTrace(
        predicted_positions=np.zeros(num_query_frames, dtype=np.float64),
        predicted_velocities=np.zeros(num_query_frames, dtype=np.float64),
        predicted_position_variances=np.zeros(num_query_frames, dtype=np.float64),
        filtered_positions=np.zeros(num_query_frames, dtype=np.float64),
        filtered_velocities=np.zeros(num_query_frames, dtype=np.float64),
        measurement_indices=np.zeros(num_query_frames, dtype=np.int64),
        measurement_scores=np.zeros(num_query_frames, dtype=np.float64),
        innovations=np.zeros(num_query_frames, dtype=np.float64),
        innovation_variances=np.zeros(num_query_frames, dtype=np.float64),
        search_left_bounds=np.zeros(num_query_frames, dtype=np.int64),
        search_right_bounds=np.zeros(num_query_frames, dtype=np.int64),
        search_radii=np.zeros(num_query_frames, dtype=np.float64),
        used_recovery_window=np.zeros(num_query_frames, dtype=bool),
        row_finite_counts=np.zeros(num_query_frames, dtype=np.int64),
        row_best_score_margins=np.zeros(num_query_frames, dtype=np.float64),
        transition_usage={label: 0 for label in transition_labels},
    )


def _record_trace_step(
    trace: KalmanOnlineTrace,
    query_index: int,
    predicted_state: np.ndarray,
    predicted_variance: float,
    search_window: SearchWindow,
    filtered_state: np.ndarray,
    measurement_index: int,
    measurement_score: float,
    row_diagnostics: RowUpdateDiagnostics,
    kalman_diagnostics: KalmanUpdateDiagnostics,
) -> None:
    trace.predicted_positions[query_index] = float(predicted_state[0])
    trace.predicted_velocities[query_index] = float(predicted_state[1])
    trace.predicted_position_variances[query_index] = predicted_variance
    trace.filtered_positions[query_index] = float(filtered_state[0])
    trace.filtered_velocities[query_index] = float(filtered_state[1])
    trace.measurement_indices[query_index] = measurement_index
    trace.measurement_scores[query_index] = measurement_score
    trace.innovations[query_index] = kalman_diagnostics.innovation
    trace.innovation_variances[query_index] = kalman_diagnostics.innovation_variance
    trace.search_left_bounds[query_index] = search_window.left
    trace.search_right_bounds[query_index] = search_window.right
    trace.search_radii[query_index] = search_window.radius
    trace.used_recovery_window[query_index] = row_diagnostics.used_recovery_window
    trace.row_finite_counts[query_index] = row_diagnostics.finite_candidate_count
    trace.row_best_score_margins[query_index] = row_diagnostics.best_score_margin


def _summarize_trace(trace: KalmanOnlineTrace) -> dict[str, object]:
    search_widths = trace.search_right_bounds - trace.search_left_bounds + 1
    finite_margins = trace.row_best_score_margins[np.isfinite(trace.row_best_score_margins)]
    mean_row_best_margin = float(np.mean(finite_margins)) if finite_margins.size else 0.0
    return {
        "mean_abs_innovation": float(np.mean(np.abs(trace.innovations))),
        "max_abs_innovation": float(np.max(np.abs(trace.innovations))),
        "mean_search_width": float(np.mean(search_widths)),
        "max_search_width": int(np.max(search_widths)),
        "recovery_row_count": int(np.count_nonzero(trace.used_recovery_window)),
        "mean_row_finite_count": float(np.mean(trace.row_finite_counts)),
        "mean_row_best_score_margin": mean_row_best_margin,
        "transition_usage": dict(trace.transition_usage),
    }


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
    search_policy: SearchPolicyConfig,
    query_index: int,
) -> SearchWindow:
    radius = float(
        max(
            search_policy.min_search_half_window,
            int(np.ceil(search_policy.uncertainty_scale * np.sqrt(max(predicted_variance, 0.0)))),
        )
    )
    if query_index == 0 and search_policy.start_anywhere:
        return SearchWindow(
            left=0,
            right=num_reference_frames - 1,
            radius=float(max(num_reference_frames, 1)),
            recovery_mode="start_anywhere",
        )

    left = max(minimum_reference_index, int(np.floor(predicted_position - radius)))
    right = min(num_reference_frames - 1, int(np.ceil(predicted_position + radius)))
    if left > right:
        left = minimum_reference_index
        right = minimum_reference_index
    return SearchWindow(
        left=left,
        right=right,
        radius=max(radius, 1.0),
        recovery_mode="window",
    )


def _update_streaming_row(
    local_cost_row: np.ndarray,
    row_history: dict[int, tuple[np.ndarray, np.ndarray]],
    search_window: SearchWindow,
    query_index: int,
    predicted_position: float,
    coupling_config: CouplingConfig,
    step_pattern: StepPatternConfig,
) -> tuple[np.ndarray, np.ndarray, RowUpdateDiagnostics]:
    num_reference_frames = local_cost_row.size
    current_totals = np.full(num_reference_frames, np.inf, dtype=np.float64)
    current_lengths = np.zeros(num_reference_frames, dtype=np.float64)
    transition_usage = {transition.label: 0 for transition in step_pattern.transitions}

    for reference_index in range(search_window.left, search_window.right + 1):
        local_cost = float(local_cost_row[reference_index])
        if coupling_config.position_prior_weight > 0.0:
            local_cost += coupling_config.position_prior_weight * (
                ((reference_index - predicted_position) / search_window.radius) ** 2
            )

        if query_index == 0:
            current_totals[reference_index] = local_cost
            current_lengths[reference_index] = 1.0
            continue

        best_total = np.inf
        best_length = 0.0
        best_normalized = np.inf
        best_transition_label: str | None = None

        for transition in step_pattern.transitions:
            if query_index < transition.query_advance:
                continue
            predecessor_index = reference_index - transition.reference_advance
            if predecessor_index < 0 or predecessor_index >= num_reference_frames:
                continue
            predecessor_totals, predecessor_lengths = row_history[transition.query_advance]
            if not np.isfinite(predecessor_totals[predecessor_index]):
                continue

            candidate_total = predecessor_totals[predecessor_index] + transition.weight * local_cost
            candidate_length = predecessor_lengths[predecessor_index] + transition.weight
            candidate_normalized = candidate_total / candidate_length
            if candidate_normalized < best_normalized:
                best_total = candidate_total
                best_length = candidate_length
                best_normalized = candidate_normalized
                best_transition_label = transition.label
                continue
            if (
                np.isclose(candidate_normalized, best_normalized)
                and step_pattern.tie_break_policy == "lowest_total_cost"
                and candidate_total < best_total
            ):
                best_total = candidate_total
                best_length = candidate_length
                best_normalized = candidate_normalized
                best_transition_label = transition.label

        current_totals[reference_index] = best_total
        current_lengths[reference_index] = best_length
        if best_transition_label is not None:
            transition_usage[best_transition_label] += 1

    if not np.isfinite(current_totals[search_window.left : search_window.right + 1]).any():
        if search_window.left == 0 and search_window.right == num_reference_frames - 1:
            reachable_counts = {
                lag: int(np.count_nonzero(np.isfinite(totals)))
                for lag, (totals, _) in row_history.items()
            }
            raise RuntimeError(
                "Streaming online DTW produced no reachable cells for the current row. "
                f"query_index={query_index} reachable_counts={reachable_counts}"
            )
        recovered_totals, recovered_lengths, recovered_diagnostics = _update_streaming_row(
            local_cost_row=local_cost_row,
            row_history=row_history,
            search_window=SearchWindow(
                left=0,
                right=num_reference_frames - 1,
                radius=max(float(num_reference_frames), 1.0),
                recovery_mode="full_width",
            ),
            query_index=query_index,
            predicted_position=predicted_position,
            coupling_config=coupling_config,
            step_pattern=step_pattern,
        )
        recovered_diagnostics.used_recovery_window = True
        return recovered_totals, recovered_lengths, recovered_diagnostics

    normalized_scores = _compute_normalized_scores(current_totals, current_lengths)
    finite_scores = normalized_scores[np.isfinite(normalized_scores)]
    best_score, second_best_score, best_score_margin = _compute_score_margin(finite_scores)
    return (
        current_totals,
        current_lengths,
        RowUpdateDiagnostics(
            finite_candidate_count=int(finite_scores.size),
            best_score=best_score,
            second_best_score=second_best_score,
            best_score_margin=best_score_margin,
            used_recovery_window=False,
            transition_usage=transition_usage,
        ),
    )


def _compute_score_margin(finite_scores: np.ndarray) -> tuple[float, float, float]:
    if finite_scores.size == 0:
        return np.inf, np.inf, 0.0
    if finite_scores.size == 1:
        return float(finite_scores[0]), np.inf, 0.0

    ordered_scores = np.partition(finite_scores, 1)
    best_score = float(ordered_scores[0])
    second_best_score = float(ordered_scores[1])
    return best_score, second_best_score, second_best_score - best_score


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
    tracker_config: TrackerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    transition = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    process_noise = np.diag(
        [tracker_config.process_position_variance, tracker_config.process_velocity_variance],
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
    tracker_config: TrackerConfig,
) -> tuple[np.ndarray, np.ndarray, KalmanUpdateDiagnostics]:
    observation = np.array([[1.0, 0.0]], dtype=np.float64)
    identity = np.eye(2, dtype=np.float64)
    measurement_noise = np.array([[tracker_config.measurement_variance]], dtype=np.float64)

    innovation = np.array([measurement], dtype=np.float64) - (observation @ state)
    innovation_covariance = observation @ covariance @ observation.T + measurement_noise
    kalman_gain = covariance @ observation.T @ np.linalg.inv(innovation_covariance)
    updated_state = state + (kalman_gain @ innovation).reshape(-1)
    updated_covariance = (identity - kalman_gain @ observation) @ covariance

    updated_state[1] = float(
        np.clip(updated_state[1], tracker_config.min_velocity, tracker_config.max_velocity)
    )
    updated_state[0] = float(np.clip(updated_state[0], previous_position, num_reference_frames - 1))
    updated_covariance = (updated_covariance + updated_covariance.T) * 0.5
    return (
        updated_state,
        updated_covariance,
        KalmanUpdateDiagnostics(
            innovation=float(innovation[0]),
            innovation_variance=float(innovation_covariance[0, 0]),
        ),
    )


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

    tracker_config = config.to_architecture().tracker
    state = np.array([measurements[0], initial_velocity], dtype=np.float64)
    covariance = np.diag([tracker_config.position_variance, tracker_config.velocity_variance]).astype(
        np.float64
    )
    filtered_states = np.zeros((measurements.size, 2), dtype=np.float64)
    observed_mask = np.isfinite(measurements)
    previous_position = max(0.0, float(measurements[0]))

    for query_index in range(measurements.size):
        if query_index > 0:
            state, covariance = _kalman_predict(state, covariance, tracker_config=tracker_config)

        measurement = float(measurements[query_index]) if np.isfinite(measurements[query_index]) else state[0]
        state, covariance, _ = _kalman_update(
            state=state,
            covariance=covariance,
            measurement=measurement,
            num_reference_frames=num_reference_frames,
            previous_position=previous_position,
            tracker_config=tracker_config,
        )
        filtered_states[query_index] = state
        previous_position = state[0]

    return filtered_states, observed_mask


def _estimate_initial_velocity(
    num_reference_frames: int,
    num_query_frames: int,
) -> float:
    return float(num_reference_frames / max(num_query_frames, 1))
