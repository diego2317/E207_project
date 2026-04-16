"""Small in-repo online time warping baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from scripts.models import AlignmentResult, FeatureSequence

MOVE_DIAGONAL = np.int8(0)
MOVE_VERTICAL = np.int8(1)
MOVE_HORIZONTAL = np.int8(2)
MOVE_START = np.int8(3)
MOVE_UNSET = np.int8(-1)


@dataclass(slots=True)
class OLTWConfig:
    """Configuration for the in-repo OLTW baseline."""

    metric: str = "cosine"
    search_radius: int = 50
    scoring_mode: str = "cumulative"
    window_policy: str = "adaptive_band"


@dataclass(slots=True)
class OLTWState:
    """Row-wise dynamic-programming state for OLTW."""

    previous_costs: np.ndarray
    current_costs: np.ndarray
    previous_lengths: np.ndarray
    current_lengths: np.ndarray
    traceback_moves: np.ndarray
    best_reference_index: int
    processed_query_frames: int = 0
    evaluated_cells: int = 0


def run_oltw(
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    metric: str = "cosine",
    search_radius: int = 50,
    scoring_mode: str = "cumulative",
    window_policy: str = "adaptive_band",
) -> AlignmentResult:
    """Run a simple row-by-row OLTW baseline and return a shared alignment result."""

    if reference_features.values.shape[0] == 0 or query_features.values.shape[0] == 0:
        raise ValueError("OLTW requires non-empty reference and query feature sequences.")
    if search_radius < 0:
        raise ValueError("search_radius must be non-negative.")

    config = OLTWConfig(
        metric=metric,
        search_radius=search_radius,
        scoring_mode=scoring_mode,
        window_policy=window_policy,
    )
    _validate_config(config)

    num_reference = reference_features.values.shape[0]
    num_query = query_features.values.shape[0]
    state = _initialize_state(num_reference=num_reference, num_query=num_query)

    for query_index in range(num_query):
        window_start, window_end = _compute_window_bounds(
            best_reference_index=state.best_reference_index,
            num_reference=num_reference,
            config=config,
        )
        _update_row(
            state=state,
            reference_features=reference_features,
            query_features=query_features,
            query_index=query_index,
            window_start=window_start,
            window_end=window_end,
            config=config,
        )
        state.best_reference_index = _select_best_reference_index(
            costs=state.current_costs,
            lengths=state.current_lengths,
            scoring_mode=config.scoring_mode,
        )
        state.processed_query_frames = query_index + 1
        if query_index < num_query - 1:
            state.previous_costs[:] = state.current_costs
            state.previous_lengths[:] = state.current_lengths
            state.current_costs.fill(np.inf)
            state.current_lengths.fill(0)

    terminal_reference_index = state.best_reference_index
    terminal_score = float(state.current_costs[terminal_reference_index])
    terminal_length = int(state.current_lengths[terminal_reference_index])
    if not np.isfinite(terminal_score):
        raise ValueError("OLTW failed to find a finite terminal alignment score.")

    path = _traceback_path(
        traceback_moves=state.traceback_moves,
        terminal_query_index=num_query - 1,
        terminal_reference_index=terminal_reference_index,
    )
    reference_indices = path[:, 0]
    query_indices = path[:, 1]

    return AlignmentResult(
        method_name="oltw",
        reference_id=reference_features.metadata.get("recording_id", "reference"),
        query_id=query_features.metadata.get("recording_id", "query"),
        reference_times=reference_features.frame_times[reference_indices],
        query_times=query_features.frame_times[query_indices],
        path=path,
        metadata={
            "distance_metric": config.metric,
            "search_radius": config.search_radius,
            "scoring_mode": config.scoring_mode,
            "window_policy": config.window_policy,
            "processed_query_frames": state.processed_query_frames,
            "evaluated_cells": state.evaluated_cells,
            "terminal_reference_index": terminal_reference_index,
            "terminal_score": terminal_score,
            "terminal_path_length": terminal_length,
        },
    )


def run_oltw_global(
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    metric: str = "cosine",
    search_radius: int = 50,
    scoring_mode: str = "cumulative",
    window_policy: str = "global_band",
) -> AlignmentResult:
    """Placeholder for OLTW-global built on the same scoring/window interfaces."""

    raise NotImplementedError(
        "OLTW-global is not yet implemented. Reuse the OLTW scoring/window helpers in "
        "scripts.oltw to add a global-band policy on top of the same AlignmentResult contract."
    )


def _validate_config(config: OLTWConfig) -> None:
    if config.scoring_mode != "cumulative":
        raise ValueError(
            f"Unsupported scoring_mode={config.scoring_mode!r}. Use 'cumulative' for now."
        )
    if config.window_policy != "adaptive_band":
        raise ValueError(
            f"Unsupported window_policy={config.window_policy!r}. Use 'adaptive_band' for now."
        )


def _initialize_state(num_reference: int, num_query: int) -> OLTWState:
    return OLTWState(
        previous_costs=np.full(num_reference, np.inf, dtype=np.float64),
        current_costs=np.full(num_reference, np.inf, dtype=np.float64),
        previous_lengths=np.zeros(num_reference, dtype=np.int32),
        current_lengths=np.zeros(num_reference, dtype=np.int32),
        traceback_moves=np.full((num_query, num_reference), MOVE_UNSET, dtype=np.int8),
        best_reference_index=0,
    )


def _compute_window_bounds(
    best_reference_index: int,
    num_reference: int,
    config: OLTWConfig,
) -> tuple[int, int]:
    lower_bound = max(0, best_reference_index - config.search_radius)
    upper_bound = min(num_reference - 1, best_reference_index + config.search_radius)
    return lower_bound, upper_bound


def _update_row(
    state: OLTWState,
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    query_index: int,
    window_start: int,
    window_end: int,
    config: OLTWConfig,
) -> None:
    query_frame = query_features.values[query_index : query_index + 1]
    local_costs = cdist(
        reference_features.values[window_start : window_end + 1],
        query_frame,
        metric=config.metric,
    ).ravel()
    state.evaluated_cells += int(local_costs.size)

    for offset, reference_index in enumerate(range(window_start, window_end + 1)):
        local_cost = float(local_costs[offset])
        candidate_cost, candidate_length, candidate_move = _select_predecessor(
            state=state,
            query_index=query_index,
            reference_index=reference_index,
            config=config,
        )
        if candidate_move == MOVE_UNSET:
            continue
        state.current_costs[reference_index] = candidate_cost + local_cost
        state.current_lengths[reference_index] = candidate_length + 1
        state.traceback_moves[query_index, reference_index] = candidate_move


def _select_predecessor(
    state: OLTWState,
    query_index: int,
    reference_index: int,
    config: OLTWConfig,
) -> tuple[float, int, np.int8]:
    if query_index == 0 and reference_index == 0:
        return 0.0, 0, MOVE_START

    candidates: list[tuple[float, int, np.int8]] = []
    if query_index > 0 and reference_index > 0 and np.isfinite(state.previous_costs[reference_index - 1]):
        candidates.append(
            (
                float(state.previous_costs[reference_index - 1]),
                int(state.previous_lengths[reference_index - 1]),
                MOVE_DIAGONAL,
            )
        )
    if query_index > 0 and np.isfinite(state.previous_costs[reference_index]):
        candidates.append(
            (
                float(state.previous_costs[reference_index]),
                int(state.previous_lengths[reference_index]),
                MOVE_VERTICAL,
            )
        )
    if reference_index > 0 and np.isfinite(state.current_costs[reference_index - 1]):
        candidates.append(
            (
                float(state.current_costs[reference_index - 1]),
                int(state.current_lengths[reference_index - 1]),
                MOVE_HORIZONTAL,
            )
        )

    if not candidates:
        return np.inf, 0, MOVE_UNSET

    if config.scoring_mode == "cumulative":
        return min(candidates, key=lambda item: item[0])

    raise ValueError(f"Unsupported scoring_mode={config.scoring_mode!r}.")


def _select_best_reference_index(
    costs: np.ndarray,
    lengths: np.ndarray,
    scoring_mode: str,
) -> int:
    finite_indices = np.flatnonzero(np.isfinite(costs))
    if finite_indices.size == 0:
        raise ValueError("OLTW failed to produce any finite scores for the current query frame.")
    if scoring_mode == "cumulative":
        best_offset = int(np.argmin(costs[finite_indices]))
        return int(finite_indices[best_offset])
    raise ValueError(f"Unsupported scoring_mode={scoring_mode!r}.")


def _traceback_path(
    traceback_moves: np.ndarray,
    terminal_query_index: int,
    terminal_reference_index: int,
) -> np.ndarray:
    query_index = int(terminal_query_index)
    reference_index = int(terminal_reference_index)
    reversed_path: list[tuple[int, int]] = [(reference_index, query_index)]

    while True:
        move = traceback_moves[query_index, reference_index]
        if move == MOVE_START:
            break
        if move == MOVE_DIAGONAL:
            query_index -= 1
            reference_index -= 1
        elif move == MOVE_VERTICAL:
            query_index -= 1
        elif move == MOVE_HORIZONTAL:
            reference_index -= 1
        else:
            raise ValueError(
                "Traceback encountered an unset predecessor at "
                f"(query_index={query_index}, reference_index={reference_index})."
            )
        reversed_path.append((reference_index, query_index))

    reversed_path.reverse()
    return np.asarray(reversed_path, dtype=np.int64)
