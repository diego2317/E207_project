"""Offline DTW baseline implementation and shared alignment wrapper."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from scripts.models import AlignmentResult, FeatureSequence

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only in environments without numba
    njit = None


MOVE_DIAGONAL = np.int8(0)
MOVE_VERTICAL = np.int8(1)
MOVE_HORIZONTAL = np.int8(2)
MOVE_START = np.int8(-1)


if njit is not None:

    @njit(cache=True)
    def _accumulate_cost_numba(local_cost: np.ndarray) -> tuple[float, np.ndarray]:
        num_reference, num_query = local_cost.shape
        backpointers = np.empty((num_reference, num_query), dtype=np.int8)
        previous_costs = np.empty(num_query, dtype=np.float64)
        current_costs = np.empty(num_query, dtype=np.float64)

        previous_costs[0] = local_cost[0, 0]
        backpointers[0, 0] = MOVE_START
        for query_index in range(1, num_query):
            previous_costs[query_index] = (
                previous_costs[query_index - 1] + local_cost[0, query_index]
            )
            backpointers[0, query_index] = MOVE_HORIZONTAL

        for reference_index in range(1, num_reference):
            current_costs[0] = previous_costs[0] + local_cost[reference_index, 0]
            backpointers[reference_index, 0] = MOVE_VERTICAL
            for query_index in range(1, num_query):
                diagonal_cost = previous_costs[query_index - 1]
                vertical_cost = previous_costs[query_index]
                horizontal_cost = current_costs[query_index - 1]

                best_cost = diagonal_cost
                best_move = MOVE_DIAGONAL
                if vertical_cost < best_cost:
                    best_cost = vertical_cost
                    best_move = MOVE_VERTICAL
                if horizontal_cost < best_cost:
                    best_cost = horizontal_cost
                    best_move = MOVE_HORIZONTAL

                current_costs[query_index] = local_cost[reference_index, query_index] + best_cost
                backpointers[reference_index, query_index] = best_move

            temp = previous_costs
            previous_costs = current_costs
            current_costs = temp

        return float(previous_costs[num_query - 1]), backpointers

else:

    def _accumulate_cost_numba(local_cost: np.ndarray) -> tuple[float, np.ndarray]:
        return _accumulate_cost_reference(local_cost)


def run_offline_dtw(
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    metric: str = "cosine",
) -> AlignmentResult:
    """Run offline DTW and return a shared alignment result."""

    reference_values = _prepare_feature_values(reference_features.values)
    query_values = _prepare_feature_values(query_features.values)
    if reference_values.shape[0] == 0 or query_values.shape[0] == 0:
        raise ValueError("Offline DTW requires non-empty reference and query feature sequences.")

    local_cost = np.ascontiguousarray(
        cdist(reference_values, query_values, metric=metric),
        dtype=np.float64,
    )
    total_cost, backpointers = _accumulate_cost(local_cost)
    path = _backtrack_path(backpointers)

    reference_indices = path[:, 0]
    query_indices = path[:, 1]

    return AlignmentResult(
        method_name="offline_dtw",
        reference_id=reference_features.metadata.get("recording_id", "reference"),
        query_id=query_features.metadata.get("recording_id", "query"),
        reference_times=reference_features.frame_times[reference_indices],
        query_times=query_features.frame_times[query_indices],
        path=path,
        metadata={
            "distance_metric": metric,
            "local_cost_shape": tuple(int(value) for value in local_cost.shape),
            "total_cost": total_cost,
            "normalized_cost": total_cost / max(len(path), 1),
        },
    )


def _prepare_feature_values(values: np.ndarray) -> np.ndarray:
    prepared = np.asarray(values, dtype=np.float32)
    if prepared.ndim != 2:
        raise ValueError("Offline DTW inputs must be 2D feature matrices.")
    return np.ascontiguousarray(prepared)


def _accumulate_cost(local_cost: np.ndarray) -> tuple[float, np.ndarray]:
    prepared_cost = np.ascontiguousarray(local_cost, dtype=np.float64)
    if prepared_cost.ndim != 2:
        raise ValueError("DTW local_cost must be a 2D matrix.")
    if prepared_cost.shape[0] == 0 or prepared_cost.shape[1] == 0:
        raise ValueError("DTW local_cost must be non-empty.")
    return _accumulate_cost_numba(prepared_cost)


def _accumulate_cost_reference(local_cost: np.ndarray) -> tuple[float, np.ndarray]:
    num_reference, num_query = local_cost.shape
    accumulated = np.full((num_reference, num_query), np.inf, dtype=np.float64)
    backpointers = np.full((num_reference, num_query), MOVE_START, dtype=np.int8)

    accumulated[0, 0] = local_cost[0, 0]
    for query_index in range(1, num_query):
        accumulated[0, query_index] = accumulated[0, query_index - 1] + local_cost[0, query_index]
        backpointers[0, query_index] = MOVE_HORIZONTAL

    for reference_index in range(1, num_reference):
        accumulated[reference_index, 0] = (
            accumulated[reference_index - 1, 0] + local_cost[reference_index, 0]
        )
        backpointers[reference_index, 0] = MOVE_VERTICAL
        for query_index in range(1, num_query):
            diagonal_cost = accumulated[reference_index - 1, query_index - 1]
            vertical_cost = accumulated[reference_index - 1, query_index]
            horizontal_cost = accumulated[reference_index, query_index - 1]

            previous_cost = diagonal_cost
            move_code = MOVE_DIAGONAL
            if vertical_cost < previous_cost:
                previous_cost = vertical_cost
                move_code = MOVE_VERTICAL
            if horizontal_cost < previous_cost:
                previous_cost = horizontal_cost
                move_code = MOVE_HORIZONTAL

            accumulated[reference_index, query_index] = (
                local_cost[reference_index, query_index] + previous_cost
            )
            backpointers[reference_index, query_index] = move_code

    return float(accumulated[-1, -1]), backpointers


def _backtrack_path(backpointers: np.ndarray) -> np.ndarray:
    i, j = np.array(backpointers.shape) - 1
    reversed_path: list[tuple[int, int]] = [(int(i), int(j))]

    while i > 0 or j > 0:
        move_code = int(backpointers[i, j])
        if move_code == int(MOVE_DIAGONAL):
            i -= 1
            j -= 1
        elif move_code == int(MOVE_VERTICAL):
            i -= 1
        elif move_code == int(MOVE_HORIZONTAL):
            j -= 1
        else:
            raise ValueError(f"Invalid backpointer value {move_code} at {(i, j)}")
        reversed_path.append((int(i), int(j)))

    reversed_path.reverse()
    return np.asarray(reversed_path, dtype=np.int64)
