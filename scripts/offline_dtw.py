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

    @njit(cache=True)
    def _accumulate_euclidean_cost_numba(
        reference_values: np.ndarray,
        query_values: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        num_reference, num_dims = reference_values.shape
        num_query = query_values.shape[0]
        backpointers = np.empty((num_reference, num_query), dtype=np.int8)
        previous_costs = np.empty(num_query, dtype=np.float64)
        current_costs = np.empty(num_query, dtype=np.float64)

        first_reference = reference_values[0]
        first_query = query_values[0]
        local_cost = 0.0
        for dim_index in range(num_dims):
            difference = first_reference[dim_index] - first_query[dim_index]
            local_cost += difference * difference
        previous_costs[0] = np.sqrt(local_cost)
        backpointers[0, 0] = MOVE_START

        for query_index in range(1, num_query):
            local_cost = 0.0
            query_frame = query_values[query_index]
            for dim_index in range(num_dims):
                difference = first_reference[dim_index] - query_frame[dim_index]
                local_cost += difference * difference
            previous_costs[query_index] = previous_costs[query_index - 1] + np.sqrt(local_cost)
            backpointers[0, query_index] = MOVE_HORIZONTAL

        for reference_index in range(1, num_reference):
            reference_frame = reference_values[reference_index]
            local_cost = 0.0
            for dim_index in range(num_dims):
                difference = reference_frame[dim_index] - first_query[dim_index]
                local_cost += difference * difference
            current_costs[0] = previous_costs[0] + np.sqrt(local_cost)
            backpointers[reference_index, 0] = MOVE_VERTICAL

            for query_index in range(1, num_query):
                query_frame = query_values[query_index]
                local_cost = 0.0
                for dim_index in range(num_dims):
                    difference = reference_frame[dim_index] - query_frame[dim_index]
                    local_cost += difference * difference

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

                current_costs[query_index] = np.sqrt(local_cost) + best_cost
                backpointers[reference_index, query_index] = best_move

            temp = previous_costs
            previous_costs = current_costs
            current_costs = temp

        return float(previous_costs[num_query - 1]), backpointers

    @njit(cache=True)
    def _accumulate_cosine_cost_numba(
        reference_values: np.ndarray,
        query_values: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        num_reference, num_dims = reference_values.shape
        num_query = query_values.shape[0]
        backpointers = np.empty((num_reference, num_query), dtype=np.int8)
        previous_costs = np.empty(num_query, dtype=np.float64)
        current_costs = np.empty(num_query, dtype=np.float64)

        first_reference = reference_values[0]
        first_query = query_values[0]
        dot_product = 0.0
        reference_norm = 0.0
        query_norm = 0.0
        for dim_index in range(num_dims):
            reference_value = first_reference[dim_index]
            query_value = first_query[dim_index]
            dot_product += reference_value * query_value
            reference_norm += reference_value * reference_value
            query_norm += query_value * query_value
        if reference_norm == 0.0 or query_norm == 0.0:
            previous_costs[0] = np.nan
        else:
            previous_costs[0] = 1.0 - dot_product / np.sqrt(reference_norm * query_norm)
        backpointers[0, 0] = MOVE_START

        for query_index in range(1, num_query):
            query_frame = query_values[query_index]
            dot_product = 0.0
            reference_norm = 0.0
            query_norm = 0.0
            for dim_index in range(num_dims):
                reference_value = first_reference[dim_index]
                query_value = query_frame[dim_index]
                dot_product += reference_value * query_value
                reference_norm += reference_value * reference_value
                query_norm += query_value * query_value
            if reference_norm == 0.0 or query_norm == 0.0:
                local_cost = np.nan
            else:
                local_cost = 1.0 - dot_product / np.sqrt(reference_norm * query_norm)
            previous_costs[query_index] = previous_costs[query_index - 1] + local_cost
            backpointers[0, query_index] = MOVE_HORIZONTAL

        for reference_index in range(1, num_reference):
            reference_frame = reference_values[reference_index]
            dot_product = 0.0
            reference_norm = 0.0
            query_norm = 0.0
            for dim_index in range(num_dims):
                reference_value = reference_frame[dim_index]
                query_value = first_query[dim_index]
                dot_product += reference_value * query_value
                reference_norm += reference_value * reference_value
                query_norm += query_value * query_value
            if reference_norm == 0.0 or query_norm == 0.0:
                local_cost = np.nan
            else:
                local_cost = 1.0 - dot_product / np.sqrt(reference_norm * query_norm)
            current_costs[0] = previous_costs[0] + local_cost
            backpointers[reference_index, 0] = MOVE_VERTICAL

            for query_index in range(1, num_query):
                query_frame = query_values[query_index]
                dot_product = 0.0
                reference_norm = 0.0
                query_norm = 0.0
                for dim_index in range(num_dims):
                    reference_value = reference_frame[dim_index]
                    query_value = query_frame[dim_index]
                    dot_product += reference_value * query_value
                    reference_norm += reference_value * reference_value
                    query_norm += query_value * query_value
                if reference_norm == 0.0 or query_norm == 0.0:
                    local_cost = np.nan
                else:
                    local_cost = 1.0 - dot_product / np.sqrt(reference_norm * query_norm)

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

                current_costs[query_index] = local_cost + best_cost
                backpointers[reference_index, query_index] = best_move

            temp = previous_costs
            previous_costs = current_costs
            current_costs = temp

        return float(previous_costs[num_query - 1]), backpointers

    @njit(cache=True)
    def _backtrack_path_numba(backpointers: np.ndarray) -> np.ndarray:
        i = backpointers.shape[0] - 1
        j = backpointers.shape[1] - 1
        path_length = 1

        while i > 0 or j > 0:
            move_code = backpointers[i, j]
            if move_code == MOVE_DIAGONAL:
                i -= 1
                j -= 1
            elif move_code == MOVE_VERTICAL:
                i -= 1
            elif move_code == MOVE_HORIZONTAL:
                j -= 1
            else:
                raise ValueError("Invalid backpointer value.")
            path_length += 1

        path = np.empty((path_length, 2), dtype=np.int64)
        i = backpointers.shape[0] - 1
        j = backpointers.shape[1] - 1
        path_index = path_length - 1
        path[path_index, 0] = i
        path[path_index, 1] = j

        while path_index > 0:
            move_code = backpointers[i, j]
            if move_code == MOVE_DIAGONAL:
                i -= 1
                j -= 1
            elif move_code == MOVE_VERTICAL:
                i -= 1
            elif move_code == MOVE_HORIZONTAL:
                j -= 1
            else:
                raise ValueError("Invalid backpointer value.")
            path_index -= 1
            path[path_index, 0] = i
            path[path_index, 1] = j

        return path

else:

    def _accumulate_cost_numba(local_cost: np.ndarray) -> tuple[float, np.ndarray]:
        return _accumulate_cost_reference(local_cost)

    def _accumulate_euclidean_cost_numba(
        reference_values: np.ndarray,
        query_values: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        local_cost = _compute_local_cost_reference(reference_values, query_values, metric="euclidean")
        return _accumulate_cost_reference(local_cost)

    def _accumulate_cosine_cost_numba(
        reference_values: np.ndarray,
        query_values: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        local_cost = _compute_local_cost_reference(reference_values, query_values, metric="cosine")
        return _accumulate_cost_reference(local_cost)

    def _backtrack_path_numba(backpointers: np.ndarray) -> np.ndarray:
        return _backtrack_path_reference(backpointers)


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

    normalized_metric = metric.strip().lower()
    local_cost_shape = (int(reference_values.shape[0]), int(query_values.shape[0]))
    if normalized_metric == "cosine":
        optimized_reference_values = _prepare_feature_values(
            reference_values,
            dtype=np.float64,
        )
        optimized_query_values = _prepare_feature_values(
            query_values,
            dtype=np.float64,
        )
        total_cost, path = _run_offline_dtw_cosine_optimized(
            optimized_reference_values,
            optimized_query_values,
        )
    else:
        total_cost, path = _run_offline_dtw_reference(
            reference_values,
            query_values,
            metric=metric,
        )

    return _build_alignment_result(
        reference_features=reference_features,
        query_features=query_features,
        metric=metric,
        total_cost=total_cost,
        path=path,
        local_cost_shape=local_cost_shape,
    )


def _prepare_feature_values(
    values: np.ndarray,
    dtype: np.dtype[np.float32] | np.dtype[np.float64] | type[np.float32] | type[np.float64] = np.float32,
) -> np.ndarray:
    prepared = np.asarray(values, dtype=dtype)
    if prepared.ndim != 2:
        raise ValueError("Offline DTW inputs must be 2D feature matrices.")
    return np.ascontiguousarray(prepared)


def _run_offline_dtw_reference(
    reference_values: np.ndarray,
    query_values: np.ndarray,
    metric: str,
) -> tuple[float, np.ndarray]:
    local_cost = _compute_local_cost_reference(reference_values, query_values, metric=metric)
    total_cost, backpointers = _accumulate_cost(local_cost)
    return total_cost, _backtrack_path_reference(backpointers)


def _run_offline_dtw_optimized(
    reference_values: np.ndarray,
    query_values: np.ndarray,
    metric: str,
) -> tuple[float, np.ndarray]:
    total_cost, backpointers = _accumulate_cost_by_metric(
        reference_values,
        query_values,
        metric=metric,
    )
    return total_cost, _backtrack_path(backpointers)


def _run_offline_dtw_cosine_optimized(
    reference_values: np.ndarray,
    query_values: np.ndarray,
) -> tuple[float, np.ndarray]:
    local_cost = _compute_cosine_local_cost_optimized(reference_values, query_values)
    total_cost, backpointers = _accumulate_cost(local_cost)
    return total_cost, _backtrack_path(backpointers)


def _compute_local_cost_reference(
    reference_values: np.ndarray,
    query_values: np.ndarray,
    metric: str,
) -> np.ndarray:
    return np.ascontiguousarray(
        cdist(reference_values, query_values, metric=metric),
        dtype=np.float64,
    )


def _compute_cosine_local_cost_optimized(
    reference_values: np.ndarray,
    query_values: np.ndarray,
) -> np.ndarray:
    reference_norms = np.sqrt(np.sum(reference_values * reference_values, axis=1))
    query_norms = np.sqrt(np.sum(query_values * query_values, axis=1))

    normalized_reference = np.array(reference_values, copy=True, dtype=np.float64)
    normalized_query = np.array(query_values, copy=True, dtype=np.float64)

    valid_reference = reference_norms > 0.0
    valid_query = query_norms > 0.0
    normalized_reference[valid_reference] /= reference_norms[valid_reference, None]
    normalized_query[valid_query] /= query_norms[valid_query, None]

    local_cost = np.ascontiguousarray(1.0 - normalized_reference @ normalized_query.T, dtype=np.float64)
    local_cost[~valid_reference, :] = np.nan
    local_cost[:, ~valid_query] = np.nan
    return local_cost


def _accumulate_cost_by_metric(
    reference_values: np.ndarray,
    query_values: np.ndarray,
    metric: str,
) -> tuple[float, np.ndarray]:
    if metric == "euclidean":
        return _accumulate_euclidean_cost_numba(reference_values, query_values)
    if metric == "cosine":
        return _accumulate_cosine_cost_numba(reference_values, query_values)
    raise ValueError(f"Unsupported optimized metric {metric!r}.")


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
    prepared_backpointers = np.ascontiguousarray(backpointers, dtype=np.int8)
    if prepared_backpointers.ndim != 2:
        raise ValueError("DTW backpointers must be a 2D matrix.")
    if prepared_backpointers.shape[0] == 0 or prepared_backpointers.shape[1] == 0:
        raise ValueError("DTW backpointers must be non-empty.")
    return _backtrack_path_numba(prepared_backpointers)


def _backtrack_path_reference(backpointers: np.ndarray) -> np.ndarray:
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


def _build_alignment_result(
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    metric: str,
    total_cost: float,
    path: np.ndarray,
    local_cost_shape: tuple[int, int],
) -> AlignmentResult:
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
            "local_cost_shape": local_cost_shape,
            "total_cost": total_cost,
            "normalized_cost": total_cost / max(len(path), 1),
        },
    )
