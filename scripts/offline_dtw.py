"""Offline DTW baseline implementation and shared alignment wrapper."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from scripts.models import AlignmentResult, FeatureSequence


def run_offline_dtw(
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    metric: str = "cosine",
) -> AlignmentResult:
    """Run offline DTW and return a shared alignment result."""

    local_cost = cdist(reference_features.values, query_features.values, metric=metric)
    accumulated_cost, backpointers = _accumulate_cost(local_cost)
    path = _backtrack_path(backpointers)

    reference_indices = path[:, 0]
    query_indices = path[:, 1]
    total_cost = float(accumulated_cost[-1, -1])

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


def _accumulate_cost(local_cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_reference, num_query = local_cost.shape
    accumulated = np.full((num_reference, num_query), np.inf, dtype=np.float64)
    backpointers = np.full((num_reference, num_query), -1, dtype=np.int8)

    accumulated[0, 0] = local_cost[0, 0]
    for i in range(num_reference):
        for j in range(num_query):
            if i == 0 and j == 0:
                continue

            candidates: list[tuple[float, int]] = []
            if i > 0 and j > 0:
                candidates.append((accumulated[i - 1, j - 1], 0))
            if i > 0:
                candidates.append((accumulated[i - 1, j], 1))
            if j > 0:
                candidates.append((accumulated[i, j - 1], 2))

            previous_cost, move_code = min(candidates, key=lambda item: item[0])
            accumulated[i, j] = local_cost[i, j] + previous_cost
            backpointers[i, j] = move_code

    return accumulated, backpointers


def _backtrack_path(backpointers: np.ndarray) -> np.ndarray:
    i, j = np.array(backpointers.shape) - 1
    reversed_path: list[tuple[int, int]] = [(int(i), int(j))]

    while i > 0 or j > 0:
        move_code = int(backpointers[i, j])
        if move_code == 0:
            i -= 1
            j -= 1
        elif move_code == 1:
            i -= 1
        elif move_code == 2:
            j -= 1
        else:
            raise ValueError(f"Invalid backpointer value {move_code} at {(i, j)}")
        reversed_path.append((int(i), int(j)))

    reversed_path.reverse()
    return np.asarray(reversed_path, dtype=np.int64)
