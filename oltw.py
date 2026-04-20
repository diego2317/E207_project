"""Fully offline version of Online Time Warping (OLTW) algorithm per Dixon et al."""

# standard imports
from typing import Callable, Union

# library imports
import numpy as np
from librosa.sequence import dtw
from numba import njit

# core imports
from ...constants import OLTW_STEPS, OLTW_WEIGHTS
from ...cost import CostMetric, normalize_by_path_length
from ...cost.cosine import cosine_mat2mat_parallel

# custom imports
from .base import OfflineAlignment
from ..utils import _validate_dtw_steps_weights, _validate_query_features_shape, _arrange_oltw_steps

# set constants for incrementing
BOTH = 0
ROW = 1
COLUMN = 2


# ---------------------------------------------------------------------------
# Numba-compiled OLTW path walk (no Python in the inner loop)
# c_int < 0 means "no band" (c is None). prev_int < 0 means no previous step.
# ---------------------------------------------------------------------------


@njit(cache=True)
def _get_min_cost_indices_numba(t: int, j: int, c_int: int, D_normalized: np.ndarray):
    """Min cost indices in current row/column; row wins ties. Returns (x, y)."""
    if c_int >= 0:
        row_start = max(0, j - c_int + 1)
        col_start = max(0, t - c_int + 1)
        cur_row = D_normalized[t, row_start:j + 1]
        cur_col = D_normalized[col_start:t + 1, j]
    else:
        row_start = 0
        col_start = 0
        cur_row = D_normalized[t, :j + 1]
        cur_col = D_normalized[:t + 1, j]

    row_min = np.min(cur_row)
    row_min_idx = np.argmin(cur_row)
    col_min = np.min(cur_col)
    col_min_idx = np.argmin(cur_col)
    if row_min <= col_min:
        return t, row_start + row_min_idx
    return col_start + col_min_idx, j


@njit(cache=True)
def _get_inc_numba(
    t: int,
    j: int,
    c_int: int,
    cur_run_count: int,
    prev_int: int,
    max_run_count: int,
    D_normalized: np.ndarray,
) -> int:
    """Return BOTH=0, ROW=1, or COLUMN=2."""
    if c_int >= 0 and t < c_int:
        return BOTH
    if cur_run_count >= max_run_count:
        return COLUMN if prev_int == ROW else ROW
    x, y = _get_min_cost_indices_numba(t, j, c_int, D_normalized)
    if x < t:
        return COLUMN
    if y < j:
        return ROW
    return BOTH


@njit(cache=True)
def _oltw_path_numba(
    D_normalized: np.ndarray,
    window_steps: np.ndarray,
    max_run_count: int,
    c_int: int,
    ref_length: int,
    query_length: int,
):
    """Compute OLTW path in Numba. Returns (path_buf, path_len). path_buf is (2, max_path_len)."""
    max_path_len = ref_length + query_length
    path_buf = np.zeros((2, max_path_len), dtype=np.int64)
    path_buf[0, 0] = 0
    path_buf[1, 0] = 0
    path_len = 1
    t, j = 0, 0
    cur_run_count = 0
    prev_int = -1  # no previous step

    while t < ref_length - 1 and j < query_length - 1:
        inc = _get_inc_numba(t, j, c_int, cur_run_count, prev_int, max_run_count, D_normalized)
        row_step = window_steps[inc, 0]
        col_step = window_steps[inc, 1]
        t = min(t + row_step, ref_length - 1)
        j = min(j + col_step, query_length - 1)
        if inc == prev_int:
            cur_run_count += 1
        else:
            cur_run_count = 1
        if inc != BOTH:
            prev_int = inc
        path_buf[0, path_len] = j
        path_buf[1, path_len] = t
        path_len += 1

    return path_buf, path_len


class OfflineOLTW(OfflineAlignment):
    """Fully offline version of OLTW Algorithm without banding."""

    def __init__(
        self,
        reference_features: np.ndarray,
        DTW_steps: np.ndarray = OLTW_STEPS,
        DTW_weights: np.ndarray = OLTW_WEIGHTS,
        window_steps: np.ndarray = OLTW_STEPS,
        cost_metric: Union[str, Callable, CostMetric] = "cosine",
        max_run_count: int = 3,
        c: int = None,
        use_parallel_cost: bool = False,
    ):
        """Initialize OfflineOLTW algorithm.

        Args:
            reference_features: Features for the reference audio.
                Shape (n_features, n_frames)
            DTW_steps: cost matrix windowing steps. Shape (n_steps, 2)
            DTW_weights: cost matrix windowing weights. Shape (n_steps, 1)
            window_steps: OLTW transition steps. Shape (n_steps, 2)
                Index 0 = reference (row), 1 = query (column), 2 = both.
            cost_metric: Cost metric to use for computing distances.
                Can be a string name, callable function, or CostMetric instance.
            max_run_count: Maximum run count. Defaults to 3.
            c: band size for comparing costs. # TODO: optimize based on banding
            use_parallel_cost: If True and cost_metric is cosine, compute cost
                matrix in parallel (Numba). Can help when BLAS is single-threaded;
                may be slower when BLAS is already multi-threaded.
        """
        super().__init__(reference_features, cost_metric)
        self.use_parallel_cost = use_parallel_cost

        # initialize query and reference locations
        self.t, self.j = 0, 0  # t is reference (row), j is query (column)
        self.c = c

        # validate steps and weights
        DTW_steps = np.array(DTW_steps)
        DTW_weights = np.array(DTW_weights)
        window_steps = np.array(window_steps)
        _validate_dtw_steps_weights(DTW_steps, DTW_weights) # validate DTW steps and weights
        window_steps = _arrange_oltw_steps(window_steps) # arrange steps by comparing slopes
        self.DTW_steps = DTW_steps
        self.DTW_weights = DTW_weights

        # validate and store transition steps
        # window_steps should have shape (3, 2) for [BOTH, ROW, COLUMN] transitions
        # Each row specifies (row_increment, column_increment)
        if window_steps.shape[0] != 3 or window_steps.shape[1] != 2:
            raise ValueError(f"window_steps must have shape (3, 2), got {window_steps.shape}")
        self.window_steps = window_steps

        # initialize alignment parameters
        self.max_run_count = max_run_count
        self.cur_run_count = 0
        self.prev = None  # previous step taken

        # path built in align() with preallocated arrays
        self.path = None

    def get_inc(self, D_normalized: np.ndarray):
        """Check which direction to increment based on normalized costs."""
        # handle initial period
        if self.c is not None and self.t < self.c:
            return BOTH

        # handle maximum run count
        if self.cur_run_count >= self.max_run_count:
            if self.prev == ROW:
                return COLUMN
            else:
                return ROW

        # calculate min cost and select which direction to increment
        x, y = self._get_min_cost_indices(D_normalized)

        if x < self.t:
            return COLUMN
        if y < self.j:
            return ROW
        else:
            return BOTH

    def _get_min_cost_indices(self, D_normalized: np.ndarray):
        """Calculate the min cost index in current row and column.

        Args:
            D_normalized (np.ndarray): Normalized accumulated cost matrix.
        """
        # Access the last c elements of the current row and column, or all history if c is None.
        if self.c is not None:
            row_start = max(0, self.j - self.c + 1)
            col_start = max(0, self.t - self.c + 1)
            cur_row = D_normalized[self.t, row_start:self.j + 1]
            cur_col = D_normalized[col_start:self.t + 1, self.j]
        else:
            row_start = 0
            col_start = 0
            cur_row = D_normalized[self.t, :self.j + 1]
            cur_col = D_normalized[:self.t + 1, self.j]

        # Vectorized: row wins ties (same as original loop order)
        row_min = np.min(cur_row)
        row_min_idx = np.argmin(cur_row)
        col_min = np.min(cur_col)
        col_min_idx = np.argmin(cur_col)
        if row_min <= col_min:
            return self.t, row_start + row_min_idx
        return col_start + col_min_idx, self.j

    def align(self, query_features: np.ndarray):
        """Align query features to reference features.

        Args:
            query_features: Query feature matrix. Shape (n_features, n_frames)

        Returns:
            Alignment path from OLTW. Shape (query_length, )
        """
        # validate input query features
        _validate_query_features_shape(query_features)

        # reset state for this alignment run
        self.t, self.j = 0, 0
        self.cur_run_count = 0
        self.prev = None

        # compute full cost matrix (optional Numba parallel for cosine)
        if self.use_parallel_cost and getattr(self.cost_metric, "name", None) == "cosine":
            C = cosine_mat2mat_parallel(
                np.ascontiguousarray(self.reference_features, dtype=np.float64),
                np.ascontiguousarray(query_features, dtype=np.float64),
            )
        else:
            C = self.cost_metric.mat2mat(self.reference_features, query_features)

        # compute accumulated cost matrix up front
        D = dtw(
            backtrack=False,  # no backtrack since we only care about the accumulated cost matrix
            C=C,
            step_sizes_sigma=self.DTW_steps,
            weights_mul=self.DTW_weights,
        )

        # normalize by path length (manhattan distance)
        D_normalized = normalize_by_path_length(D)
        if D_normalized.dtype != np.float64:
            D_normalized = np.ascontiguousarray(D_normalized, dtype=np.float64)

        query_length = query_features.shape[1]
        ref_length = self.reference_length
        c_int = int(self.c) if self.c is not None else -1
        window_steps_i64 = np.asarray(self.window_steps, dtype=np.int64)

        path_buf, path_len = _oltw_path_numba(
            D_normalized,
            window_steps_i64,
            self.max_run_count,
            c_int,
            ref_length,
            query_length,
        )
        self.path = path_buf[:, :path_len]
        return self.path


def run_offline_oltw(
    reference_features: np.ndarray,
    query_features: np.ndarray,
    DTW_steps: np.ndarray = OLTW_STEPS,
    DTW_weights: np.ndarray = OLTW_WEIGHTS,
    window_steps: np.ndarray = OLTW_STEPS,
    cost_metric: Union[str, Callable, CostMetric] = "cosine",
    max_run_count: int = 3,
    c: int = None,
    use_parallel_cost: bool = True,
):
    """Offline OLTW algorithm.

    Args:
        reference_features: Reference features. Shape (n_features, n_frames)
        query_features: Query features. Shape (n_features, n_frames)
        DTW_steps: DTW window steps. Shape (n_steps, 2)
        DTW_weights: DTW window weights. Shape (n_steps, 1)
        window_steps: OLTW transition steps. Shape (n_steps, 2)
        cost_metric: Cost metric to use for computing distances.
            Can be a string name, callable function, or CostMetric instance.
        max_run_count: Maximum run count. Defaults to 3.
        c: band size for comparing costs.
        use_parallel_cost: If True and cost_metric is cosine, use Numba-parallel cost matrix.
    """
    offline_oltw = OfflineOLTW(
        reference_features, DTW_steps, DTW_weights, window_steps,
        cost_metric, max_run_count, c, use_parallel_cost=use_parallel_cost,
    )
    return offline_oltw.align(query_features)
