from functools import partial

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.strategies._internal.utils import defines_strategy
from numpy.testing import assert_allclose
from pandas.testing import assert_series_equal

assert_series_equal = partial(
    assert_series_equal,
    rtol=1e-5,
    atol=1e-6,
    check_index=False,
    check_dtype=False,
    check_names=False,
)

assert_allclose = partial(assert_allclose, rtol=1e-5, atol=1e-6)


# 同个数组中的数如果差距过大，在计算时太小的数会被忽略
STABLE_FLOAT_MIN, STABLE_FLOAT_MAX = -1000, 1000
STABLE_INT_MIN, STABLE_INT_MAX = int(-1e5), int(1e5)
dtype_list = [np.float64, np.float32, np.int32, np.int64]
dtype_element_map_stable = {
    np.float64: st.floats(
        width=16, min_value=STABLE_FLOAT_MIN, max_value=STABLE_FLOAT_MAX
    ),
    np.float32: st.floats(
        width=16, min_value=STABLE_FLOAT_MIN, max_value=STABLE_FLOAT_MAX
    ),
    np.int32: st.integers(min_value=STABLE_INT_MIN, max_value=STABLE_INT_MAX),
    # both teapy and pandas will overflow
    np.int64: st.integers(min_value=STABLE_INT_MIN, max_value=STABLE_INT_MAX),
}


@defines_strategy()
def make_arr(shape=100, nan_p=0.05, unique=False, dtype=None, stable=False):
    """make a random array using hypothesis
    shape: array shape
    nan_p: probability of nan in array
    dtype: dtype of the array
    stable: maximum value of array < minimum value of array * 1e6
    """
    if type(shape) is int:
        shape = (shape,)

    @st.composite
    def draw_arr(draw):
        arr_dtype = draw(st.sampled_from(dtype_list)) if dtype is None else dtype
        arr = draw(
            arrays(
                arr_dtype,
                shape,
                elements=dtype_element_map_stable[arr_dtype],
                unique=unique,
            )
        )
        if nan_p > 0 and arr_dtype in [np.float64, np.float32]:
            nan_mask = np.random.binomial(1, nan_p, shape)
            np.putmask(arr, nan_mask, np.nan)
        if stable:
            if arr_dtype in [np.float64, np.float32]:
                min_, max_ = np.nanmax(arr), np.nanmin(arr)
            else:
                min_, max_ = np.max(arr), np.min(arr)
            if ~np.isnan(max_) and max_ < min_ * 1e6:
                arr[arr < 1e-4] = 0.0

        return arr

    return draw_arr()
