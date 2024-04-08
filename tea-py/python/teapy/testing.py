from functools import partial
from math import isclose

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.strategies._internal.utils import defines_strategy
from numpy.testing import assert_allclose
from pandas.testing import assert_series_equal

rtol = 1e-5
atol = 1e-3

settings.register_profile("test", suppress_health_check=[HealthCheck(3)])
settings.load_profile("test")

isclose = partial(isclose, rel_tol=rtol, abs_tol=atol)


def assert_isclose3(a, b, c, *args):
    assert isclose(a, b)
    assert isclose(b, c)
    for v in args:
        assert isclose(a, v)


assert_series_equal = partial(
    assert_series_equal,
    rtol=rtol,
    atol=atol,
    check_index=False,
    check_dtype=False,
    check_names=False,
)

assert_allclose = partial(assert_allclose, rtol=rtol, atol=atol)


def assert_allclose3(a, b, c, *args):
    assert_allclose(a, b)
    assert_allclose(b, c)
    for v in args:
        assert_allclose(a, v)


# 同个数组中的数如果差距过大, 在计算时太小的数会被忽略
STABLE_FLOAT_MIN, STABLE_FLOAT_MAX = -10, 10
STABLE_INT_MIN, STABLE_INT_MAX = int(-1e3), int(1e3)
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

dtype_element_map_unstable = {
    np.float64: st.floats(width=64),
    np.float32: st.floats(width=32),
    # both teapy and pandas will overflow
    np.int32: st.integers(min_value=-(2**25), max_value=2**25 - 1),
    np.int64: st.integers(min_value=-(2**55), max_value=2**55 - 1),
}


@defines_strategy()
def make_arr(shape=100, nan_p=0.05, unique=False, dtype=None, stable=True):
    """
    make a random array using hypothesis
    shape: Array shape
    nan_p: Probability of nan in array
    dtype: Dtype of the array
    stable: Limit the difference of array elements to avoid floating point errors
    """
    assert nan_p >= 0, "nan_p must in 0 - 1"
    assert nan_p <= 1, "nan_p must in 0 - 1"

    if isinstance(shape, int):
        shape = (shape,)

    @st.composite
    def draw_arr(draw):
        arr_dtype = draw(st.sampled_from(dtype_list)) if dtype is None else dtype
        dtype_map = dtype_element_map_stable if stable else dtype_element_map_unstable
        arr = draw(
            arrays(
                arr_dtype,
                shape,
                elements=dtype_map[arr_dtype],
                unique=unique,
            )
        )
        if nan_p > 0 and arr_dtype in [np.float64, np.float32]:
            nan_mask = np.random.binomial(1, nan_p, shape)
            np.putmask(arr, nan_mask, np.nan)
        if stable and arr_dtype in [np.float64, np.float32]:
            min_, max_ = np.nanmax(arr), np.nanmin(arr)
            if ~np.isnan(max_) and max_ - abs(min_) * 1e6:  # suppose max > 0
                arr = np.where(arr < 1e-4, np.random.randn(*arr.shape), arr)
            # else:
            #     min_, max_ = np.max(arr), np.min(arr)
        return arr

    return draw_arr()
