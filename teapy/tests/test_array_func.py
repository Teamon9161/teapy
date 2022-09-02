import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

import teapy as tp
from teapy.testing import assert_allclose, assert_series_equal, make_arr


@given(make_arr(30, unique=True, nan_p=0), st.integers(1, 5))
def test_argsort(arr, window):
    res1 = tp.argsort(arr)
    res2 = arr.argsort()
    assert_allclose(res1, res2)


@given(make_arr(30), st.integers(1, 5))
def test_rank(arr, window):
    res1 = tp.rank(arr)
    res2 = pd.Series(arr).rank()
    assert_series_equal(pd.Series(res1), res2)

    res3 = tp.rank(arr, pct=True)
    res4 = pd.Series(arr).rank(pct=True)
    assert_series_equal(pd.Series(res3), res4)


def test_array_func_2d():
    arr = np.random.randn(20, 20)
    res1 = tp.rank(arr, axis=0)
    res2 = pd.DataFrame(arr).rank(axis=0).values
    assert_allclose(res1, res2)
