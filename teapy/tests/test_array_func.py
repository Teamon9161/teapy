import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

import teapy as tp
from teapy.testing import assert_allclose, assert_series_equal, make_arr


@given(make_arr(100, unique=True, nan_p=0), st.integers(1, 20))
def test_argsort(arr, window):
    res1 = tp.argsort(arr)
    res2 = arr.argsort()
    assert_allclose(res1, res2)


@given(make_arr(100), st.integers(1, 20))
def test_rank(arr, window):
    res1 = tp.rank(arr)
    res2 = pd.Series(arr).rank()
    assert_series_equal(pd.Series(res1), res2)
