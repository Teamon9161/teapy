import numpy as np
import pandas as pd
import teapy as tp
from hypothesis import given
from hypothesis import strategies as st
from teapy import Expr
from teapy.testing import assert_allclose3, make_arr


@given(make_arr(30, unique=True, stable=True), st.integers(1, 5), st.booleans())
def test_ts_stable(arr, window, stable):
    # test moving stable
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_stable(arr, window, min_periods=min_periods, stable=stable)
    res2 = Expr(arr).ts_stable(window, min_periods=min_periods, stable=stable).eview()
    res3 = arr.rolling(window, min_periods=min_periods).apply(
        lambda x: x.mean() / x.std() if x.std() != 0 else np.nan
    )
    assert_allclose3(res1, res2, res3)


@given(make_arr(30, unique=True, stable=True), st.integers(1, 5), st.booleans())
def test_ts_meanstdnorm(arr, window, stable):
    # test moving meanstd normalization
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_meanstdnorm(arr, window, min_periods=min_periods, stable=stable)
    res2 = (
        Expr(arr).ts_meanstdnorm(window, min_periods=min_periods, stable=stable).eview()
    )
    res3 = arr.rolling(window, min_periods=min_periods).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else np.nan
    )
    assert_allclose3(res1, res2, res3)


@given(make_arr(50, unique=True), st.integers(1, 5))
def test_ts_minmaxnorm(arr, window):
    # test moving minmax normalization
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_minmaxnorm(arr, window, min_periods=min_periods)
    res2 = Expr(arr).ts_minmaxnorm(window, min_periods=min_periods).eview()
    res3 = arr.rolling(window, min_periods=min_periods).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min())
        if x.max() != x.min()
        else np.nan
    )
    assert_allclose3(res1, res2, res3)
