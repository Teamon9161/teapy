import numpy as np
import pandas as pd
import teapy as tp
from hypothesis import given
from hypothesis import strategies as st
from teapy import Expr
from teapy.testing import assert_allclose3, make_arr


@given(make_arr(30), st.integers(1, 5))
def test_ts_max(arr, window):
    # test moving maximum
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_max(arr, window, min_periods=min_periods)
    res2 = Expr(arr).ts_max(window, min_periods=min_periods).eview()
    res3 = arr.rolling(window, min_periods=min_periods).max()
    assert_allclose3(res1, res2, res3)


@given(make_arr(30), st.integers(1, 5))
def test_ts_min(arr, window):
    # test moving minimum
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_min(arr, window, min_periods=min_periods)
    res2 = Expr(arr).ts_min(window, min_periods=min_periods).eview()
    res3 = arr.rolling(window, min_periods=min_periods).min()
    assert_allclose3(res1, res2, res3)


@given(make_arr(30, nan_p=0, unique=True), st.integers(1, 5))
def test_ts_argmax(arr, window):
    # test moving maximum index
    # for repeated values, teapy always takes the last one, while pandas takes the first one
    # so we use `unique=True`
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_argmax(arr, window, min_periods=min_periods)
    res2 = Expr(arr).ts_argmax(window, min_periods=min_periods).eview()
    res3 = arr.rolling(window, min_periods=min_periods).apply(
        lambda x: x.idxmax() - x.index[0] + 1
    )
    assert_allclose3(res1, res2, res3)


@given(make_arr(30, nan_p=0, unique=True), st.integers(1, 5))
def test_ts_argmin(arr, window):
    # test moving minimum index
    # for repeated values, teapy always takes the last one, while pandas takes the first one
    # so we use `unique=True`
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_argmin(arr, window, min_periods=min_periods)
    res2 = Expr(arr).ts_argmin(window, min_periods=min_periods).eview()
    res3 = arr.rolling(window, min_periods=min_periods).apply(
        lambda x: x.idxmin() - x.index[0] + 1
    )
    assert_allclose3(res1, res2, res3)


@given(make_arr(30, nan_p=0, unique=True), st.integers(1, 5))
def test_ts_rank(arr, window):
    # test moving rank
    # for repeated values, teapy always takes the last one, while pandas takes the first one
    # so we use `unique=True`
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_rank(arr, window, min_periods=min_periods)
    res2 = Expr(arr).ts_rank(window, min_periods=min_periods).eview()
    res3 = arr.rolling(window, min_periods=min_periods).apply(
        lambda x: x.rank().iloc[-1]
    )
    assert_allclose3(res1, res2, res3)

    # rank pct
    res4 = tp.ts_rank(arr, window, pct=True, min_periods=min_periods)
    res5 = Expr(arr).ts_rank(window, pct=True, min_periods=min_periods).eview()
    res6 = arr.rolling(window, min_periods=min_periods).apply(
        lambda x: x.rank(pct=True).iloc[-1]
    )
    assert_allclose3(res4, res5, res6)
