import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

import teapy as tp
from teapy.testing import assert_series_equal, make_arr


@given(make_arr(30), st.integers(1, 5))
def test_ts_max(arr, window):
    # 测试移动最大值
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_max(arr, window, min_periods=min_periods)
    res2 = arr.rolling(window, min_periods=min_periods).max()
    assert_series_equal(res1, res2)


@given(make_arr(30), st.integers(1, 5))
def test_ts_min(arr, window):
    # 测试移动最小值
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_min(arr, window, min_periods=min_periods)
    res2 = arr.rolling(window, min_periods=min_periods).min()
    assert_series_equal(res1, res2)


@given(make_arr(30, nan_p=0, unique=True), st.integers(1, 5))
def test_ts_argmax(arr, window):
    # 测试移动最大值索引
    # 对于重复值teapy总是取最后一个，而pandas取的是第一个，因此在无重复值的测试下进行
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_argmax(arr, window, min_periods=min_periods)
    res2 = arr.rolling(window, min_periods=min_periods).apply(
        lambda x: x.idxmax() - x.index[0] + 1
    )
    assert_series_equal(res1, res2)


@given(make_arr(30, nan_p=0, unique=True), st.integers(1, 5))
def test_ts_argmin(arr, window):
    # 测试移动最小值索引
    # 对于重复值teapy总是取最后一个，而pandas取的是第一个，在无重复值的测试下进行
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_argmin(arr, window, min_periods=min_periods)
    res2 = arr.rolling(window, min_periods=min_periods).apply(
        lambda x: x.idxmin() - x.index[0] + 1
    )
    assert_series_equal(res1, res2)


@given(make_arr(30, nan_p=0, unique=True), st.integers(1, 5))
def test_ts_rank(arr, window):
    # 测试移动排名
    # 对于重复值teapy总是取最后一个，而pandas取的是第一个，在无重复值的测试下进行
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_rank(arr, window, min_periods=min_periods)
    res2 = arr.rolling(window, min_periods=min_periods).apply(
        lambda x: x.rank().iloc[-1]
    )
    assert_series_equal(res1, res2)

    # rank pct
    res3 = tp.ts_rank(arr, window, pct=True, min_periods=min_periods)
    res4 = arr.rolling(window, min_periods=min_periods).apply(
        lambda x: x.rank(pct=True).iloc[-1]
    )
    assert_series_equal(res3, res4)
