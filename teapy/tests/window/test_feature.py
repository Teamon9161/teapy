import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

import teapy as tp
from teapy.testing import assert_series_equal, make_arr


@given(make_arr(30), st.integers(1, 5))
def test_ts_sma(arr, window):
    # 测试移动平均
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_mean(arr, window, min_periods=min_periods)
    res2 = pd.Series(arr).rolling(window, min_periods=min_periods).mean()
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(30), st.integers(1, 5))
def test_ts_ewm(arr, window):
    # 测试指数加权移动平均
    min_periods = np.random.randint(1, window + 1)

    def ewm(s):
        alpha = 2 / window
        n = s.count()
        if n > 0:
            weight = np.logspace(n - 1, 0, num=n, base=(1 - alpha))
            weight /= weight.sum()
            return (weight * s[~s.isna()]).sum()
        else:
            return np.nan

    res1 = tp.ts_ewm(arr, window, min_periods=min_periods)
    res2 = pd.Series(arr).rolling(window, min_periods=min_periods).apply(ewm)
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(30), st.integers(1, 5))
def test_ts_wma(arr, window):
    # 测试加权移动平均
    min_periods = np.random.randint(1, window + 1)

    def wma(s):
        n = s.count()
        if n > 0:
            weight = np.arange(n) + 1
            weight = weight / weight.sum()
            return (weight * s[~s.isna()]).sum()
        else:
            return np.nan

    res1 = tp.ts_wma(arr, window, min_periods=min_periods)
    res2 = pd.Series(arr).rolling(window, min_periods=min_periods).apply(wma)
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(30), st.integers(1, 5))
def test_ts_sum(arr, window):
    # 测试移动求和
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_sum(arr, window, min_periods=min_periods)
    res2 = pd.Series(arr).rolling(window, min_periods=min_periods).sum()
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(30), st.integers(1, 3))
def test_ts_prod(arr, window):
    # 测试移动连乘
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_prod(arr, window, min_periods=min_periods)
    res2 = (
        pd.Series(arr)
        .rolling(window, min_periods=min_periods)
        .apply(lambda x: x.prod())
    )
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(30), st.integers(1, 3))
def test_ts_prod_mean(arr, window):
    # 测试移动几何平均
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_prod_mean(arr, window, min_periods=min_periods)
    res2 = (
        pd.Series(arr)
        .rolling(window, min_periods=min_periods)
        .apply(lambda x: x.prod() ** (1 / x.notnull().sum()))
    )
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(30, stable=True), st.integers(2, 5))
def test_ts_std(arr, window):
    # 测试移动标准差
    min_periods = np.random.randint(2, window + 1)
    res1 = tp.ts_std(arr, window, min_periods=min_periods)
    res2 = (
        pd.Series(arr).rolling(window, min_periods=min_periods).apply(lambda s: s.std())
    )
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(30, unique=True, stable=True), st.integers(3, 5))
def test_ts_skew(arr, window):
    # 测试移动偏度
    min_periods = np.random.randint(3, window + 1)
    res1 = tp.ts_skew(arr, window, min_periods=min_periods)
    # 不直接使用Rolling.skew的原因是当window中所有元素一样时会给出nan
    res2 = (
        pd.Series(arr)
        .rolling(window, min_periods=min_periods)
        .apply(lambda s: s.skew())
    )
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(30, unique=True, stable=True), st.integers(4, 5))
def test_ts_kurt(arr, window):
    # 测试移动峰度
    min_periods = np.random.randint(4, window + 1)
    res1 = tp.ts_kurt(arr, window, min_periods=min_periods)
    # 不直接使用Rolling.kurt的原因是当window中所有元素一样时会给出nan，
    res2 = (
        pd.Series(arr)
        .rolling(window, min_periods=min_periods)
        .apply(lambda s: s.kurt())
    )
    assert_series_equal(pd.Series(res1), res2)
