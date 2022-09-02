import numpy as np
import pandas as pd
import teapy as tp
from hypothesis import given, strategies as st, assume
from teapy.testing import assert_series_equal, make_arr


@given(make_arr(100), st.integers(1, 20), st.integers(1, 20))
def test_ts_sma(arr, window, min_periods):
    assume(min_periods <= window)
    # 测试移动平均
    res1 = tp.ts_mean(arr, window, min_periods=min_periods)
    res2 = pd.Series(arr).rolling(
        window, min_periods=min_periods).mean()
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(100), st.integers(1, 20), st.integers(1, 20))
def test_ts_ewm(arr, window, min_periods):
    assume(min_periods <= window)
    def ewm(s):
        alpha = 2 / window
        n = s.count()
        if n > 0:
            weight = np.logspace(n-1, 0, num=n, base=(1-alpha))
            weight /= weight.sum()
            return (weight * s[~s.isna()]).sum()
        else:
            return np.nan
    # 测试指数加权移动平均
    res1 = tp.ts_ewm(arr, window, min_periods=min_periods)
    res2 = pd.Series(arr).rolling(window, min_periods=min_periods).apply(ewm)
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(100), st.integers(1, 20), st.integers(1, 20))
def test_ts_wma(arr, window, min_periods):
    # 测试加权移动平均
    assume(min_periods <= window)
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


@given(make_arr(100), st.integers(1, 20), st.integers(1, 20))
def test_ts_sum(arr, window, min_periods):
    assume(min_periods <= window)
    # 测试移动求和
    res1 = tp.ts_sum(arr, window, min_periods=min_periods)
    res2 = pd.Series(arr).rolling(
        window, min_periods=min_periods).sum()
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(100), st.integers(1, 3), st.integers(1, 3))
def test_ts_prod(arr, window, min_periods):
    assume(min_periods <= window)
    # 测试移动连乘
    res1 = tp.ts_prod(arr, window, min_periods=min_periods)
    res2 = pd.Series(arr).rolling(
        window, min_periods=min_periods).apply(lambda x: x.prod())
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(100), st.integers(1, 3), st.integers(1, 3))
def test_ts_prod_mean(arr, window, min_periods):
    assume(min_periods <= window)
    # 测试移动几何平均
    res1 = tp.ts_prod_mean(arr, window, min_periods=min_periods)
    res2 = pd.Series(arr).rolling(window, min_periods=min_periods).apply(
        lambda x: x.prod() ** (1 / x.notnull().sum()))
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(100), st.integers(1, 20), st.integers(1, 20))
def test_ts_max(arr, window, min_periods):
    assume(min_periods <= window)
    # 测试移动最大值
    res1 = tp.ts_max(arr, window, min_periods=min_periods)
    res2 = pd.Series(arr).rolling(
        window, min_periods=min_periods).max()
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(100), st.integers(1, 20), st.integers(1, 20))
def test_ts_min(arr, window, min_periods):
    assume(min_periods <= window)
    # 测试移动最小值
    res1 = tp.ts_min(arr, window, min_periods=min_periods)
    res2 = pd.Series(arr).rolling(
        window, min_periods=min_periods).min()
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(100, nan_p=0, unique=True), st.integers(1, 20), st.integers(1, 20))
def test_ts_argmax(arr, window, min_periods):
    # 对于重复值teapy总是取最后一个，而pandas取的是第一个，在无重复值的测试下进行
    assume(min_periods <= window)
    # 测试移动最大值索引
    res1 = tp.ts_argmax(arr, window, min_periods=min_periods)
    res2 = pd.Series(arr).rolling(window, min_periods=min_periods).apply(
        lambda x: x.idxmax() - x.index[0] + 1
    )
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(100, nan_p=0, unique=True), st.integers(1, 20), st.integers(1, 20))
def test_ts_argmin(arr, window, min_periods):
    # 对于重复值teapy总是取最后一个，而pandas取的是第一个，在无重复值的测试下进行
    assume(min_periods <= window)
    # 测试移动最小值索引
    res1 = tp.ts_argmin(arr, window, min_periods=min_periods)
    res2 = pd.Series(arr).rolling(window, min_periods=min_periods).apply(
        lambda x: x.idxmin() - x.index[0] + 1
    )
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(100, stable=True), st.integers(2, 20), st.integers(1, 20))
def test_ts_std(arr, window, min_periods):
    # 测试移动标准差
    assume(min_periods <= window)
    # 测试移动最小值
    res1 = tp.ts_std(arr, window, min_periods=min_periods)
    res2 = pd.Series(arr).rolling(
        window, min_periods=min_periods).apply(lambda s: s.std())
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(100, stable=True), st.integers(3, 20), st.integers(3, 20))
def test_ts_skew(arr, window, min_periods):
    # 测试移动偏度
    assume(min_periods <= window)
    res1 = tp.ts_skew(arr, window, min_periods=min_periods)
    # 不直接使用Rolling.skew的原因是当window中所有元素一样时会给出nan
    res2 = pd.Series(arr).rolling(window, min_periods=min_periods).apply(
        lambda s: s.skew())
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(100, unique=True, stable=True), st.integers(4, 20), st.integers(4, 20))
def test_ts_kurt(arr, window, min_periods):
    # 测试移动峰度
    assume(min_periods <= window)
    res1 = tp.ts_kurt(arr, window, min_periods=min_periods)
    # 不直接使用Rolling.kurt的原因是当window中所有元素一样时会给出nan，
    res2 = pd.Series(arr).rolling(window, min_periods=min_periods).apply(
        lambda s: s.kurt())
    assert_series_equal(pd.Series(res1), res2)
