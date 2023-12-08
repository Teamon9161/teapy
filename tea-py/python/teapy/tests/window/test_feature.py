import numpy as np
import pandas as pd
import teapy as tp
from hypothesis import given
from hypothesis import strategies as st
from teapy import Expr
from teapy.testing import assert_allclose3, make_arr


@given(make_arr(30), st.integers(1, 5), st.booleans())
def test_ts_sma(arr, window, stable):
    # test moving average
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_sma(arr, window, min_periods=min_periods, stable=stable)
    res2 = Expr(arr).ts_sma(window, min_periods=min_periods, stable=stable).eview()
    res3 = arr.rolling(window, min_periods=min_periods).mean()
    assert_allclose3(res1, res2, res3)


@given(make_arr(30), st.integers(1, 5), st.booleans())
def test_ts_ewm(arr, window, stable):
    # test exponential weighted moving average
    arr = pd.Series(arr, copy=False)
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

    res1 = tp.ts_ewm(arr, window, min_periods=min_periods, stable=stable)
    res2 = Expr(arr).ts_ewm(window, min_periods=min_periods, stable=stable).eview()
    res3 = arr.rolling(window, min_periods=min_periods).apply(ewm)
    assert_allclose3(res1, res2, res3)


@given(make_arr(30), st.integers(1, 5), st.booleans())
def test_ts_wma(arr, window, stable):
    # test weighted moving average
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)

    def wma(s):
        n = s.count()
        if n > 0:
            weight = np.arange(n) + 1
            weight = weight / weight.sum()
            return (weight * s[~s.isna()]).sum()
        else:
            return np.nan

    res1 = tp.ts_wma(arr, window, min_periods=min_periods, stable=stable)
    res2 = Expr(arr).ts_wma(window, min_periods=min_periods, stable=stable).eview()
    res3 = arr.rolling(window, min_periods=min_periods).apply(wma)
    assert_allclose3(res1, res2, res3)


@given(make_arr(30), st.integers(1, 5), st.booleans())
def test_ts_sum(arr, window, stable):
    # test moving sum
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_sum(arr, window, min_periods=min_periods, stable=stable)
    res2 = Expr(arr).ts_sum(window, min_periods=min_periods, stable=stable).eview()
    res3 = arr.rolling(window, min_periods=min_periods).sum()
    assert_allclose3(res1, res2, res3)


@given(make_arr(30), st.integers(1, 3))
def test_ts_prod(arr, window):
    # test moving product
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_prod(arr, window, min_periods=min_periods)
    res2 = Expr(arr).ts_prod(window, min_periods=min_periods).eview()
    res3 = arr.rolling(window, min_periods=min_periods).apply(lambda x: x.prod())
    assert_allclose3(res1, res2, res3)


@given(make_arr(30), st.integers(1, 3))
def test_ts_prod_mean(arr, window):
    # test moving geometric mean
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_prod_mean(arr, window, min_periods=min_periods)
    res2 = Expr(arr).ts_prod_mean(window, min_periods=min_periods).eview()
    res3 = arr.rolling(window, min_periods=min_periods).apply(
        lambda x: x.prod() ** (1 / x.notnull().sum())
    )
    assert_allclose3(res1, res2, res3)


@given(make_arr(30, unique=True), st.integers(2, 5), st.booleans())
def test_ts_std(arr, window, stable):
    # test moving standard deviation
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(2, window + 1)
    res1 = tp.ts_std(arr, window, min_periods=min_periods, stable=True)
    res2 = Expr(arr).ts_std(window, min_periods=min_periods).eview()
    res3 = arr.rolling(window, min_periods=min_periods).apply(lambda s: s.std())
    assert_allclose3(res1, res2, res3)


@given(make_arr(30), st.integers(2, 5), st.booleans())
def test_ts_var(arr, window, stable):
    # test moving variance
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(2, window + 1)
    res1 = tp.ts_var(arr, window, min_periods=min_periods, stable=stable)
    res2 = Expr(arr).ts_var(window, min_periods=min_periods, stable=stable).eview()
    res3 = arr.rolling(window, min_periods=min_periods).apply(lambda s: s.var())
    assert_allclose3(res1, res2, res3)


@given(make_arr(30, unique=True), st.integers(3, 5), st.booleans())
def test_ts_skew(arr, window, stable):
    # test moving skewness
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(3, window + 1)
    res1 = tp.ts_skew(arr, window, min_periods=min_periods, stable=stable)
    res2 = Expr(arr).ts_skew(window, min_periods=min_periods, stable=stable).eview()
    # 不直接使用Rolling.skew的原因是当window中所有元素一样时会给出nan
    res3 = arr.rolling(window, min_periods=min_periods).apply(lambda s: s.skew())
    assert_allclose3(res1, res2, res3)


@given(make_arr(30, unique=True), st.integers(4, 5), st.booleans())
def test_ts_kurt(arr, window, stable):
    # test moving kurtosis
    arr = pd.Series(arr, copy=False)
    min_periods = np.random.randint(4, window + 1)
    res1 = tp.ts_kurt(arr, window, min_periods=min_periods, stable=stable)
    res2 = Expr(arr).ts_kurt(window, min_periods=min_periods, stable=stable).eview()
    # the reason why we don't use Rolling.kurt is that when all
    # elements in window are the same, the function will return nan
    res3 = arr.rolling(window, min_periods=min_periods).apply(lambda s: s.kurt())
    assert_allclose3(res1, res2, res3)
