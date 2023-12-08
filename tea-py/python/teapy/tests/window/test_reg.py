import numpy as np
import pandas as pd
import statsmodels.api as sm
import teapy as tp
from hypothesis import given
from hypothesis import strategies as st
from teapy import Expr
from teapy.testing import assert_allclose3, make_arr


@given(make_arr(30), st.integers(1, 5), st.booleans())
def test_ts_reg(arr, window, stable):
    # test moving regression
    def ts_reg(s):
        s = s.dropna()
        if s.size > 1:
            reg = sm.OLS(s, sm.add_constant(np.arange(s.size) + 1)).fit()
            return reg.params.iloc[0] + reg.params.iloc[1] * s.size
        else:
            return np.nan

    arr = pd.Series(arr)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_reg(arr, window, min_periods=min_periods, stable=stable)
    res2 = Expr(arr).ts_reg(window, min_periods=min_periods, stable=stable).eview()
    res3 = arr.rolling(window, min_periods=min_periods).apply(ts_reg)
    assert_allclose3(res1, res2, res3)


@given(make_arr(30), st.integers(1, 5), st.booleans())
def test_ts_reg_intercept(arr, window, stable):
    # test moving regression intercept
    def ts_reg_intercept(s):
        s = s.dropna()
        if s.size > 1:
            reg = sm.OLS(s, sm.add_constant(np.arange(s.size) + 1)).fit()
            return reg.params.iloc[0]
        else:
            return np.nan

    arr = pd.Series(arr)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_reg_intercept(arr, window, min_periods=min_periods, stable=stable)
    res2 = (
        Expr(arr)
        .ts_reg_intercept(window, min_periods=min_periods, stable=stable)
        .eview()
    )
    res3 = arr.rolling(window, min_periods=min_periods).apply(ts_reg_intercept)
    assert_allclose3(res1, res2, res3)


@given(make_arr(30), st.integers(1, 5), st.booleans())
def test_ts_reg_slope(arr, window, stable):
    # test moving regression slope
    def ts_reg_slope(s):
        s = s.dropna()
        if s.size > 1:
            reg = sm.OLS(s, sm.add_constant(np.arange(s.size) + 1)).fit()
            return reg.params.iloc[1]
        else:
            return np.nan

    arr = pd.Series(arr)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_reg_slope(arr, window, min_periods=min_periods, stable=stable)
    res2 = (
        Expr(arr).ts_reg_slope(window, min_periods=min_periods, stable=stable).eview()
    )
    res3 = arr.rolling(window, min_periods=min_periods).apply(ts_reg_slope)
    assert_allclose3(res1, res2, res3)


@given(make_arr(30), st.integers(1, 5), st.booleans())
def test_ts_tsf(arr, window, stable):
    # test moving time series forecast
    def ts_tsf(s):
        s = s.dropna()
        if s.size > 1:
            reg = sm.OLS(s, sm.add_constant(np.arange(s.size) + 1)).fit()
            return reg.params.iloc[0] + reg.params.iloc[1] * (s.size + 1)
        else:
            return np.nan

    arr = pd.Series(arr)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_tsf(arr, window, min_periods=min_periods, stable=stable)
    res2 = Expr(arr).ts_tsf(window, min_periods=min_periods, stable=stable).eview()
    res3 = arr.rolling(window, min_periods=min_periods).apply(ts_tsf)
    assert_allclose3(res1, res2, res3)
