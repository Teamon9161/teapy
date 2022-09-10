import numpy as np
import pandas as pd
import statsmodels.api as sm
from hypothesis import given
from hypothesis import strategies as st

import teapy as tp
from teapy.testing import assert_series_equal, make_arr


@given(make_arr(30), st.integers(1, 5), st.booleans())
def test_ts_reg(arr, window, stable):
    def ts_reg(s):
        s = s.dropna()
        if s.size > 1:
            reg = sm.OLS(s, sm.add_constant(np.arange(s.size) + 1)).fit()
            return reg.params[0] + reg.params[1] * s.size
        else:
            return np.nan

    arr = pd.Series(arr)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_reg(arr, window, min_periods=min_periods, stable=stable)
    res2 = arr.rolling(window, min_periods=min_periods).apply(ts_reg)
    assert_series_equal(res1, res2)


@given(make_arr(30), st.integers(1, 5), st.booleans())
def test_ts_reg_intercept(arr, window, stable):
    def ts_reg_intercept(s):
        s = s.dropna()
        if s.size > 1:
            reg = sm.OLS(s, sm.add_constant(np.arange(s.size) + 1)).fit()
            return reg.params[0]
        else:
            return np.nan

    arr = pd.Series(arr)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_reg_intercept(arr, window, min_periods=min_periods, stable=stable)
    res2 = arr.rolling(window, min_periods=min_periods).apply(ts_reg_intercept)
    assert_series_equal(res1, res2)


@given(make_arr(30), st.integers(1, 5), st.booleans())
def test_ts_reg_slope(arr, window, stable):
    def ts_reg_slope(s):
        s = s.dropna()
        if s.size > 1:
            reg = sm.OLS(s, sm.add_constant(np.arange(s.size) + 1)).fit()
            return reg.params[1]
        else:
            return np.nan

    arr = pd.Series(arr)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_reg_slope(arr, window, min_periods=min_periods, stable=stable)
    res2 = arr.rolling(window, min_periods=min_periods).apply(ts_reg_slope)
    assert_series_equal(res1, res2)


@given(make_arr(30), st.integers(1, 5), st.booleans())
def test_ts_tsf(arr, window, stable):
    def ts_tsf(s):
        s = s.dropna()
        if s.size > 1:
            reg = sm.OLS(s, sm.add_constant(np.arange(s.size) + 1)).fit()
            return reg.params[0] + reg.params[1] * (s.size + 1)
        else:
            return np.nan

    arr = pd.Series(arr)
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_tsf(arr, window, min_periods=min_periods, stable=stable)
    res2 = arr.rolling(window, min_periods=min_periods).apply(ts_tsf)
    assert_series_equal(res1, res2)
