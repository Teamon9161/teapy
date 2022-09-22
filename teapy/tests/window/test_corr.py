import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

import teapy as tp
from teapy.testing import assert_series_equal, make_arr


@given(make_arr(30, unique=True, stable=True), st.integers(3, 5), st.booleans())
def test_ts_cov(arr, window, stable):
    # 测试移动协方差
    min_periods = np.random.randint(1, window + 1)
    arr1, arr2 = np.array_split(arr, 2)
    arr1 = pd.Series(arr1, copy=False)
    arr2 = pd.Series(arr2, copy=False)
    res1 = tp.ts_cov(arr1, arr2, window, min_periods=min_periods, stable=stable)
    res2 = pd.Series(arr1).rolling(window, min_periods=min_periods).cov(pd.Series(arr2))
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(30, unique=True, stable=True), st.integers(3, 5), st.booleans())
def test_ts_corr(arr, window, stable):
    # 测试移动相关系数
    min_periods = np.random.randint(1, window + 1)
    arr1, arr2 = np.array_split(arr, 2)
    arr1 = pd.Series(arr1, copy=False)
    arr2 = pd.Series(arr2, copy=False)
    res1 = tp.ts_corr(arr1, arr2, window, min_periods=min_periods, stable=stable)
    res2 = arr1.rolling(window, min_periods=min_periods).corr(arr2)
    assert_series_equal(res1, res2)
