import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

import teapy as tp
from teapy.testing import assert_series_equal, make_arr


@given(make_arr(200, unique=True), st.integers(3, 20))
def test_ts_cov(arr, window):
    # 测试移动协方差
    min_periods = np.random.randint(1, window + 1)
    arr1, arr2 = np.array_split(arr, 2)
    res1 = tp.ts_cov(arr1, arr2, window, min_periods=min_periods)
    res2 = pd.Series(arr1).rolling(window, min_periods=min_periods).cov(pd.Series(arr2))
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(200, unique=True), st.integers(3, 20))
def test_ts_corr(arr, window):
    # 测试移动相关系数
    min_periods = np.random.randint(1, window + 1)
    arr1, arr2 = np.array_split(arr, 2)
    res1 = tp.ts_corr(arr1, arr2, window, min_periods=min_periods)
    res2 = (
        pd.Series(arr1).rolling(window, min_periods=min_periods).corr(pd.Series(arr2))
    )
    assert_series_equal(pd.Series(res1), res2)
