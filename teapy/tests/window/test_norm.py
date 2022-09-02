import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

import teapy as tp
from teapy.testing import assert_series_equal, make_arr


@given(make_arr(30, unique=True, stable=True), st.integers(1, 5))
def test_ts_stable(arr, window):
    # 测试移动stable标准化
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_stable(arr, window, min_periods=min_periods)
    res2 = (
        pd.Series(arr)
        .rolling(window, min_periods=min_periods)
        .apply(lambda x: x.mean() / x.std() if x.std() != 0 else np.nan)
    )
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(30, unique=True, stable=True), st.integers(1, 5))
def test_ts_meanstdnorm(arr, window):
    # 测试移动meanstd标准化
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_meanstdnorm(arr, window, min_periods=min_periods)
    res2 = (
        pd.Series(arr)
        .rolling(window, min_periods=min_periods)
        .apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else np.nan)
    )
    assert_series_equal(pd.Series(res1), res2)


@given(make_arr(50, unique=True), st.integers(1, 5))
def test_ts_minmaxnorm(arr, window):
    # 测试移动stable标准化
    min_periods = np.random.randint(1, window + 1)
    res1 = tp.ts_minmaxnorm(arr, window, min_periods=min_periods)
    res2 = (
        pd.Series(arr)
        .rolling(window, min_periods=min_periods)
        .apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min())
            if x.max() != x.min()
            else np.nan
        )
    )
    assert_series_equal(pd.Series(res1), res2)
