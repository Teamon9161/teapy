import numpy as np
import pandas as pd
import teapy as tp
from hypothesis import given
from hypothesis import strategies as st
from teapy import Expr
from teapy.testing import assert_allclose3, make_arr


@given(make_arr(30), st.integers(3, 5), st.booleans())
def test_ts_cov(arr, window, stable):
    # test moving covariance
    min_periods = np.random.randint(1, window + 1)
    arr1, arr2 = np.array_split(arr, 2)
    arr1 = pd.Series(arr1, copy=False)
    arr2 = pd.Series(arr2, copy=False)
    res1 = tp.ts_cov(arr1, arr2, window, min_periods=min_periods, stable=stable)
    res2 = (
        Expr(arr1).ts_cov(arr2, window, min_periods=min_periods, stable=stable).eview()
    )
    res3 = pd.Series(arr1).rolling(window, min_periods=min_periods).cov(pd.Series(arr2))
    assert_allclose3(res1, res2, res3)


@given(make_arr(30, unique=True), st.integers(3, 5), st.booleans())
def test_ts_corr(arr, window, stable):
    # test moving correlation
    min_periods = np.random.randint(1, window + 1)
    arr1, arr2 = np.array_split(arr, 2)
    arr1 = pd.Series(arr1, copy=False)
    arr2 = pd.Series(arr2, copy=False)
    res1 = tp.ts_corr(arr1, arr2, window, min_periods=min_periods, stable=stable)
    res2 = (
        Expr(arr1).ts_corr(arr2, window, min_periods=min_periods, stable=stable).eview()
    )
    res3 = arr1.rolling(window, min_periods=min_periods).corr(arr2)
    assert_allclose3(res1, res2, res3)
