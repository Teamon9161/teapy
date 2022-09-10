import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

import teapy as tp
from teapy.testing import assert_allclose, assert_series_equal, isclose, make_arr


@given(make_arr(30, unique=True, nan_p=0), st.integers(1, 5))
def test_argsort(arr, window):
    res1 = tp.argsort(arr)
    res2 = arr.argsort()
    assert_allclose(res1, res2)


@given(make_arr(30), st.integers(1, 5))
def test_rank(arr, window):
    res1 = tp.rank(arr)
    res2 = pd.Series(arr).rank()
    assert_series_equal(pd.Series(res1), res2)

    res3 = tp.rank(arr, pct=True)
    res4 = pd.Series(arr).rank(pct=True)
    assert_series_equal(pd.Series(res3), res4)


@given(make_arr((10, 10), nan_p=0.5), st.integers(0, 1))
def test_count_nan(arr, axis):
    res1 = np.count_nonzero(np.isnan(arr), axis=axis)
    res2 = tp.count_nan(arr, axis=axis)
    assert_allclose(res1, res2)
    res3 = np.count_nonzero(~np.isnan(arr), axis=axis)
    res4 = tp.count_notnan(arr, axis=axis)
    assert_allclose(res3, res4)


@given(make_arr((10, 10)), st.integers(0, 1))
def test_mean(arr, axis):
    arr = pd.DataFrame(arr)
    res1 = arr.mean(axis=axis)
    res2 = tp.mean(arr, axis=axis)
    res3 = tp.mean(arr, axis=axis, stable=True)
    assert_series_equal(res1, res2)
    assert_series_equal(res1, res3)


@given(make_arr((10, 10)), st.integers(0, 1))
def test_std(arr, axis):
    arr = pd.DataFrame(arr)
    res1 = arr.std(axis=axis)
    res2 = tp.std(arr, axis=axis)
    res3 = tp.std(arr, axis=axis, stable=True)
    assert_series_equal(res1, res2)
    assert_series_equal(res1, res3)


@given(make_arr((10, 10), unique=True), st.integers(0, 1))
def test_skew(arr, axis):
    arr = pd.DataFrame(arr)
    res1 = arr.skew(axis=axis)
    res2 = tp.skew(arr, axis=axis)
    res3 = tp.skew(arr, axis=axis, stable=True)
    assert_series_equal(res1, res2)
    assert_series_equal(res1, res3)


@given(make_arr((10, 10), unique=True), st.integers(0, 1))
def test_kurt(arr, axis):
    arr = pd.DataFrame(arr)
    res1 = arr.kurt(axis=axis)
    res2 = tp.kurt(arr, axis=axis)
    res3 = tp.kurt(arr, axis=axis, stable=True)
    assert_series_equal(res1, res2)
    assert_series_equal(res1, res3)


@given(make_arr((10, 2)))
def test_cov(arr):
    cov_pd = pd.DataFrame(arr).cov().iloc[0, 1]
    arr1, arr2 = np.array_split(arr, 2, axis=1)
    cov1 = tp.cov(arr1, arr2, axis=0)[0]
    cov2 = tp.cov(arr1, arr2, axis=0, stable=True)[0]
    assert isclose(cov_pd, cov1) & isclose(cov_pd, cov2)


@given(make_arr((10, 2)))
def test_corr(arr):
    corr_pd = pd.DataFrame(arr).corr().iloc[0, 1]
    arr1, arr2 = np.array_split(arr, 2, axis=1)
    corr1 = tp.corr(arr1, arr2, axis=0)[0]
    corr2 = tp.corr(arr1, arr2, axis=0, stable=True)[0]
    if np.isnan(corr_pd):
        assert np.isnan(corr1) & np.isnan(corr2)
    else:
        assert isclose(corr_pd, corr1) & isclose(corr_pd, corr2)


def test_array_func_2d():
    arr = np.random.randn(20, 20)
    res1 = tp.rank(arr, axis=0)
    res2 = pd.DataFrame(arr).rank(axis=0).values
    assert_allclose(res1, res2)
