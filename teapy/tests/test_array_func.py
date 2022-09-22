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


@given(make_arr((10, 10)), st.integers(0, 1))
def test_median(arr, axis):
    res1 = tp.median(arr, axis=axis)
    res2 = np.nanmedian(arr, axis=axis)
    assert_allclose(res1, res2)


@given(make_arr((10, 10)), st.floats(0, 1), st.integers(0, 1))
def test_quantile(arr, q, axis):
    res1 = tp.quantile(arr, q, axis=axis)
    res2 = np.nanquantile(arr, q, axis=axis)
    assert_allclose(res1, res2)


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


def test_fillna():
    s = pd.Series([np.nan, 5, 6, 733, np.nan, 34, np.nan, np.nan])
    for method in ["ffill", "bfill"]:
        assert_series_equal(tp.fillna(s, method), s.fillna(method=method))
        # test inplace
        s1 = s.copy()
        tp.fillna(s1, method, inplace=True)
        assert_series_equal(s1, s.fillna(method=method))
    # test fill value directly
    fill_value = 101
    assert_series_equal(tp.fillna(s, value=fill_value), s.fillna(fill_value))
    tp.fillna(s, value=fill_value, inplace=True)
    assert_series_equal(s, pd.Series([101, 5, 6, 733, 101, 34, 101, 101]))


def test_clip():
    s = pd.Series([np.nan, 5, 6, 733, np.nan, 34, 456, np.nan])
    assert_series_equal(tp.clip(s, 5, 100), s.clip(5, 100))
    s1 = s.copy()
    tp.clip(s1, 5, 100, inplace=True)
    assert_series_equal(s1, s.clip(5, 100))


def test_remove_nan():
    s = pd.Series([np.nan, 5, 6, 12, np.nan, 1, np.nan, np.nan])
    assert_series_equal(tp.remove_nan(s), s.dropna())


def test_zscore():
    s = pd.Series(np.arange(12).astype(float))
    for stable in False, True:
        s1 = tp.zscore(s, stable)
        s2 = (s - s.mean()) / s.std()
        assert_series_equal(s1, s2)
        s_copy = s.copy()
        tp.zscore(s_copy, stable, inplace=True)
        assert_series_equal(s_copy, s2)


def test_winsorize():
    s = pd.Series(np.arange(12).astype(float))

    # quantile method
    q = 0.05
    s1 = tp.winsorize(s, "quantile", q)
    lower, upper = np.nanquantile(s, [q, 1 - q])
    s2 = s.clip(lower, upper)
    assert_series_equal(s1, s2)
    # test quantile inplace
    s_copy = s.copy()
    tp.winsorize(s_copy, "quantile", q, inplace=True)
    assert_series_equal(s_copy, s2)

    # median method
    q = 1
    s1 = tp.winsorize(s, "median", q)
    median = s.median()
    mad = (s - median).abs().median()
    s2 = s.clip(median - q * mad, median + q * mad)
    assert_series_equal(s1, s2)

    # sigma method
    q = 1.2
    s1 = tp.winsorize(s, "sigma", q)
    mean = s.mean()
    std = s.std()
    s2 = s.clip(mean - q * std, mean + q * std)
    assert_series_equal(s1, s2)
