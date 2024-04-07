import datetime
import time

import numpy as np
import pandas as pd
import teapy as tp
from hypothesis import given
from hypothesis import strategies as st
from teapy import Expr
from teapy.testing import (
    assert_allclose,
    assert_allclose3,
    assert_isclose3,
    assert_series_equal,
    make_arr,
)


def test_cast_datetime():
    t = int(time.time())
    now1 = datetime.datetime.utcfromtimestamp(t)
    now2 = tp.Expr(t).cast("datetime(s)").eview()
    assert now1 == now2


@given(make_arr(30, unique=True, nan_p=0))
def test_argsort(arr):
    res1 = tp.argsort(arr)
    res2 = Expr(arr).argsort().eview()
    res3 = arr.argsort()
    assert_allclose3(res1, res2, res3)
    res1 = tp.argsort(arr, rev=True)
    res2 = Expr(arr).argsort(rev=True).eview()
    res3 = arr.argsort()[::-1]
    assert_allclose3(res1, res2, res3)


@given(make_arr(30))
def test_rank(arr):
    res1 = tp.rank(arr)
    res2 = Expr(arr).rank().eview()
    res3 = pd.Series(arr).rank().values
    assert_allclose3(res1, res2, res3)

    res4 = tp.rank(arr, rev=True)
    res5 = Expr(arr).rank(rev=True).eview()
    res6 = pd.Series(arr).rank(ascending=False).values
    assert_allclose3(res4, res5, res6)

    res7 = tp.rank(arr, pct=True)
    res8 = Expr(arr).rank(pct=True).eview()
    res9 = pd.Series(arr).rank(pct=True).values
    assert_allclose3(res7, res8, res9)


@given(make_arr((10, 10)), st.integers(0, 1))
def test_max(arr, axis):
    res1 = tp.max(arr, axis=axis)
    res2 = Expr(arr).max(axis=axis).eview()
    res3 = np.nanmax(arr, axis=axis)
    assert_allclose3(res1, res2, res3)


def test_arg_partition():
    arr = np.array(
        [
            [6, 0, -2, 8, -7],
            [-2, 3, -10, 2, 9],
            [-11, -2, 5, 9, 3],
            [-6, 1, 3, -1, 6],
            [-10, -5, -1, -8, -3],
        ]
    )
    res1 = Expr(arr).arg_partition(1, axis=0).eview()
    exp1 = np.array([[2, 4, 1, 4, 0], [4, 2, 0, 3, 4]])
    assert_allclose(res1, exp1)
    res2 = Expr(arr).arg_partition(1, axis=1, rev=True).eview()
    exp2 = np.array([[3, 0], [4, 1], [3, 2], [4, 2], [2, 4]])
    assert_allclose(res2, exp2)
    arr = Expr([1, np.nan, 3, np.nan, np.nan])
    assert_allclose(arr.arg_partition(2, rev=True).eview(), np.array([2, 0, -1]))


def test_partition():
    arr = np.array(
        [
            [6, 0, -2, 8, -7],
            [-2, 3, -10, 2, 9],
            [-11, -2, 5, 9, 3],
            [-6, 1, 3, -1, 6],
            [-10, -5, -1, -8, -3],
        ]
    )
    res1 = Expr(arr).partition(1, axis=0).eview()
    exp1 = np.array([[-11, -5, -10, -8, -7], [-10, -2, -2, -1, -3]])
    assert_allclose(res1, exp1)
    res2 = Expr(arr).partition(1, axis=1, rev=True).eview()
    exp2 = np.array([[8, 6], [9, 3], [9, 5], [6, 3], [-1, -3]])
    assert_allclose(res2, exp2)
    res3 = Expr(arr).partition(4, axis=1, rev=True, sort=True).eview()
    exp3 = np.array(
        [
            [8, 6, 0, -2, -7],
            [9, 3, 2, -2, -10],
            [9, 5, 3, -2, -11],
            [6, 3, 1, -1, -6],
            [-1, -3, -5, -8, -10],
        ]
    )
    assert_allclose(res3, exp3)
    arr = Expr([1, np.nan, 3, np.nan, np.nan])
    assert_allclose(arr.partition(2, rev=True).eview(), np.array([3, 1, np.nan]))


@given(make_arr((10, 10)), st.integers(0, 1))
def test_min(arr, axis):
    res1 = tp.min(arr, axis=axis)
    res2 = Expr(arr).min(axis=axis).eview()
    res3 = np.nanmin(arr, axis=axis)
    assert_allclose3(res1, res2, res3)


@given(make_arr((10, 10)), st.integers(0, 1))
def test_sum(arr, axis):
    for stable in [True, False]:
        res1 = tp.sum(arr, axis=axis, stable=stable)
        res2 = Expr(arr).sum(axis=axis, stable=stable).eview()
        res3 = np.nansum(arr, axis=axis)
        assert_allclose3(res1, res2, res3)


@given(st.integers(0, 1))
def test_prod(axis):
    arr = np.random.randn(100, 10)
    res1 = tp.prod(arr, axis=axis)
    res2 = Expr(arr).prod(axis=axis).eview()
    res3 = np.nanprod(arr, axis=axis)
    assert_allclose3(res1, res2, res3)


@given(make_arr((10, 10)), st.integers(0, 1))
def test_cumsum(arr, axis):
    for stable in [True, False]:
        res1 = tp.cumsum(arr, axis=axis, stable=stable)
        res2 = Expr(arr).cumsum(axis=axis, stable=stable).eview()
        res3 = pd.DataFrame(arr).cumsum(axis=axis)
        assert_allclose3(res1, res2, res3)


@given(st.integers(0, 1))
def test_cumprod(axis):
    arr = np.random.randn(100, 10)
    res1 = tp.cumprod(arr, axis=axis)
    res2 = Expr(arr).cumprod(axis=axis).eview()
    res3 = pd.DataFrame(arr).cumprod(axis=axis)
    assert_allclose3(res1, res2, res3)


@given(make_arr((10, 10)), st.integers(0, 1))
def test_median(arr, axis):
    res1 = tp.median(arr, axis=axis)
    res2 = Expr(arr).median(axis=axis).eview()
    res3 = np.nanmedian(arr, axis=axis)
    assert_allclose3(res1, res2, res3)


@given(make_arr((10, 10)), st.floats(0, 1), st.integers(0, 1))
def test_quantile(arr, q, axis):
    res1 = tp.quantile(arr, q, axis=axis)
    res2 = Expr(arr).quantile(q, axis=axis).eview()
    res3 = np.nanquantile(arr, q, axis=axis)
    assert_allclose3(res1, res2, res3)


@given(make_arr((10, 10), nan_p=0.5), st.integers(0, 1))
def test_count_nan(arr, axis):
    res1 = np.count_nonzero(np.isnan(arr), axis=axis)
    res2 = tp.count_nan(arr, axis=axis)
    res3 = Expr(arr).count_nan(axis=axis).eview()
    assert_allclose3(res1, res2, res3)
    res4 = np.count_nonzero(~np.isnan(arr), axis=axis)
    res5 = tp.count_notnan(arr, axis=axis)
    res6 = Expr(arr).count_notnan(axis=axis).eview()
    assert_allclose3(res4, res5, res6)


@given(make_arr((10, 10)), st.integers(0, 1))
def test_mean(arr, axis):
    res1 = pd.DataFrame(arr).mean(axis=axis).values
    res2 = tp.mean(arr, axis=axis)
    res3 = tp.mean(arr, axis=axis, stable=True)
    res4 = Expr(arr).mean(axis=axis, stable=True).eview()
    assert_allclose3(res1, res2, res3, res4)


@given(make_arr((10, 10)), st.integers(0, 1))
def test_std(arr, axis):
    res1 = pd.DataFrame(arr).std(axis=axis).values
    res2 = tp.std(arr, axis=axis)
    res3 = tp.std(arr, axis=axis, stable=True)
    res4 = Expr(arr).std(axis=axis, stable=True).eview()
    assert_allclose3(res1, res2, res3, res4)


@given(make_arr((10, 10)), st.integers(0, 1))
def test_var(arr, axis):
    res1 = pd.DataFrame(arr).var(axis=axis).values
    res2 = tp.var(arr, axis=axis)
    res3 = tp.var(arr, axis=axis, stable=True)
    res4 = Expr(arr).var(axis=axis, stable=True).eview()
    assert_allclose3(res1, res2, res3, res4)


@given(make_arr((10, 10), unique=True), st.integers(0, 1))
def test_skew(arr, axis):
    res1 = pd.DataFrame(arr).skew(axis=axis).values
    res2 = tp.skew(arr, axis=axis)
    res3 = tp.skew(arr, axis=axis, stable=True)
    res4 = Expr(arr).skew(axis=axis, stable=True).eview()
    assert_allclose3(res1, res2, res3, res4)


@given(make_arr((10, 10), unique=True), st.integers(0, 1))
def test_kurt(arr, axis):
    res1 = pd.DataFrame(arr).kurt(axis=axis).values
    res2 = tp.kurt(arr, axis=axis)
    res3 = tp.kurt(arr, axis=axis, stable=True)
    res4 = Expr(arr).kurt(axis=axis, stable=True).eview()
    assert_allclose3(res1, res2, res3, res4)


@given(make_arr((10, 2)))
def test_cov(arr):
    cov_pd = pd.DataFrame(arr).cov().iloc[0, 1]
    arr1, arr2 = np.array_split(arr, 2, axis=1)
    cov1 = tp.cov(arr1, arr2, axis=0)[0]
    cov2 = tp.cov(arr1, arr2, axis=0, stable=True)[0]
    cov3 = Expr(arr1).cov(Expr(arr2), axis=0)[0].eview()
    cov4 = Expr(arr1).cov(Expr(arr2), axis=0, stable=True)[0].eview()
    assert_isclose3(cov_pd, cov1, cov2, cov3, cov4)


@given(make_arr((10, 2)))
def test_corr(arr):
    for method in ["pearson", "spearman"]:
        corr_pd = pd.DataFrame(arr).corr(method).iloc[0, 1]
        arr1, arr2 = np.array_split(arr, 2, axis=1)
        corr1 = tp.corr(arr1, arr2, axis=0, method=method)[0]
        corr2 = tp.corr(arr1, arr2, axis=0, method=method, stable=True)[0]
        corr3 = Expr(arr1).corr(Expr(arr2), axis=0, method=method)[0].eview()
        corr4 = (
            Expr(arr1).corr(Expr(arr2), axis=0, method=method, stable=True)[0].eview()
        )
        if np.isnan(corr_pd):
            assert np.isnan(corr1) & np.isnan(corr2) & np.isnan(corr3) & np.isnan(corr4)
        else:
            assert_isclose3(corr_pd, corr1, corr2, corr3, corr4)


def test_array_func_2d():
    arr = np.random.randn(20, 20)
    res1 = tp.rank(arr, axis=1)
    res2 = Expr(arr).rank(axis=1).eview()
    res3 = pd.DataFrame(arr).rank(axis=1).values
    assert_allclose3(res1, res2, res3)


def test_fillna():
    s = pd.Series([np.nan, 5, 6, 733, np.nan, 34, np.nan, np.nan])
    for method in ["ffill", "bfill"]:
        assert_allclose3(
            tp.fillna(s, method),
            Expr(s, copy=True).fillna(method).eview(),
            getattr(s, method)().values,
        )
        # test inplace
        s1 = s.copy()
        tp.fillna(s1, method, inplace=True)
        assert_allclose(s1, getattr(s, method)().values)

    # test fill value directly
    fill_value = 101
    assert_allclose3(
        tp.fillna(s, value=fill_value),
        Expr(s, copy=True).fillna(value=fill_value).eview(),
        s.fillna(fill_value).values,
    )
    # test ffill and bfill with value
    assert_allclose(
        tp.Expr(s.copy()).fillna("ffill", 0).eview(), [0, 5, 6, 733, 733, 34, 34, 34]
    )
    assert_allclose(
        tp.Expr(s.copy()).fillna("bfill", 0).eview(), [5, 5, 6, 733, 34, 34, 0, 0]
    )
    # test inplace
    tp.fillna(s, value=fill_value, inplace=True)
    assert_allclose(s, np.array([101, 5, 6, 733, 101, 34, 101, 101]))


test_fillna()


def test_clip():
    s = pd.Series([np.nan, 5, 6, 733, np.nan, 34, 456, np.nan])
    assert_allclose3(
        tp.clip(s, 5, 100), Expr(s, copy=True).clip(5, 100).eview(), s.clip(5, 100)
    )
    # test inplace
    s1 = s.copy()
    tp.clip(s1, 5, 100, inplace=True)
    assert_allclose(s1, s.clip(5, 100))


@given(make_arr((10, 2), nan_p=0.2), st.integers(0, 1))
def test_dropna(arr, axis):
    # test dropna 1d
    s = pd.Series([np.nan, 5, 6, 12, np.nan, 1, np.nan, np.nan])
    assert_allclose3(tp.dropna(s), Expr(s).dropna().eview(), s.dropna())
    # test dropna 2d
    assert_allclose(
        Expr(arr).dropna(axis=axis).eview(), pd.DataFrame(arr).dropna(axis=axis)
    )


def test_zscore():
    s = pd.Series(np.arange(12).astype(float))
    for stable in False, True:
        s1 = tp.zscore(s, stable)  # eager
        s2 = Expr(s, copy=True).zscore(stable).eview()  # lazy
        s3 = (s - s.mean()) / s.std()  # expect
        assert_allclose3(s1, s2, s3)
        # test inplace
        s_copy = s.copy()
        tp.zscore(s_copy, stable, inplace=True)
        assert_series_equal(s_copy, s3)


def test_where():
    a = np.random.randn(10, 10)
    b = tp.where(a > 0, 1, 0)
    assert_allclose(np.where(a > 0, 1, 0), b.eview())
    b = tp.where(a > 0, 1, a)
    assert_allclose(np.where(a > 0, 1, a), b.eview())
    b = tp.where(a > 0, a, 0)
    assert_allclose(np.where(a > 0, a, 0), b.eview())


def test_winsorize():
    s = pd.Series(np.arange(12).astype(float))

    # quantile method
    q = 0.05
    s1 = tp.winsorize(s, "quantile", q)
    s2 = Expr(s, copy=True).winsorize("quantile", q).eview()
    lower, upper = np.nanquantile(s, [q, 1 - q])
    s3 = s.clip(lower, upper)
    assert_allclose3(s1, s2, s3)
    # test quantile inplace
    s_copy = s.copy()
    tp.winsorize(s_copy, "quantile", q, inplace=True)
    assert_series_equal(s_copy, s3)

    # median method
    q = 1
    s1 = tp.winsorize(s, "median", q)
    s2 = Expr(s, copy=True).winsorize("median", q).eview()
    median = s.median()
    mad = (s - median).abs().median()
    s3 = s.clip(median - q * mad, median + q * mad)
    assert_allclose3(s1, s2, s3)

    # sigma method
    q = 1.2
    s1 = tp.winsorize(s, "sigma", q)
    s2 = Expr(s, copy=True).winsorize("sigma", q).eview()
    mean = s.mean()
    std = s.std()
    s3 = s.clip(mean - q * std, mean + q * std)
    assert_allclose3(s1, s2, s3)


@given(make_arr(20), st.integers(1, 10))
def test_split_group(arr, group):
    def split_group_py(x: pd.Series, group=10, method="average"):
        size = x.size - np.count_nonzero(np.isnan(x))
        out = np.ceil(x.rank(method=method) / (size / group))
        return out.fillna(0)

    s1 = tp.split_group(arr, group)
    s2 = Expr(arr).split_group(group).eview()
    arr = pd.Series(arr)
    s3 = split_group_py(arr, group)
    assert_allclose3(s1, s2, s3)
