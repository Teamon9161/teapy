from warnings import warn

import numpy as np

from . import teapy as _tp
from .wrapper import wrap

__all__ = [
    "sum",
    "min",
    "max",
    "mean",
    "median",
    "quantile",
    "std",
    "var",
    "skew",
    "kurt",
    "count_nan",
    "count_notnan",
    "argsort",
    "rank",
    "cov",
    "corr",
    "fillna",
    "zscore",
    "winsorize",
    "remove_nan",
    "split_group",
    "clip",
]


@wrap("array_agg_func")
def sum(arr, stable=False, axis=0, par=False):
    """
    Sum of array elements in a given axis.

    Parameters
    ----------
    arr : np.ndarray, pd.Series, pd.DataFrame, tuple, list
        Elements to sum.
    stable : bool
        whether to use Kahan summation to reduce the numerical error
    axis : int or None
        axis along which a sum is performed.
    par : bool
        whether to parallelize.

    Returns
    -------
    array_like.

    """


@wrap("array_agg_func")
def min(arr, axis=0, par=False):
    pass


@wrap("array_agg_func")
def max(arr, axis=0, par=False):
    pass


@wrap("array_agg_func")
def mean(arr, stable=False, axis=0, par=False):
    pass


@wrap("array_agg_func")
def median(arr, axis=0, par=False):
    pass


@wrap("array_agg_func")
def quantile(arr, q, method="Linear", axis=0, par=False):
    pass


@wrap("array_agg_func")
def std(arr, stable=False, axis=0, par=False):
    pass


@wrap("array_agg_func")
def var(arr, stable=False, axis=0, par=False):
    pass


@wrap("array_agg_func")
def skew(arr, stable=False, axis=0, par=False):
    pass


@wrap("array_agg_func")
def kurt(arr, stable=False, axis=0, par=False):
    pass


@wrap("array_agg_func")
def count_nan(arr, axis=0, par=False):
    pass


@wrap("array_agg_func")
def count_notnan(arr, axis=0, par=False):
    pass


@wrap("array_func")
def argsort(arr, axis=0, par=False, rev=False):
    pass


@wrap("array_agg_func2")
def cov(arr1, arr2, stable=False, axis=0, par=False):
    pass


@wrap("array_agg_func2")
def corr(arr1, arr2, method=None, stable=False, axis=0, par=False):
    pass


@wrap("array_func")
def rank(arr, pct=False, axis=0, par=False, rev=False):
    pass


@wrap("array_func", inplace=True)
def clip(arr, min, max, axis=0, par=False):
    pass


@wrap("base")
def remove_nan(arr):
    pass


@wrap("array_func")
def split_group(arr, axis=0, par=False):
    pass


@wrap("array_agg_func", inplace=True)
def fillna(arr, method=None, value=None, axis=0, par=False, inplace=False):
    pass


@wrap("array_func", use=True)
def zscore(arr, stable=False, axis=0, par=False, inplace=False):
    if inplace:
        if arr.dtype not in [np.float64, np.float32, np.float16]:
            warn(
                "the dtype of arr is not float, so note that the result is not float too"
            )
        return _tp.zscore_inplace(arr, stable, axis=axis, par=par)
    else:
        return _tp.zscore(arr, stable, axis=axis, par=par)


@wrap("array_func", use=True)
def winsorize(
    arr, method=None, method_params=None, stable=False, axis=0, par=False, inplace=False
):
    """
    Perform winsorization on a given axis. Winsorization is the process of replacing
    the extreme values of statistical data in order to limit the effect of the outliers
    on the calculations or the results obtained by using that data.

    Parameters
    ----------
    arr : np.ndarray, pd.Series, pd.DataFrame, tuple, list
        Elements to sum.
    method : quantile | median | sigma, default is quantile
        quantile: if method_params is 1%, then all elements greater than the
            99% quantile will be set to the 99% quantile, and all elements less
            than the 1% quantile will be set to the 1% quantile.

        median: if method_params is 3, calculate median value at first, and then
            calculate MAD. MAD is the median of the `|v - median|` array where v is the
            element of the array. All elements greater than `median + 3 * MAD` will be
            set to `median + 3 * MAD`, and all elements less than `median - 3 * MAD` will
            be set to `median - 3 * MAD` by default.

        sigma: if method_params is 3, calulate the mean and standard deviation of the
            array, all elements greater than `mean + 3 * std` will be set `mean + 3 * std`,
            and all elements less than `mean - 3 * std` will be set to `mean - 3 * std`.
    method_params : float64
        if method is quantile: the default is 1%.
        if method is median: the default is 3.
        if method is sigma: the default is 3.
    stable :  bool
        whether to use Kahan summation to reduce the numerical error
    axis : int or None
        axis along which a winsorize is performed.
    par : bool
        whether to parallelize.
    inplace : bool
        Whether to modify the data rather than creating a new one.

    Returns
    -------
    None if `inplace` is true else array_like.

    """
    if arr.dtype not in [np.float64, np.float32, np.float16]:
        warn("the dtype of arr is not float, so note that the result is not float too")
    if inplace:
        return _tp.winsorize_inplace(
            arr, method, method_params, stable, axis=axis, par=par
        )
    else:
        return _tp.winsorize(arr, method, method_params, stable, axis=axis, par=par)
