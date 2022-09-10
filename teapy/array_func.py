from .wrapper import wrap

__all__ = [
    "sum",
    "min",
    "max",
    "mean",
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
]


@wrap("array_agg_func")
def sum(arr, axis=None, stable=False, par=False, keepdims=False):
    """
    Sum of array elements in a given axis.

    Parameters
    ----------
    arr : np.ndarray, pd.Series, pd.DataFrame, tuple, list
        Elements to sum.
    axis : int or None
        axis along which a sum is performed.
    stable : bool
        whether to use Kahan summation to reduce the numerical error
    par : bool
        whether to parallelize.
    keepdims : bool
        whether to keep the shape of the array.

    Returns
    -------
    array_like.

    """


@wrap("array_agg_func")
def min(arr, axis=None, stable=False, par=False, keepdims=False):
    pass


@wrap("array_agg_func")
def max(arr, axis=None, stable=False, par=False, keepdims=False):
    pass


@wrap("array_agg_func")
def mean(arr, axis=None, stable=False, par=False, keepdims=False):
    pass


@wrap("array_agg_func")
def std(arr, axis=None, stable=False, par=False, keepdims=False):
    pass


@wrap("array_agg_func")
def var(arr, axis=None, stable=False, par=False, keepdims=False):
    pass


@wrap("array_agg_func")
def skew(arr, axis=None, stable=False, par=False, keepdims=False):
    pass


@wrap("array_agg_func")
def kurt(arr, axis=None, stable=False, par=False, keepdims=False):
    pass


@wrap("array_agg_func")
def count_nan(arr, axis=None, stable=False, par=False, keepdims=False):
    pass


@wrap("array_agg_func")
def count_notnan(arr, axis=None, stable=False, par=False, keepdims=False):
    pass


@wrap("array_func")
def argsort(arr, axis=None, stable=False, par=False):
    pass


@wrap("array_agg_func2")
def cov(arr, axis=None, stable=False, par=False):
    pass


@wrap("array_agg_func2")
def corr(arr1, arr2, axis=None, stable=False, par=False, keepdims=False):
    pass


@wrap("rank_func")
def rank(arr, axis=None, pct=False, par=False):
    pass
