from .wrapper import impl_by_lazy


@impl_by_lazy()
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


@impl_by_lazy()
def prod(arr, axis=0, par=False):
    pass


@impl_by_lazy()
def cumsum(arr, stable=False, axis=0, par=False):
    pass


@impl_by_lazy()
def cumprod(arr, axis=0, par=False):
    pass


@impl_by_lazy()
def min(arr, axis=0, par=False):
    pass


@impl_by_lazy()
def max(arr, axis=0, par=False):
    pass


@impl_by_lazy()
def mean(arr, stable=False, axis=0, par=False):
    pass


@impl_by_lazy()
def median(arr, axis=0, par=False):
    pass


@impl_by_lazy()
def quantile(arr, q, method="Linear", axis=0, par=False):
    pass


@impl_by_lazy()
def std(arr, stable=False, axis=0, par=False):
    pass


@impl_by_lazy()
def var(arr, stable=False, axis=0, par=False):
    pass


@impl_by_lazy()
def skew(arr, stable=False, axis=0, par=False):
    pass


@impl_by_lazy()
def kurt(arr, stable=False, axis=0, par=False):
    pass


@impl_by_lazy()
def count_nan(arr, axis=0, par=False):
    pass


@impl_by_lazy()
def count_notnan(arr, axis=0, par=False):
    pass


@impl_by_lazy()
def argsort(arr, axis=0, par=False, rev=False):
    pass


@impl_by_lazy("default2")
def cov(arr1, arr2, stable=False, axis=0, par=False):
    pass


@impl_by_lazy("default2")
def corr(arr1, arr2, method=None, stable=False, axis=0, par=False):
    pass


@impl_by_lazy()
def rank(arr, pct=False, axis=0, par=False, rev=False):
    pass


@impl_by_lazy("inplace")
def clip(arr, min, max, axis=0, par=False, inplace=False):
    pass


@impl_by_lazy()
def dropna(arr, axis=0, how="any", par=False):
    pass


@impl_by_lazy()
def split_group(arr, axis=0, par=False):
    pass


@impl_by_lazy("inplace")
def fillna(arr, method="vfill", value=None, axis=0, par=False, inplace=False):
    pass


@impl_by_lazy("inplace")
def zscore(arr, stable=False, axis=0, par=False, inplace=False, warning=True):
    pass


@impl_by_lazy("inplace")
def winsorize(
    arr,
    method=None,
    method_params=None,
    stable=False,
    axis=0,
    par=False,
    inplace=False,
    warning=True,
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
            set to `median + 3 * MAD`, and all elements less than
            `median - 3 * MAD` will be set to `median - 3 * MAD` by default.

        sigma: if method_params is 3, calulate the mean and standard deviation of the
            array, all elements greater than `mean + 3 * std` will be set
            `mean + 3 * std`, and all elements less than `mean - 3 * std`
            will be set to `mean - 3 * std`.
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
    warning: bool
        Whether to warn when the dtype of input is not float.


    Returns
    -------
    None if `inplace` is true else array_like.

    """


@impl_by_lazy()
def shift(arr, n=1, fill=None, axis=0, par=False):
    pass
