from . import tears as _tp
from .wrapper import impl_by_lazy

__all__ = [
    "ts_sum",
    "ts_sma",
    "ts_ewm",
    "ts_wma",
    "ts_prod",
    "ts_prod_mean",
    "ts_std",
    "ts_var",
    "ts_skew",
    "ts_kurt",
    "ts_max",
    "ts_argmax",
    "ts_min",
    "ts_argmin",
    "ts_stable",
    "ts_minmaxnorm",
    "ts_meanstdnorm",
    "ts_reg",
    "ts_tsf",
    "ts_reg_slope",
    "ts_reg_intercept",
    "ts_cov",
    "ts_corr",
    "ts_rank",
    "ts_decay_linear",
    "ts_mean",
    "ts_ema",
]

# alias
try:
    _tp.ts_decay_linear = _tp.ts_wma
    _tp.ts_mean = _tp.ts_sma
    _tp.ts_ema = _tp.ts_ewm
except AttributeError:
    pass


@impl_by_lazy()
def ts_sum(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_sma(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_ewm(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_wma(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_prod(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_prod_mean(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_std(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_var(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_skew(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_kurt(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_max(arr, window, min_periods=1, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_min(arr, window, min_periods=1, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_argmax(arr, window, min_periods=1, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_argmin(arr, window, min_periods=1, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_stable(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_minmaxnorm(arr, window, min_periods=1, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_meanstdnorm(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_reg(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_tsf(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_reg_slope(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_reg_intercept(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_decay_linear(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_mean(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_ema(arr, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy("default2")
def ts_cov(arr1, arr2, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy("default2")
def ts_corr(arr1, arr2, window, min_periods=1, stable=False, axis=None, par=False):
    pass


@impl_by_lazy()
def ts_rank(arr, window, min_periods=1, pct=False, axis=None, par=False):
    pass
