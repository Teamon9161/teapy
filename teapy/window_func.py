from . import teapy as _tp
from .wrapper import wrap

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
_tp.ts_decay_linear = _tp.ts_wma
_tp.ts_mean = _tp.ts_sma
_tp.ts_ema = _tp.ts_ewm


@wrap("ts_func")
def ts_sum(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_sma(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_ewm(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_wma(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_prod(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_prod_mean(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_std(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_var(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_skew(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_kurt(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_max(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_min(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_argmax(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_argmin(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_stable(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_minmaxnorm(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_meanstdnorm(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_reg(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_tsf(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_reg_slope(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_reg_intercept(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_decay_linear(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_mean(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func")
def ts_ema(arr, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func2")
def ts_cov(arr1, arr2, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func2")
def ts_corr(arr1, arr2, window, axis=None, min_periods=1, stable=False, par=False):
    pass


@wrap("ts_func", use=True)
def ts_rank(arr, window, axis=None, min_periods=1, pct=False, stable=False, par=False):
    if not pct:
        return _tp.ts_rank(
            arr, window, axis=axis, min_periods=min_periods, stable=stable, par=par
        )
    else:
        return _tp.ts_rank_pct(
            arr, window, axis=axis, min_periods=min_periods, stable=stable, par=par
        )
