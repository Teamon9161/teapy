from functools import wraps

import numpy as np
import pandas as pd

from . import teapy as _tp

# window func of one input array
ts_func_list = [
    # feature
    "ts_sum",
    "ts_sma",
    "ts_ewm",
    "ts_wma",
    "ts_prod",
    "ts_prod_mean",
    "ts_std",
    "ts_skew",
    "ts_kurt",
    # compare
    "ts_max",
    "ts_min",
    "ts_argmax",
    "ts_argmin",
    # norm
    "ts_stable",
    "ts_minmaxnorm",
    "ts_meanstdnorm",
    # reg
    "ts_reg",
    "ts_tsf",
    "ts_reg_slope",
    "ts_reg_intercept",
    # alias
    "ts_decay_linear",
    "ts_mean",
    "ts_ema",
]

# window func of two input arrays
ts_func2_list = ["ts_cov", "ts_corr"]

# alias
_tp.ts_decay_linear = _tp.ts_wma
_tp.ts_mean = _tp.ts_sma
_tp.ts_ema = _tp.ts_ewm


def ts_func_wrapper(func):
    @wraps(func)
    def _wrapper(arr, window, axis=None, min_periods=1, par=False, **kwargs):
        assert window > 0, f"window must be an integer greater than 0, not {window}"
        assert (
            min_periods >= 1
        ), f"min_periods must be an integer equal to 1 or greater, not{min_periods}"
        if isinstance(arr, (list, tuple)):
            arr = np.asanyarray(arr)
            return func(
                arr,
                window=window,
                axis=axis,
                min_periods=min_periods,
                par=par,
                **kwargs,
            )
        elif isinstance(arr, pd.Series):
            index, name = arr.index, arr.name
            arr = arr.values
            out = func(
                arr,
                window=window,
                axis=axis,
                min_periods=min_periods,
                par=par,
                **kwargs,
            )
            return pd.Series(out, index=index, name=name, copy=False)
        elif isinstance(arr, pd.DataFrame):
            index, columns = arr.index, arr.columns
            arr = arr.values
            out = func(
                arr,
                window=window,
                axis=axis,
                min_periods=min_periods,
                par=par,
                **kwargs,
            )
            return pd.DataFrame(out, index=index, columns=columns, copy=False)
        elif isinstance(arr, np.ndarray):
            return func(
                arr,
                window=window,
                axis=axis,
                min_periods=min_periods,
                par=par,
                **kwargs,
            )
        else:
            raise TypeError(f"Not supported arr type, {type(arr)}")

    return _wrapper


def ts_func2_wrapper(func):
    @wraps(func)
    def _wrapper(arr1, arr2, window, axis=None, min_periods=1, par=False):
        assert window > 0, f"window must be an integer greater than 0, not {window}"
        assert (
            min_periods >= 1
        ), f"min_periods must be an integer equal to 1 or greater, not{min_periods}"
        assert type(arr1) == type(
            arr2
        ), f"input array of {func.__name__} must have the same type"
        if isinstance(arr1, (list, tuple)):
            arr1, arr2 = np.asanyarray(arr1), np.asanyarray(arr2)
            return func(
                arr1, arr2, window=window, axis=axis, min_periods=min_periods, par=par
            )
        elif isinstance(arr1, pd.Series):
            index, name = arr1.index, arr1.name
            arr1, arr2 = arr1.values, arr2.values
            out = func(
                arr1, arr2, window=window, axis=axis, min_periods=min_periods, par=par
            )
            return pd.Series(out, index=index, name=name, copy=False)
        elif isinstance(arr1, pd.DataFrame):
            index, columns = arr1.index, arr1.columns
            out = func(
                arr1.values,
                arr2.values,
                window=window,
                axis=axis,
                min_periods=min_periods,
                par=par,
            )
            return pd.DataFrame(out, index=index, columns=columns, copy=False)
        elif isinstance(arr1, np.ndarray):
            return func(
                arr1, arr2, window=window, axis=axis, min_periods=min_periods, par=par
            )
        else:
            raise TypeError(f"Not supported arr type, {type(arr1)}")

    return _wrapper


# define window func of one input array
for func_name in ts_func_list:
    exec(f"{func_name} = ts_func_wrapper(_tp.{func_name})")

# define window func of two input arrays
for func_name in ts_func2_list:
    exec(f"{func_name} = ts_func2_wrapper(_tp.{func_name})")


@ts_func_wrapper
def ts_rank(arr, window, axis=None, min_periods=1, pct=False, par=False):
    if not pct:
        return _tp.ts_rank(arr, window, axis=axis, min_periods=min_periods, par=par)
    else:
        return _tp.ts_rank_pct(arr, window, axis=axis, min_periods=min_periods, par=par)
