from functools import wraps

import numpy as np

from . import teapy as _tp
from .converter import Converter

__all__ = ["wrap"]


def array_func_wrapper(func):
    def _wrapper(arr, axis=None, stable=False, par=False, **kwargs):
        if axis is not None:
            assert axis >= 0, "axis must equal to 0 or greater"
        conv = Converter()
        out = func(conv(arr), axis=axis, stable=stable, par=par, **kwargs)
        res = conv(out, step="out")
        del conv
        return res

    return _wrapper


def array_agg_func_wrapper(func):
    def _wrapper(arr, axis=None, stable=False, par=False, keepdims=False, **kwargs):
        if axis is not None:
            assert axis >= 0, "axis must equal to 0 or greater"
        else:
            axis = 0
        conv = Converter()
        out = func(
            conv(arr), axis=axis, stable=stable, par=par, keepdims=keepdims, **kwargs
        )
        if not keepdims:
            out = np.squeeze(out, axis)
            # deal with special case
            if conv.otype == "pd.DataFrame":
                conv.otype = "pd.Series"
        res = conv(out, step="out")
        del conv
        return res

    return _wrapper


def array_agg_func2_wrapper(func):
    def _wrapper(
        arr1, arr2, axis=None, stable=False, par=False, keepdims=False, **kwargs
    ):
        if axis is not None:
            assert axis >= 0, "axis must equal to 0 or greater"
        else:
            axis = 0
        conv1, conv2 = Converter(), Converter()
        out = func(
            conv1(arr1),
            conv2(arr2),
            axis=axis,
            stable=stable,
            par=par,
            keepdims=keepdims,
            **kwargs,
        )
        # use arr1 to determine the dtype of output
        if not keepdims:
            out = np.squeeze(out, axis)
            # deal with special case
            if conv1.otype == "pd.DataFrame":
                conv1.otype = "pd.Series"
        res = conv1(out, step="out")
        del conv1, conv2
        return res

    return _wrapper


def rank_wrapper(func):
    def _wrapper(arr, axis=None, pct=False, par=False, **kwargs):
        if axis is not None:
            assert axis >= 0, "axis must equal to 0 or greater"
        conv = Converter()
        out = func(conv(arr), axis=axis, pct=pct, par=par, **kwargs)
        res = conv(out, step="out")
        del conv
        return res

    return _wrapper


def ts_func_wrapper(func):
    def _wrapper(
        arr, window, axis=None, min_periods=1, stable=False, par=False, **kwargs
    ):
        assert window > 0, f"window must be an integer greater than 0, not {window}"
        assert (
            min_periods >= 1
        ), f"min_periods must be an integer equal to 1 or greater, not{min_periods}"
        conv = Converter()
        out = func(
            conv(arr),
            window=window,
            axis=axis,
            min_periods=min_periods,
            stable=stable,
            par=par,
            **kwargs,
        )
        res = conv(out, step="out")
        del conv
        return res

    return _wrapper


def ts_func2_wrapper(func):
    def _wrapper(
        arr1, arr2, window, axis=None, min_periods=1, stable=False, par=False, **kwargs
    ):
        assert window > 0, f"window must be an integer greater than 0, not {window}"
        assert (
            min_periods >= 1
        ), f"min_periods must be an integer equal to 1 or greater, not{min_periods}"
        assert type(arr1) == type(
            arr2
        ), f"input array of {func.__name__} must have the same type"
        conv1, conv2 = Converter(), Converter()
        out = func(
            conv1(arr1),
            conv2(arr2),
            window=window,
            axis=axis,
            min_periods=min_periods,
            stable=stable,
            par=par,
            **kwargs,
        )
        res = conv1(out, step="out")
        del conv1, conv2
        return res

    return _wrapper


wrapper_dict = {
    "array_func": array_func_wrapper,
    "array_agg_func": array_agg_func_wrapper,
    "array_agg_func2": array_agg_func2_wrapper,
    "ts_func": ts_func_wrapper,
    "ts_func2": ts_func2_wrapper,
    "rank_func": rank_wrapper,
}


def wrap(func_type: str, use: bool = False):
    """
    Link the function to a function in _tp,

    Parameters
    ----------
    func_type: str
        each func_type map to a type of wrapper, so the func_type decide which wrapper
        shoulb be used to wrap this function.
    use: bool
        whether to use this function, by default, the function is only used to match
        the function in _tp, set use = True can force using this function

    Returns
    -------
    wrapped function.

    """
    global __all__
    wrapper = wrapper_dict[func_type]

    def _wrapfunc(func):
        __all__.append(func.__name__)
        if not use:
            return wraps(func)(wrapper(getattr(_tp, func.__name__)))
        else:
            return wraps(func)(wrapper(func))

    return _wrapfunc
