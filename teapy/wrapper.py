from functools import wraps

from . import teapy as _tp
from .converter import Converter

__all__ = ["wrap"]


def inplace_wrapper(func):
    @wraps(func)
    def _wrapper(*args, inplace=False, **kwargs):
        func_name = f"{func.__name__}" + "_inplace" if inplace else f"{func.__name__}"
        return getattr(_tp, func_name)(*args, **kwargs)

    return _wrapper


def base_wrapper(func):
    @wraps(func)
    def _wrapper(arr, *args, **kwargs):
        conv = Converter()
        out = func(conv(arr), *args, **kwargs)
        res = conv(out, step="out")
        del conv
        return res

    return _wrapper


def array_func_wrapper(func):
    @wraps(func)
    def _wrapper(arr, *args, axis=0, **kwargs):
        assert axis >= 0, "axis must equal to 0 or greater"
        conv = Converter()
        out = func(conv(arr), *args, axis=axis, **kwargs)
        res = conv(out, step="out")
        del conv
        return res

    return _wrapper


def array_agg_func_wrapper(func):
    @wraps(func)
    def _wrapper(arr, *args, axis=0, **kwargs):
        assert axis >= 0, "axis must equal to 0 or greater"
        conv = Converter()
        out = func(conv(arr), *args, axis=axis, **kwargs)
        # deal with special case
        if conv.otype == "pd.DataFrame":
            conv.otype = "pd.Series"
        res = conv(out, step="out")
        del conv
        return res

    return _wrapper


def array_agg_func2_wrapper(func):
    @wraps(func)
    def _wrapper(arr1, arr2, *args, axis=0, **kwargs):
        assert axis >= 0, "axis must equal to 0 or greater"
        conv1, conv2 = Converter(), Converter()
        out = func(
            conv1(arr1),
            conv2(arr2),
            *args,
            axis=axis,
            **kwargs,
        )

        # deal with special case
        if conv1.otype == "pd.DataFrame":
            conv1.otype = "pd.Series"
        # use dtype of arr1 to determine the dtype of output
        res = conv1(out, step="out")
        del conv1, conv2
        return res

    return _wrapper


def ts_func_wrapper(func):
    @wraps(func)
    def _wrapper(arr, window, *args, min_periods=1, axis=0, **kwargs):
        assert axis >= 0, "axis must equal to 0 or greater"
        assert window > 0, f"window must be an integer greater than 0, not {window}"
        assert (
            min_periods >= 1
        ), f"min_periods must be an integer equal to 1 or greater, not{min_periods}"
        conv = Converter()
        out = func(
            conv(arr),
            window,
            *args,
            axis=axis,
            min_periods=min_periods,
            **kwargs,
        )
        res = conv(out, step="out")
        del conv
        return res

    return _wrapper


def ts_func2_wrapper(func):
    @wraps(func)
    def _wrapper(arr1, arr2, window, *args, min_periods=1, axis=0, **kwargs):
        assert axis >= 0, "axis must equal to 0 or greater"
        assert window > 0, f"window must be an integer greater than 0, not {window}"
        assert (
            min_periods >= 1
        ), f"min_periods must be an integer equal to 1 or greater, not{min_periods}"
        # assert type(arr1) == type(
        #     arr2
        # ), f"input array of {func.__name__} must have the same type"
        conv1, conv2 = Converter(), Converter()
        out = func(
            conv1(arr1),
            conv2(arr2),
            window,
            *args,
            axis=axis,
            min_periods=min_periods,
            **kwargs,
        )
        res = conv1(out, step="out")
        del conv1, conv2
        return res

    return _wrapper


wrapper_dict = {
    "base": base_wrapper,
    "array_func": array_func_wrapper,
    "array_agg_func": array_agg_func_wrapper,
    "array_agg_func2": array_agg_func2_wrapper,
    "ts_func": ts_func_wrapper,
    "ts_func2": ts_func2_wrapper,
}


def wrap(func_type: str, use: bool = False, inplace=False):
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
    inplace: bool
        if `inplace` is True, There must be two functions in `_tp`, one is a inplace
        function with a "_inplace" suffix, while the other function return a new array.

    Returns
    -------
    wrapped function.

    """
    wrapper = wrapper_dict[func_type]

    def _wrapfunc(func):
        try:
            func = getattr(_tp, func.__name__) if not use else func
            func = inplace_wrapper(func) if inplace else func
            wrapper_func = wrapper(func)
            return wraps(wrapper_func)(wrapper_func)
        except AttributeError:
            return wrapper(lambda x: x)  # if src doesn't have feature: eager_api

    return _wrapfunc
