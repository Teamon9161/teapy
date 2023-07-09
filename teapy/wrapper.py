from functools import wraps

from . import teapy as _tp


def default_wrapper(func):
    @wraps(func)
    def _wrapper(arr, *args, **kwargs):
        func_name = f"{func.__name__}"
        return getattr(_tp.PyExpr(arr), func_name)(*args, **kwargs).value()

    return _wrapper


def default2_wrapper(func):
    @wraps(func)
    def _wrapper(arr1, arr2, *args, **kwargs):
        func_name = f"{func.__name__}"
        return getattr(_tp.PyExpr(arr1), func_name)(
            _tp.PyExpr(arr2), *args, **kwargs
        ).value()

    return _wrapper


def inplace_wrapper(func):
    @wraps(func)
    def _wrapper(arr, *args, inplace=False, **kwargs):
        func_name = f"{func.__name__}"
        if inplace:
            getattr(_tp.PyExpr(arr), func_name)(*args, **kwargs).eval()
            return
        else:
            return getattr(_tp.PyExpr(arr, copy=True), func_name)(
                *args, **kwargs
            ).value()

    return _wrapper


def impl_by_lazy(func_type: str = "default"):
    if func_type == "default":
        return default_wrapper
    elif func_type == "default2":
        return default2_wrapper
    elif func_type == "inplace":
        return inplace_wrapper
    else:
        raise ValueError("Not support func_type: %s" % func_type)
