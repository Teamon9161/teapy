from functools import wraps

import numpy as np
import pandas as pd

from . import teapy as _tp

array_func_list = [
    "argsort",
]


def array_func_wrapper(func):
    @wraps(func)
    def _wrapper(arr, axis=None, par=False, **kwargs):
        if axis is not None:
            assert axis >= 0, "axis must equal to 0 or greater"
        if isinstance(arr, (list, tuple)):
            arr = np.asanyarray(arr)
            return func(arr, axis=axis, par=par, **kwargs)
        elif isinstance(arr, pd.Series):
            index, name = arr.index, arr.name
            arr = arr.values
            out = func(arr, axis=axis, par=par, **kwargs)
            return pd.Series(out, index=index, name=name, copy=False)
        elif isinstance(arr, pd.DataFrame):
            index, columns = arr.index, arr.columns
            arr = arr.values
            out = func(arr, axis=axis, par=par, **kwargs)
            return pd.DataFrame(out, index=index, columns=columns, copy=False)
        elif isinstance(arr, np.ndarray):
            return func(arr, axis=axis, par=par, **kwargs)
        else:
            raise TypeError(f"Not supported arr type, {type(arr)}")

    return _wrapper


for func_name in array_func_list:
    exec(f"{func_name} = array_func_wrapper(_tp.{func_name})")


@array_func_wrapper
def rank(arr, axis=None, pct=False, par=False):
    if not pct:
        return _tp.rank(arr, axis=axis, par=par)
    else:
        return _tp.rank_pct(arr, axis=axis, par=par)
