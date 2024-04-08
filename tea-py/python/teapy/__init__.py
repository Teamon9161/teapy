from .array_func import *
from .expr import Expr, register
from .mod_func import *
from .py_datadict import DataDict, from_dataframe, from_pd, from_pl, scan_ipc
from .selector import Selector
from .tears import (
    arange,
    concat,
    context,
    eval_exprs,
    expr_register,
    get_version,
    nan,
    stack,
    timedelta,
    where_,
)
from .tears import calc_ret_single as _calc_ret_single
from .tears import calc_ret_single_with_spread as _calc_ret_single_with_spread
from .tears import full as _full
from .tears import parse_expr as asexpr
from .tears import parse_expr_list as asexprs
from .window_func import *

__version__ = get_version()

s = Selector


def eval(lazy_list):
    if not isinstance(lazy_list, (tuple, list)):
        lazy_list = [lazy_list]
    if len(lazy_list) == 0:
        return
    else:
        if isinstance(lazy_list[0], Expr):
            return eval_exprs(lazy_list, inplace=True)
        elif isinstance(lazy_list[0], DataDict):
            exprs = []
            for dd in lazy_list:
                exprs.extend(dd.exprs)
            return eval_exprs(exprs, inplace=True)
        else:
            msg = f"eval only accept list(Expr | DataDict), find {type(lazy_list[0])}"
            raise ValueError(msg)


def full(shape, fill_value=nan):
    if isinstance(shape, Selector):
        return Selector().mod_func("full")(shape, fill_value=fill_value)
    return _full(shape, fill_value)


def where(cond, x, y):
    if any(isinstance(i, Selector) for i in [cond, x, y]):
        return Selector().mod_func("where")(cond, x, y)
    return where_(cond, x, y)


def calc_ret_single(
    pos,
    opening_cost,
    closing_cost,
    init_cash,
    multiplier=1,
    leverage=1,
    slippage=0,
    ticksize=0,
    c_rate=3e-4,
    blowup=False,
    commision_type="percent",
    contract_change_signal=None,
):
    from numbers import Number

    if isinstance(slippage, Number):
        return _calc_ret_single(
            pos,
            opening_cost,
            closing_cost,
            init_cash,
            multiplier=multiplier,
            leverage=leverage,
            slippage=slippage,
            ticksize=ticksize,
            c_rate=c_rate,
            blowup=blowup,
            commision_type=commision_type,
            contract_change_signal=contract_change_signal,
        )
    else:
        return _calc_ret_single_with_spread(
            pos,
            opening_cost,
            closing_cost,
            spread=slippage,
            init_cash=init_cash,
            multiplier=multiplier,
            leverage=leverage,
            c_rate=c_rate,
            blowup=blowup,
            commision_type=commision_type,
            contract_change_signal=contract_change_signal,
        )
