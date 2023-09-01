from numpy import nan

from .array_func import *
from .datadict import DataDict, from_pd
from .expr import Expr, register
from .mod_func import *
from .teapy import arange
from .teapy import calc_ret_single as _calc_ret_single
from .teapy import calc_ret_single_with_spread as _calc_ret_single_with_spread
from .teapy import (
    concat,
    context,
    eval_dicts,
    eval_exprs,
    expr_register,
    full,
    get_version,
)
from .teapy import parse_expr as asexpr
from .teapy import parse_expr_list as asexprs
from .teapy import stack, timedelta
from .teapy import where_ as where
from .window_func import *

__version__ = get_version()


def eval(lazy_list):
    if len(lazy_list) == 0:
        return
    else:
        if isinstance(lazy_list[0], Expr):
            return eval_exprs(lazy_list, inplace=True)
        elif isinstance(lazy_list[0], DataDict):
            return eval_dicts(lazy_list, inplace=True)
        else:
            raise ValueError("eval() only accept list of Expr or DataDict")


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
