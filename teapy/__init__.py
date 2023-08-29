from numpy import nan

from .array_func import *
from .datadict import DataDict, from_pd
from .expr import Expr, register
from .mod_func import *
from .teapy import (
    arange,
    calc_ret_single,
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
