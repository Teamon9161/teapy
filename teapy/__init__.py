from .array_func import *
from .datadict import DataDict, from_pd
from .teapy import PyExpr as Expr
from .teapy import (
    arange,
    calc_ret_single,
    concat,
    eval,
    expr_register,
    full,
    get_version,
)
from .teapy import parse_expr as asexpr
from .teapy import parse_expr_list as asexprs
from .teapy import stack, timedelta
from .teapy import where_ as where
from .window_func import *

Expr.where = Expr.where_


def register(f):
    """expr function register"""
    expr_register(f.__name__, f)
    return f


__version__ = get_version()
