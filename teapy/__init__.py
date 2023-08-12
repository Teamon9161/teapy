from .array_func import *
from .datadict import DataDict, from_pd
from .teapy import PyExpr as Expr
from .teapy import arange, calc_ret_single, concat, eval, full, get_version
from .teapy import parse_expr as asexpr
from .teapy import parse_expr_list as asexprs
from .teapy import stack, timedelta
from .teapy import where_ as where
from .teapy import expr_register
from .window_func import *

Expr.where = Expr.where_

# expr register
def register(f):
    expr_register(f.__qualname__, f)
    return f

__version__ = get_version()
