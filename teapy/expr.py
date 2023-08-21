from .teapy import PyExpr as Expr
from .teapy import expr_register as _expr_register

Expr.where = Expr.where_


def register(f):
    """expr function register"""
    _expr_register(f.__name__, f)
    return f


@register
def unique(self, others=None, keep="first"):
    if isinstance(others, (str, int)):
        others = [others]
    idx = self._get_unique_idx(others, keep)
    return self._select_unchecked(idx)


@register
def left_join(self, right, left_other=None):
    idx = self._get_left_join_idx(left_other=left_other, right=right)
    if isinstance(right, (tuple, list)):
        return [r._select_unchecked(idx) for r in right]
    else:
        return right._select_unchecked(idx)
