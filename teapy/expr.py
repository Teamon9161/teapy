from __future__ import annotations

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


@register
def rolling(self, window: str | int, by=None) -> ExprRolling:
    return ExprRolling(self, window, by=by)


class ExprRolling:
    def __init__(self, expr, window, by):
        self.expr = expr
        self.by = by
        self.window = window
        if "DateTime" in by.dtype:
            self.is_time = True

    def max(self):
        if self.is_time:
            idx = self.by._get_time_rolling_idx(self.window)
            return self.expr._rolling_select_max(idx)

    def min(self):
        if self.is_time:
            idx = self.by._get_time_rolling_idx(self.window)
            return self.expr._rolling_select_min(idx)

    def umax(self):
        if self.is_time:
            idx = self.by._get_time_rolling_idx(self.window)
            return self.expr._rolling_select_umax(idx)

    def umin(self):
        if self.is_time:
            idx = self.by._get_time_rolling_idx(self.window)
            return self.expr._rolling_select_umin(idx)

    def mean(self):
        if self.is_time:
            idx = self.by._get_time_rolling_idx(self.window)
            return self.expr._rolling_select_mean(idx)

    def std(self):
        if self.is_time:
            idx = self.by._get_time_rolling_idx(self.window)
            return self.expr._rolling_select_std(idx)
