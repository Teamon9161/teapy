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
def rolling(
    self, window: str | int = None, by=None, idx=None, offset=None, start_by="full"
) -> ExprRolling:
    """if offset is not None, start_by will be ignored"""
    if window is None and idx is None:
        raise ValueError("window or idx must be specified")
    elif window is not None and idx is not None:
        raise ValueError("window and idx cannot be specified at the same time")
    return ExprRolling(
        self, window=window, by=by, idx=idx, offset=offset, start_by=start_by
    )


class ExprRolling:
    def __init__(self, expr, window, by, idx=None, offset=None, start_by="full"):
        self.expr = expr
        self.by = by
        self.window = window
        self.idx = idx
        self.offset = offset
        self.start_by = start_by
        if by is not None and "DateTime" in by.dtype:
            self.is_time = True

    def get_idx(self):
        if self.idx is None:
            # if self.is_time:
            if self.offset is None:
                return self.by._get_time_rolling_idx(
                    self.window, start_by=self.start_by
                )
            else:
                return self.by._get_time_rolling_offset_idx(self.window, self.offset)
        else:
            return self.idx

    def __getattr__(self, name):
        idx = self.get_idx()

        def wrap_func():
            if idx.dtype != "Vec<Usize>":
                return getattr(self.expr, f"_rolling_select_{name}")(idx)
            else:
                return getattr(self.expr, f"_rolling_select_by_vecusize_{name}")(idx)

        return wrap_func
