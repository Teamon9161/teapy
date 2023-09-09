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
    return self.select(idx, check=False)


@register
def left_join(self, right, left_other=None):
    idx = self._get_left_join_idx(left_other=left_other, right=right)
    if isinstance(right, (tuple, list)):
        return [r.select(idx, check=False) for r in right]
    else:
        return right.select(idx, check=False)


@register
def rolling(
    self,
    window: str | int = None,
    by=None,
    idx=None,
    offset=None,
    start_by="full",
    others=None,
) -> ExprRolling:
    """
    window: duration, such as 1y, 2mo, 3d, 4h, 5m, 6s, or combine them
    start_by: full | duration_start, only valid if window is a duration
    others: only available in rolling.apply, this can add other expressions into the rolling context
    if offset is not None, start_by will be ignored
    """
    if window is None and idx is None:
        raise ValueError("window or idx must be specified")
    elif window is not None and idx is not None:
        raise ValueError("window and idx cannot be specified at the same time")
    return ExprRolling(
        self,
        window=window,
        by=by,
        idx=idx,
        offset=offset,
        start_by=start_by,
        others=others,
    )


class ExprRolling:
    name_prefix = "column_"

    def __init__(
        self, expr, window, by, idx=None, offset=None, others=None, start_by="full"
    ):
        self.expr = expr
        self.by = by
        self.window = window
        self.idx = idx
        self.offset = offset
        self.start_by = start_by
        self.others = others
        self.name_auto = 0
        if by is not None and "DateTime" in by.dtype:
            self.is_time = True

    def get_idx(self):
        if self.idx is None:
            # if self.is_time:
            if self.offset is None:
                if isinstance(self.window, int):
                    return self.expr._get_fix_window_rolling_idx(self.window)
                else:
                    return self.by._get_time_rolling_idx(
                        self.window, start_by=self.start_by
                    )
            else:
                return self.by._get_time_rolling_offset_idx(self.window, self.offset)
        else:
            return self.idx

    def __get_name_auto(self):
        name = self.name_prefix + str(self.name_auto)
        self.name_auto += 1
        return name

    def __getattr__(self, name):
        idx = self.get_idx()

        def wrap_func():
            if idx.dtype != "Vec<Usize>":
                return getattr(self.expr, f"_rolling_select_{name}")(idx)
            else:
                return getattr(self.expr, f"_rolling_select_by_vecusize_{name}")(idx)

        return wrap_func

    def apply(self, agg_expr):
        idx = self.get_idx()
        if self.expr.name is None:
            self.expr.alias(self.__get_name_auto(), inplace=True)
            self.name_auto
        if self.others is not None:
            for e in self.others:
                if e.name is None:
                    e.alias(self.__get_name_auto(), inplace=True)
        if isinstance(agg_expr, (list, tuple)):
            return [
                self.expr.rolling_apply_with_start(
                    ae, roll_start=idx, others=self.others
                )
                for ae in agg_expr
            ]
        else:
            return self.expr.rolling_apply_with_start(
                agg_expr, roll_start=idx, others=self.others
            )
