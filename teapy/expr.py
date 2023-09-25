from __future__ import annotations

from .datadict import name_prefix
from .teapy import PyExpr as Expr
from .teapy import expr_register as _expr_register
from .teapy import parse_expr_list as asexprs

Expr.where = Expr.where_


def register(f):
    """expr function register"""
    _expr_register(f.__name__, f)
    return f


@register
def eval_in(self, context=None, inplace=False, freeze=True):
    return self.eval(inplace=inplace, context=context, freeze=freeze)


@register
def eview(self, context=None, freeze=True):
    if freeze:
        return self.eval(False, context=context, freeze=True).view
    else:
        self.eval(True, context=context, freeze=False)
        return self.view_in(context)


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


def mask_to_idx(self):
    from .teapy import arange

    idx = arange(self.shape[0])
    return idx.filter(self)


@register
def rolling(
    self,
    window: str | int = None,
    time_expr=None,
    idx=None,
    offset=None,
    start_by="full",
    others=None,
    by=None,
) -> ExprRolling:
    """
    window: int | duration, such as 1y, 2mo, 3d, 4h, 5m, 6s, or combine them
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
        time_expr=time_expr,
        idx=idx,
        offset=offset,
        start_by=start_by,
        others=others,
        by=by,
    )


@register
def groupby(
    self, by=None, time_expr=None, info=None, others=None, closed="left"
) -> ExprGroupBy:
    """
    window: int | duration, such as 1y, 2mo, 3d, 4h, 5m, 6s, or combine them
    others: only available in rolling.apply, this can add other expressions into the rolling context
    """
    if by is None and info is None:
        raise ValueError("by or info must be specified")
    elif by is not None and info is not None:
        raise ValueError("by and info cannot be specified at the same time")
    return ExprGroupBy(
        self,
        by=by,
        time_expr=time_expr,
        info=info,
        others=others,
        closed=closed,
    )


class ExprRollMixin:
    def __init__(self, expr, window, time_expr=None, others=None, by=None):
        self.expr = expr
        self.time_expr = time_expr
        if by is not None:
            import warnings

            warnings.warn(
                "by will be deprecated in future release, please use time_expr instead"
            )
            self.time_expr = by

        self.window = window
        self.others = asexprs(others, copy=False) if others is not None else None
        self.name_auto = 0
        self.prepare()

    def __get_name_auto(self):
        name = name_prefix + str(self.name_auto)
        self.name_auto += 1
        return name

    def prepare(self):
        if self.expr.name is None:
            self.expr.alias(self.__get_name_auto(), inplace=True)
        if self.others is not None:
            if isinstance(self.others, (list, tuple)):
                for e in self.others:
                    if e.name is None:
                        e.alias(self.__get_name_auto(), inplace=True)
            else:
                if self.others.name is None:
                    self.others.alias(self.__get_name_auto(), inplace=True)
        return self.expr, self.others


class ExprRolling(ExprRollMixin):
    def __init__(
        self,
        expr,
        window,
        time_expr=None,
        idx=None,
        offset=None,
        others=None,
        start_by="full",
        by=None,
    ):
        super().__init__(expr, window, time_expr=time_expr, others=others, by=by)
        self.idx = idx
        self.offset = offset
        self.start_by = start_by
        self.idx = self.get_idx()

    def get_idx(self):
        if self.idx is None:
            # if self.is_time:
            if self.offset is None:
                if isinstance(self.window, int):
                    return self.expr._get_fix_window_rolling_idx(self.window)
                else:
                    return self.time_expr._get_time_rolling_idx(
                        self.window, start_by=self.start_by
                    )
            else:
                return self.time_expr._get_time_rolling_offset_idx(
                    self.window, self.offset
                )
        else:
            return self.idx

    def __getattr__(self, name):
        idx = self.idx

        def wrap_func():
            if idx.dtype != "Vec<Usize>":
                return getattr(self.expr, f"_rolling_select_{name}")(idx)
            else:
                return getattr(self.expr, f"_rolling_select_by_vecusize_{name}")(idx)

        return wrap_func

    def apply(self, agg_expr):
        idx = self.idx
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


class ExprGroupBy(ExprRollMixin):
    def __init__(
        self, expr, by=None, time_expr=None, info=None, others=None, closed="left"
    ) -> None:
        self.closed = closed
        self.info = info
        super().__init__(expr, by, time_expr=time_expr, others=others)
        self.info = self.get_info()

    def get_info(self):
        if self.info is None:
            if isinstance(self.window, int):
                raise NotImplementedError
            else:
                return self.time_expr._get_time_groupby_info(
                    self.window, closed=self.closed, split=True
                )
        else:
            return self.info

    def __getattr__(self, name):
        def wrap_func(*args, **kwargs):
            return getattr(self.expr, f"_group_by_time_{name}")(
                self.info[1], *args, **kwargs
            )

        return wrap_func

    def agg(self, agg_expr):
        label, start_idx = self.info
        if isinstance(agg_expr, (list, tuple)):
            return label, [
                self.expr.group_by_time(ae, group_info=start_idx, others=self.others)
                for ae in agg_expr
            ]
        else:
            return label, self.expr.group_by_time(
                agg_expr, group_info=start_idx, others=self.others
            )
