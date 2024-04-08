from __future__ import annotations

from typing import Optional

from .py_datadict import name_prefix
from .selector import selector_to_expr
from .tears import Expr
from .tears import expr_register as _expr_register
from .tears import parse_expr_list as asexprs

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


@register
def __getstate__(self):
    return {"arr": self.value(), "name": self.name}


@register
def __setstate__(self, state):
    return Expr(state["arr"], state.get("name"))


@register
def mask_to_idx(self):
    from .tears import arange

    idx = arange(self.shape[0])
    return idx.filter(self)


@register
def rolling(
    self,
    window: str | int | None = None,
    time_expr=None,
    idx=None,
    offset=None,
    start_by="full",
    others: Expr | list(Expr) | None = None,
    by=None,
    type_=None,
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
    if idx is not None and type_ is None:
        type_ = "start"  # default
        # raise ValueError("type_ must be specified if idx is not None")
    return ExprRolling(
        self,
        window=window,
        time_expr=time_expr,
        idx=idx,
        offset=offset,
        start_by=start_by,
        others=others,
        by=by,
        type_=type_,
    )


@register
def groupby(
    self,
    by=None,
    time_expr=None,
    idxs=None,
    others=None,
    closed="left",
    sort=True,
    par=False,
    type_=None,
) -> ExprGroupBy:
    """
    window: int | duration, such as 1y, 2mo, 3d, 4h, 5m, 6s, or combine them
    others: only available in rolling.apply, this can add other expressions into the rolling context
    """
    if by is None and idxs is None:
        raise ValueError("by or info must be specified")
    elif by is not None and idxs is not None:
        raise ValueError("by and info cannot be specified at the same time")
    if idxs is not None and type_ is None:
        raise ValueError("type_ must be specified if info is not None")
    return ExprGroupBy(
        self,
        by=by,
        time_expr=time_expr,
        idxs=idxs,
        others=others,
        closed=closed,
        sort=sort,
        par=par,
        type_=type_,
    )


class ExprRollMixin:
    def __init__(self, expr, window, time_expr=None, others=None, by=None):
        self.expr = expr
        self.time_expr = time_expr
        if by is not None:
            import warnings

            warnings.warn(
                "by will be deprecated in future release, please use time_expr instead",
                stacklevel=2,
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
        type_=None,
    ):
        super().__init__(expr, window, time_expr=time_expr, others=others, by=by)
        self.type = type_
        self.idx = idx
        self.offset = offset
        self.start_by = start_by
        self.get_type()
        self.idx = self.get_idx()

    def get_idx(self):
        if self.idx is None:
            if self.type == "fix":
                return self.expr._get_fix_window_rolling_idx(self.window)
            elif self.type in ["start", "time_start"]:
                return self.time_expr._get_time_rolling_idx(
                    self.window, start_by=self.start_by
                )
            elif self.type in ["offset", "time_offset"]:
                return self.time_expr._get_time_rolling_offset_idx(
                    self.window, self.offset
                )
            else:
                raise ValueError
        else:
            return self.idx

    def get_type(self):
        if self.type is None:
            if self.offset is None:
                if isinstance(self.window, int):
                    self.type = "fix"
                else:
                    self.type = "start"
            else:
                self.type = "offset"

    def __getattr__(self, name):
        idx = self.idx

        def wrap_func(*args, **kwargs):
            if self.type in ["fix", "time_start", "start"]:
                return getattr(self.expr, f"_rolling_select_{name}")(
                    idx, *args, **kwargs
                )
            elif self.type in ["offset", "time_offset"]:
                return getattr(self.expr, f"_rolling_select_by_vecusize_{name}")(
                    idx, *args, **kwargs
                )
            else:
                raise ValueError

        return wrap_func

    def apply(self, agg_expr):
        idx = self.idx
        agg_expr = selector_to_expr(agg_expr, context=True)
        if self.type in ["fix", "start", "time_start"]:
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
        elif self.type in ["offset", "time_offset"]:
            if isinstance(agg_expr, (list, tuple)):
                return [
                    self.expr.apply_with_vecusize(ae, idxs=idx, others=self.others)
                    for ae in agg_expr
                ]
            else:
                return self.expr.apply_with_vecusize(
                    agg_expr, idxs=idx, others=self.others
                )
        else:
            raise ValueError


class ExprGroupBy(ExprRollMixin):
    def __init__(
        self,
        expr,
        by=None,
        time_expr=None,
        idxs=None,
        others=None,
        closed="left",
        sort=True,
        par=False,
        type_=None,
    ) -> None:
        self.type = type_
        self.closed = closed
        self.info = idxs
        self.sort = sort
        self.par = par
        super().__init__(expr, by, time_expr=time_expr, others=others)
        self.get_type()
        self.info = self.get_info()

    def get_type(self):
        if self.type is None:
            if isinstance(self.window, int):
                self.type = "unimplemented"
            elif self.time_expr is not None and isinstance(self.window, str):
                self.type = "time"
            else:
                self.type = "default"

    def get_info(self):
        if self.info is None:
            if self.type == "unimplemented":
                # groupby fix step
                raise NotImplementedError
            elif self.type == "time":
                # groupby time
                return self.time_expr._get_group_by_time_idx(
                    self.window, closed=self.closed, split=True
                )
            elif self.type == "default":
                # groupby keys
                if isinstance(self.window, list):
                    key, other_keys = self.window[0], self.window[1:]
                else:
                    key, other_keys = self.window, None
                return key._get_group_by_idx(other_keys, sort=self.sort, par=self.par)
            else:
                raise ValueError
        else:
            return self.info

    def __getattr__(self, name):
        def wrap_func(*args, **kwargs):
            if self.type == "time":
                return getattr(self.expr, f"_group_by_startidx_{name}")(
                    self.info[1], *args, **kwargs
                )
            elif self.type == "unimplemented":
                raise NotImplementedError
            elif self.type == "default":
                return getattr(self.expr, f"_rolling_select_by_vecusize_{name}")(
                    self.info, *args, **kwargs
                )
            else:
                raise ValueError

        return wrap_func

    def agg(self, agg_expr):
        agg_expr = selector_to_expr(agg_expr, context=True)
        if self.type == "time":
            label, start_idx = self.info
            if isinstance(agg_expr, (list, tuple)):
                return label, [
                    self.expr.group_by_startidx(ae, idx=start_idx, others=self.others)
                    for ae in agg_expr
                ]
            else:
                return label, self.expr.group_by_startidx(
                    agg_expr, idx=start_idx, others=self.others
                )
        elif self.type == "default":
            if isinstance(agg_expr, (list, tuple)):
                return [
                    self.expr.apply_with_vecusize(
                        ae, idxs=self.info, others=self.others
                    )
                    for ae in agg_expr
                ]
            else:
                return self.expr.apply_with_vecusize(
                    agg_expr, idxs=self.info, others=self.others
                )
        elif self.type == "unimplemented":
            raise NotImplementedError
        else:
            raise ValueError
