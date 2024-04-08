from .selector import selector_to_expr
from .tears import Expr, eval_exprs, stack
from .tears import context as ct
from .tears import scan_ipc as _scan_ipc

name_prefix = "column_"


def scan_ipc(path, columns=None):
    return DataDict(_scan_ipc(path, columns=columns))


def from_dataframe(df, copy=False):
    return DataDict(df.to_dict(), copy=copy)


def from_pd(df, copy=False):
    return from_dataframe(df, copy=copy)


def from_pl(df, copy=False):
    return from_dataframe(df, copy=copy)


class DataDict:
    default_name = name_prefix

    def __init__(self, data=None, columns=None, copy=False, **kwargs):
        self.copy_flag = copy
        self.col_map = None
        self.auto_idx = 0
        if data is None:
            self.exprs = []
        elif isinstance(data, dict):
            if columns is None:
                self.exprs = [Expr(v, copy=copy).alias(k) for k, v in data.items()]
            else:  # columns has a higher priority
                assert len(columns) == len(data)
                self.exprs = [
                    Expr(v, copy=copy).alias(name)
                    for name, (k, v) in zip(columns, data.items())
                ]
        else:
            if isinstance(data, (list, tuple)):
                if columns is not None:
                    if len(columns) != len(data):
                        raise ValueError("columns and data must have the same length")
                else:  # use expr name in data and auto generate column names if missing
                    columns = [
                        e.name
                        if isinstance(e, Expr) and e.name is not None
                        else self._auto_name()
                        for e in data
                    ]
                self.exprs = [
                    Expr(v, copy=copy).alias(k) for k, v in zip(columns, data)
                ]
            else:
                raise ValueError("data must be a dict, list or tuple")
        if len(kwargs):
            for k, v in kwargs.items():
                self.exprs.append(Expr(v, copy=copy).alias(k))

    def _init_col_map(self, force_init=False):
        if self.col_map is None or force_init:
            self.col_map = {e.name: i for i, e in enumerate(self.exprs)}

    def _auto_name(self) -> str:
        """Generate a new column name."""
        name = self.default_name + str(self.auto_idx)
        self.auto_idx += 1
        return name

    def __iter__(self):
        return iter(self.exprs)

    @property
    def columns(self):
        return [e.name for e in self.exprs]

    @property
    def dtypes(self):
        return {e.name: e.dtype for e in self.exprs}

    @property
    def raw_data(self):
        from warnings import warn

        warn(
            "raw_data will be deprecated in future release, use exprs instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.exprs

    def is_empty(self):
        return len(self) == 0

    def _new_with_exprs(self, exprs, copy_map=False):
        dd = DataDict(exprs, copy=self.copy_flag)
        if copy_map and self.col_map is not None:
            dd.col_map = self.col_map.copy()
        return dd

    def copy(self):
        return self._new_with_exprs(self.exprs, copy_map=True)

    def get(self, key):
        if isinstance(key, int):
            return self.exprs[key]
        elif isinstance(key, str):
            self._init_col_map()
            if key.startswith("^") and key.endswith("$"):
                import re

                key = re.compile(key)
                return self._new_with_exprs(
                    [e for e in self.exprs if key.match(e.name)], copy_map=False
                )
            return self.exprs[self.col_map[key]]
        elif isinstance(key, (list, tuple)):
            new_data = []
            for k in key:
                res = self.get(k)
                if not isinstance(res, DataDict):
                    new_data.append(res)
                else:
                    new_data.extend(res.exprs)
            return DataDict(new_data, copy=self.copy_flag)
        else:
            raise TypeError("key must be int, str, list or tuple")

    def set(self, key, value):
        if isinstance(key, int):
            ori_name = self.exprs[key].name
            value = Expr(value, copy=self.copy_flag)
            new_name = ori_name if value.name is None else value.name
            self.exprs[key] = value.alias(new_name)
            # update col_map
            if self.col_map is not None and new_name != ori_name:
                self.col_map[new_name] = self.col_map.pop(ori_name)
        elif isinstance(key, str):
            self._init_col_map()
            if key in self.col_map:  # update an existing column
                idx = self.col_map[key]
                self.exprs[idx] = Expr(value, copy=self.copy_flag).alias(key)
            elif key.startswith("^") and key.endswith("$"):
                exprs = self.get(key)
                columns = [exprs.name] if isinstance(exprs, Expr) else exprs.columns
                for k, v in zip(columns, value):
                    self.set(k, v)
            else:  # add a new column
                self.exprs.append(Expr(value, copy=self.copy_flag).alias(key))
                self.col_map[key] = len(self.exprs) - 1
        elif isinstance(key, (list, tuple)):
            for k, v in zip(key, value):
                self.set(k, v)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        return self.set(key, value)

    def __delitem__(self, key):
        self.drop(key, inplace=True)

    def __len__(self):
        return len(self.exprs)

    def to_dict(self):
        return {e.name: e.view for e in self.exprs}

    def to_pd(self):
        import pandas as pd

        return pd.DataFrame(self.to_dict())

    def to_pl(self):
        import polars as pl

        return pl.DataFrame(self.to_dict())

    def __repr__(self) -> str:
        return {e.name: e for e in self.exprs}.__repr__()

    def eval(self, columns=None, inplace=False):
        if columns is None:
            self.exprs = eval_exprs(self.exprs)
        else:
            exprs = self.get(columns)
            if isinstance(exprs, Expr):
                exprs.eval(inplace=True)
            else:
                exprs = eval_exprs(exprs.exprs)
            self.set(columns, exprs)
        return self if not inplace else None

    def apply(self, func, *args, inplace=False, exclude=None, **kwargs):
        if exclude is None:
            exprs = [func(e, *args, **kwargs) for e in self.exprs]
        else:
            exprs = [func(e, *args, **kwargs) for e in self.exclude(exclude).exprs]
        if inplace:
            self.exprs = exprs
            return self
        return self._new_with_exprs(exprs)

    def _process_selector_exprs(self, exprs, context=False):
        return selector_to_expr(exprs, dd=self, context=context)

    def select(self, exprs, *args, **kwargs):
        if not isinstance(exprs, list):
            exprs = [exprs]
        if len(args):
            exprs += list(args)
        if len(kwargs):
            for k, v in kwargs.items():
                exprs.append(v.alias(k))
        exprs = self._process_selector_exprs(exprs)
        return self._new_with_exprs(exprs)

    def with_columns(self, exprs, *args, inplace=False, **kwargs):
        dd = self if inplace else self.copy()
        if not isinstance(exprs, list):
            exprs = [exprs]
        if len(args):
            exprs += list(args)
        if len(kwargs):
            for k, v in kwargs.items():
                exprs.append(v.alias(k))
        exprs = self._process_selector_exprs(exprs)
        expr_names = [e.name for e in exprs]
        dd[expr_names] = exprs
        return None if inplace else dd

    def drop(self, key, inplace=False):
        dd = self.copy() if not inplace else self
        if isinstance(key, int):
            dd.exprs.pop(key)
        elif isinstance(key, str):
            dd.exprs = [e for e in dd.exprs if e.name != key]
        elif isinstance(key, (list, tuple)):
            exprs = []
            for i, e in enumerate(dd.exprs):
                if e.name not in key and i not in key:
                    exprs.append(e)
            dd.exprs = exprs
        dd._init_col_map(force_init=True)
        return dd

    def simplify(self):
        def simplify_f(e):
            e.simplify()
            return e

        self.apply(simplify_f, inplace=True)

    def dropna(self, subset=None, how="all", inplace=False):
        if subset is None:
            subset = self.columns
        elif isinstance(subset, (str, int)):
            subset = [subset]
        if len(subset) == 0:
            return None if inplace else self
        else:
            nan_mask = self[subset[0]].is_nan()
            if how == "any":
                for c in subset[1:]:
                    nan_mask |= self[c].is_nan()
            elif how == "all":
                for c in subset[1:]:
                    nan_mask &= self[c].is_nan()
            else:
                raise ValueError("how should be either 'any' or 'all'")
            dd = self if inplace else self.copy()
            for c in subset:
                dd[c] = dd[c].filter(~nan_mask)
            return None if inplace else dd

    def _select_on_axis(self, idx, axis=0, inplace=False, check=True):
        return self.apply(
            lambda e: e.select(idx, axis=axis, check=check), inplace=inplace
        )

    def slice(self, idx, axis=0, check=True, inplace=False):
        return self._select_on_axis(idx, axis=axis, inplace=inplace, check=check)

    def filter(self, mask, inplace=False):
        dd = self if inplace else self.copy()
        idx = mask.mask_to_idx()
        dd._select_on_axis(idx, axis=0, inplace=True, check=False)
        return None if inplace else dd

    def mean(self, axis=-1, stable=False, par=False, min_periods=1):
        if axis == -1:
            return stack(self.exprs, axis=axis).mean(
                axis=-1, stable=stable, par=par, min_periods=min_periods
            )
        else:
            axis = axis + 1 if axis < 0 else axis
            return self.apply(
                lambda e: e.mean(
                    axis=axis, stable=stable, par=par, min_periods=min_periods
                )
            )

    def std(self, axis=-1, stable=False, par=False, min_periods=3):
        if axis == -1:
            return stack(self.exprs, axis=axis).std(
                axis=-1, stable=stable, par=par, min_periods=min_periods
            )
        else:
            axis = axis + 1 if axis < 0 else axis
            return self.apply(
                lambda e: e.std(
                    axis=axis, stable=stable, par=par, min_periods=min_periods
                )
            )

    def sum(self, axis=-1, stable=False, par=False):
        if axis == -1:
            return stack(self.exprs, axis=axis).sum(axis=-1, stable=stable, par=par)
        else:
            axis = axis + 1 if axis < 0 else axis
            return self.apply(lambda e: e.sum(axis=axis, stable=stable, par=par))

    def min(self, axis=-1, par=False):
        if axis == -1:
            return stack(self.exprs, axis=axis).min(axis=-1, par=par)
        else:
            axis = axis + 1 if axis < 0 else axis
            return self.apply(lambda e: e.min(axis=axis, par=par))

    def max(self, axis=-1, par=False):
        if axis == -1:
            return stack(self.exprs, axis=axis).max(axis=-1, par=par)
        else:
            axis = axis + 1 if axis < 0 else axis
            return self.apply(lambda e: e.max(axis=axis, par=par))

    def rename(self, mapper, inplace=False):
        dd = self if inplace else self.copy()
        if isinstance(mapper, (list, tuple)):
            assert len(mapper) == len(dd.columns)
            dd.exprs = [e.alias(m) for e, m in zip(dd.exprs, mapper)]
        elif isinstance(mapper, dict):
            for key, value in mapper.items():
                dd[value] = dd[key].alias(value)
                if key != value:
                    del dd[key]
        else:
            raise TypeError("mapper should be either list or dict")
        return None if inplace else dd

    def exclude(self, cols):
        if not isinstance(cols, (tuple, list)):
            cols = [cols]
        return self._new_with_exprs([e for e in self.exprs if e.name not in cols])

    def join(
        self,
        right,
        on=None,
        left_on=None,
        right_on=None,
        how="left",
        inplace=False,
        sort=True,
        rev=False,
        simplify=True,
        eager=False,
    ):
        if on is not None:
            left_on = right_on = on
        if how == "right":
            return right.join(
                self,
                left_on=right_on,
                right_on=left_on,
                how="left",
                inplace=inplace,
                eager=eager,
            )
        else:
            dd = self if inplace else self.copy()
            left_on = [left_on] if isinstance(left_on, (str, int)) else left_on
            right_on = [right_on] if isinstance(right_on, (str, int)) else right_on
            left_keys = self[left_on].exprs
            right_keys = right[right_on].exprs
            left_other = left_keys[1:] if len(left_keys) > 1 else None
            if how == "left":
                idx = left_keys[0]._get_left_join_idx(
                    left_other=left_other, right=right_keys
                )
                dd.with_columns(
                    right.drop(right_on, inplace=False)
                    ._select_on_axis(idx, 0, check=False)
                    .exprs,
                    inplace=True,
                )
                if simplify:
                    dd.simplify()
                if eager:
                    dd.eval()
                return None if inplace else dd
            elif how == "outer":
                *outer_keys, left_idx, right_idx = left_keys[0]._get_outer_join_idx(
                    left_other=left_other, right=right_keys, sort=sort, rev=rev
                )
                dd.drop(left_on, inplace=True)
                dd.apply(lambda e: e.select(left_idx, check=False), inplace=True)
                dd.with_columns(outer_keys, inplace=True)
                dd.with_columns(
                    right.drop(right_on, inplace=False)
                    .slice(right_idx, 0, check=False)
                    .exprs,
                    inplace=True,
                )
                if simplify:
                    dd.simplify()
                if eager:
                    dd.eval()
                return None if inplace else dd
            else:
                raise NotImplementedError(
                    "Only left | right | outer join is supported for now"
                )

    def sort(self, by, rev=False, inplace=False):
        if isinstance(by, (str, int)):
            by = [by]
        if self.is_empty():
            return None if inplace else self
        idx = self[0].sort(self[by].exprs, rev=rev, return_idx=True)
        dd = self if inplace else self.copy()
        dd.slice(idx, axis=0, inplace=True, check=False)
        return None if inplace else dd

    def unique(self, subset, keep="first", inplace=False):
        if isinstance(subset, (str, int)):
            subset = [subset]
        if self.is_empty():
            return None if inplace else self
        first_key, subset = subset[0], subset[1:]
        subset = None if len(subset) == 0 else self[subset].exprs
        idx = self[first_key]._get_unique_idx(subset, keep=keep)
        dd = self if inplace else self.copy()
        dd.slice(idx, axis=0, inplace=True, check=False)
        return None if inplace else dd

    def groupby(self, by=None, time_col=None, closed="left", group=True):
        return GroupBy(self, by=by, time_col=time_col, closed=closed, group=group)

    def corr(self, columns=None, method="pearson", min_periods=1, stable=False):
        from .tears import corr

        exprs = self.exprs if columns is None else self[columns].exprs
        return corr(exprs, method=method, min_periods=min_periods, stable=stable)


class GroupBy:
    def __init__(self, dd, by=None, time_col=None, closed="left", group=True) -> None:
        self.dd = dd
        self.by = by
        self.time_col = time_col
        self.closed = closed
        self.group = group

    def agg(self, exprs=None, **kwargs):
        if self.dd.is_empty():
            return self.dd
        time_expr = self.dd[self.time_col] if self.time_col is not None else None
        e = self.dd[0]
        columns = self.dd.columns
        others = self.dd[columns[1:]].exprs
        by = self.by if time_expr is not None else self.dd[self.by]
        groupby_obj = e.groupby(
            by=by, time_expr=time_expr, closed=self.closed, others=others
        )
        info, type_ = groupby_obj.info, groupby_obj.type
        self.info = info
        data = []
        if exprs is not None:
            if time_expr is None:
                data_agg = groupby_obj.agg(exprs)
            else:
                _, data_agg = groupby_obj.agg(exprs)
            data_agg = [data_agg] if not isinstance(data_agg, list) else data_agg
            data.extend(data_agg)
        if self.group and type_ == "time":
            data.append(info[0])
        else:
            if not isinstance(self.by, (list, tuple)):
                by = [self.by]
            for g in by:
                kwargs[g] = "first"
        if len(kwargs):
            data_direct = []
            for k, v in kwargs.items():
                import re

                if "(" in v and v.split("(")[0] in ["corr"]:
                    pattern = r"\((.*?))"
                    if "," in v:  # call function with args
                        pattern = r"\((.*?),"
                        eval_info = re.sub(
                            pattern,
                            f"(self.dd['{re.findall(pattern, v)[0]}'],",
                            v,
                        )
                    else:
                        pattern = r"\((.*?)\)"
                        eval_info = re.sub(
                            pattern,
                            f"(self.dd['{re.findall(pattern, v)[0]}'])",
                            v,
                        )
                else:
                    eval_info = v if "(" in v else v + "()"
                data_direct.append(
                    eval(f"self.dd['{k}'].groupby(idxs=info, type_=type_)." + eval_info)
                )
            data.extend(data_direct)
        return DataDict(data)
