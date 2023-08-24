from .teapy import PyDataDict as _DataDict
from .teapy import from_pandas, stack


def _new_with_dd(dd=None):
    if dd is None:
        # for inplace functions
        return None
    out = DataDict()
    if isinstance(dd, _DataDict):
        out._dd = dd
    else:
        raise ValueError("the return is not teapy DataDict")
    return out


def construct(func):
    """For functions that return a DataDict"""

    def inner(*args, **kwargs):
        dd = func(*args, **kwargs)
        return _new_with_dd(dd)

    return inner


def from_pd(*args, **kwargs):
    dd = from_pandas(*args, **kwargs)
    return _new_with_dd(dd)


class DataDict:
    def __init__(self, data=None, columns=None, copy=False, **kwargs):
        if data is not None or len(kwargs):
            if data is None:
                data = []
            elif isinstance(data, dict):
                columns = list(data.keys()) if columns is None else columns
                data = list(data.values())
            if len(kwargs):
                if columns is None:
                    columns = []
                if isinstance(data, tuple):
                    data = list(data)
                for k, v in kwargs.items():
                    data.append(v)
                    columns.append(k)
            self._dd = _DataDict(data=data, columns=columns, copy=copy)
        else:
            self._dd = _DataDict([])

    def __getitem__(self, key):
        out = self._dd[key]
        if isinstance(out, list):
            return DataDict(data=out)
        else:
            return out

    def __setitem__(self, item, value):
        self._dd.__setitem__(item, value)

    def __delitem__(self, item):
        self._dd.__delitem__(item)

    def __getattr__(self, attr):
        return getattr(self._dd, attr)

    def __repr__(self):
        return self._dd.__repr__()

    def __len__(self):
        return len(self._dd)

    @construct
    def copy(self):
        return self._dd.copy()

    def eval(self, cols=None, inplace=True):
        dd = self if inplace else self.copy()
        if isinstance(cols, bool):
            cols, inplace = None, cols
        dd._dd.eval(cols)
        return None if inplace else dd

    def drop(self, cols, inplace=False):
        dd = self if inplace else self.copy()
        dd._dd.drop(cols)
        return None if inplace else dd

    @construct
    def select(self, exprs):
        return self._dd.select(exprs)

    def dropna(self, subset=None, how="all", inplace=False):
        if subset is None:
            subset = self._dd.columns
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

    def mean(self, axis=-1, stable=False, par=False):
        if axis == -1:
            return stack(self.raw_data, axis=axis).mean(axis=-1, stable=stable, par=par)
        else:
            axis = axis + 1 if axis < 0 else axis
            return self.apply(lambda e: e.mean(axis=axis, stable=stable, par=par))

    def sum(self, axis=-1, stable=False, par=False):
        if axis == -1:
            return stack(self.raw_data, axis=axis).sum(axis=-1, stable=stable, par=par)
        else:
            axis = axis + 1 if axis < 0 else axis
            return self.apply(lambda e: e.sum(axis=axis, stable=stable, par=par))

    def min(self, axis=-1, par=False):
        if axis == -1:
            return stack(self.raw_data, axis=axis).min(axis=-1, par=par)
        else:
            axis = axis + 1 if axis < 0 else axis
            return self.apply(lambda e: e.min(axis=axis, par=par))

    def max(self, axis=-1, par=False):
        if axis == -1:
            return stack(self.raw_data, axis=axis).max(axis=-1, par=par)
        else:
            axis = axis + 1 if axis < 0 else axis
            return self.apply(lambda e: e.max(axis=axis, par=par))

    def rename(self, mapper, inplace=False):
        return self.with_columns(
            [
                self[key].alias(mapper[key])
                for key in self.columns
                if mapper.get(key) is not None
            ],
            inplace=inplace,
        )

    def with_columns(self, exprs, inplace=False):
        dd = self if inplace else self.copy()
        dd._dd.with_columns(exprs)
        return None if inplace else dd

    def join(
        self, right, on=None, left_on=None, right_on=None, how="left", inplace=False
    ):
        if on is not None:
            left_on = right_on = on
        if how == "right":
            return right.join(
                self, left_on=right_on, right_on=left_on, how="left", inplace=inplace
            )
        else:
            left_on = [left_on] if isinstance(left_on, (str, int)) else left_on
            right_on = [right_on] if isinstance(right_on, (str, int)) else right_on
            if how == "left":
                left_on = self[left_on].raw_data
                right_on = right[right_on].raw_data
                left_other = left_on[1:] if len(left_on) > 1 else None
                idx = left_on[0]._get_left_join_idx(
                    left_other=left_other, right=right_on
                )
                dd = self if inplace else self.copy()
                dd.with_columns(
                    right._select_on_axis_unchecked(idx, 0).raw_data, inplace=True
                )
                return None if inplace else dd
            else:
                raise NotImplementedError(
                    "Only left an right join is supported for now"
                )

    def apply(self, func, inplace=False, **kwargs):
        dd = self if inplace else self.copy()
        dd._dd.apply(func, **kwargs)
        return None if inplace else dd

    def rolling(self, window, dd, index=None, check=True, axis=0):
        return Rolling(window, self._dd, index, check, axis)

    def _select_on_axis(self, idx, axis=0, inplace=False):
        return self.apply(lambda e: e.select(idx, axis=axis), inplace=inplace)

    def _select_on_axis_unchecked(self, idx, axis=0, inplace=False):
        return self.apply(
            lambda e: e._select_unchecked(idx, axis=axis), inplace=inplace
        )

    def sort(self, by, rev=False, inplace=False):
        if isinstance(by, (str, int)):
            by = [by]
        if self.is_empty():
            return None if inplace else self
        idx = self[0].sort(self[by], rev=rev, return_idx=True)
        dd = self if inplace else self.copy()
        dd._select_on_axis_unchecked(idx, axis=0, inplace=True)
        return None if inplace else dd

    def unique(self, subset, keep="first", inplace=False):
        if isinstance(subset, (str, int)):
            subset = [subset]
        if self.is_empty():
            return None if inplace else self
        first_key, subset = subset[0], subset[1:]
        subset = None if len(subset) == 0 else self[subset]
        idx = self[first_key]._get_unique_idx(subset, keep=keep)
        dd = self if inplace else self.copy()
        dd._select_on_axis_unchecked(idx, axis=0, inplace=True)
        return None if inplace else dd

    def groupby(self, by, axis=0, sort=True, par=False, reuse=False):
        return GroupBy(self._dd, by, axis, sort, par, reuse=reuse)


class Rolling:
    def __init__(self, window, dd, index=None, check=True, axis=0) -> None:
        """
        window: rolling window, int or str
        dd: PyDataDict
        index: time index, if None then infer automatically
        check: whether to check the length of each key is equal
        axis: rolling on which axis
        """
        self.window = window
        self._dd = dd
        self.index = index
        self.check = check
        self.axis = axis

    @construct
    def apply(self, func, **kwargs):
        if isinstance(self.window, str):
            # rolling using a time duration
            return self._dd.rolling_apply_by_time(
                index=self.index,
                duration=self.window,
                axis=self.axis,
                func=func,
                check=self.check,
                **kwargs,
            )
        else:
            return self._dd.rolling_apply(
                window=self.window,
                axis=self.axis,
                func=func,
                check=self.check,
                **kwargs,
            )


class GroupBy:
    def __init__(self, dd, by, axis=0, sort=True, par=False, reuse=False) -> None:
        self._dd = dd
        self.by = by
        self.axis = axis
        self.sort = sort
        self.par = par
        self.reuse = reuse
        self.groupby = None

    @construct
    def apply(self, func, **kwargs):
        if not self.reuse:
            out = self._dd.groupby_apply(
                by=self.by,
                axis=self.axis,
                sort=self.sort,
                par=self.par,
                py_func=func,
                **kwargs,
            )
        else:
            if self.groupby is None:
                self.groupby = self._dd.groupby(
                    by=self.by, axis=self.axis, sort=self.sort, par=self.par
                )
            out = self.groupby.apply(func, **kwargs)
        return out
