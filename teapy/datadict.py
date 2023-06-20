from .teapy import PyDataDict as _DataDict
from .teapy import from_pandas, stack


def _new_with_dd(dd=None):
    if dd is None:
        # for inplace functions
        return None
    out = DataDict()
    out._dd = dd
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
        return self._dd[key]

    def __setitem__(self, item, value):
        self._dd.__setitem__(item, value)

    def __delitem__(self, item):
        self._dd.__delitem__(item)

    def __getattr__(self, attr):
        return getattr(self._dd, attr)

    @construct
    def eval(self, cols=None, inplace=True):
        return self._dd.eval(cols, inplace=inplace)

    @construct
    def select(self, exprs):
        return self._dd.select(exprs)

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

    def rename(self, mapper):
        return self.with_columns(
            [
                self[key].alias(mapper[key])
                for key in self.columns
                if mapper.get(key) is not None
            ]
        )

    @construct
    def with_columns(self, exprs, inplace=False):
        return self._dd.with_columns(exprs, inplace=inplace)

    @construct
    def join(self, right, on=None, left_on=None, right_on=None, how="left"):
        if on is not None:
            left_on = right_on = on
        if how == "right":
            return right._dd.join(
                self._dd, left_on=right_on, right_on=left_on, method="left"
            )
        else:
            return self._dd.join(right._dd, left_on, right_on, method=how)

    @construct
    def apply(self, func, **kwargs):
        return self._dd.apply(func, **kwargs)

    def rolling(self, window, dd, index=None, check=True, axis=0):
        return Rolling(window, self._dd, index, check, axis)

    @construct
    def sort_by(self, by, rev=False, inplace=False):
        return self._dd.sort_by(by=by, rev=rev, inplace=inplace)

    def groupby(self, by, axis=0, sort=True, par=False, reuse=False):
        return GroupBy(self._dd, by, axis, sort, par, reuse=reuse)

    @construct
    def unique(self, subset, keep="first", inplace=False, check=True, axis=0):
        return self._dd.unique(
            subset, keep=keep, inplace=inplace, check=check, axis=axis
        )


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
