from .teapy import PyDataDict as _DataDict
from .teapy import from_pandas


def _new_with_dd(dd=None):
    if dd is None:
        return None
    out = DataDict()
    out._dd = dd
    return out


def construct(func):
    def inner(*args, **kwargs):
        dd = func(*args, **kwargs)
        return _new_with_dd(dd)

    return inner


def from_pd(*args, **kwargs):
    dd = from_pandas(*args, **kwargs)
    return _new_with_dd(dd)


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
                py_kwargs=kwargs,
            )
        else:
            return self._dd.rolling_apply(
                window=self.window,
                axis=self.axis,
                func=func,
                check=self.check,
                py_kwargs=kwargs,
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
                py_kwargs=kwargs,
            )
        else:
            if self.groupby is None:
                self.groupby = self._dd.groupby(
                    by=self.by, axis=self.axis, sort=self.sort, par=self.par
                )
            out = self.groupby.apply(py_func=func, py_kwargs=kwargs)
        return out


class DataDict:
    def __init__(self, *args, **kwargs):
        if len(args) or len(kwargs):
            self._dd = _DataDict(*args, **kwargs)
        else:
            self._dd = None

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

    @construct
    def with_columns(self, exprs, inplace=False):
        return self._dd.with_columns(exprs, inplace=inplace)

    @construct
    def join(self, right, on=None, left_on=None, right_on=None, how="left"):
        if on is not None:
            left_on = right_on = on
        if how == "right":
            return right._dd.join(self._dd, left_on, right_on, method="left")
        else:
            return self._dd.join(right._dd, left_on, right_on, method=how)

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
