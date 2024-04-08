from .tears import Expr
from .tears import context as ct


class LazyFunc:
    def __init__(self, name, mod_func=False):
        self.name = name
        self.mod_func = mod_func

    def __repr__(self):
        base_name = self.name if not self.mod_func else f"tp.{self.name}"
        if hasattr(self, "args"):
            args = ", ".join([str(a) for a in self.args])
            if len(self.kwargs):
                kwargs = ", ".join([f"{k}={v}" for k, v in self.kwargs.items()])
                return base_name + f"({args}, {kwargs})"
            else:
                return base_name + f"({args})"
        else:
            return base_name

    def __call__(self, *args, **kwargs):
        out = LazyFunc(self.name, mod_func=self.mod_func)
        out.args = list(args)
        out.kwargs = kwargs
        return out


class Selector:
    def __init__(self, name=None):
        self.name = name
        self.current_func = None
        self.lazy_funcs = []

    def __repr__(self) -> str:
        if len(self.lazy_funcs):
            funcs_str = ".".join([str(lf) for lf in self.lazy_funcs])
            if not self.lazy_funcs[0].mod_func:
                res = f"{self.name}.{funcs_str}"
            else:
                res = funcs_str
        else:
            res = str(self.name)
        if self.current_func is not None:
            res += f".{self.current_func}"
        return res

    def __getattr__(self, name):
        return self._new_with_func(LazyFunc(name))

    def mod_func(self, func):
        return self._new_with_func(LazyFunc(func, mod_func=True), lazy_funcs=[])

    def _new_with_func(self, func, lazy_funcs=None):
        out = Selector(self.name)
        if lazy_funcs is None:
            out.lazy_funcs = self.lazy_funcs.copy()
            if self.current_func is not None:  # attribute, not function
                out.lazy_funcs.append(self.current_func)
        else:
            out.lazy_funcs = lazy_funcs
        out.current_func = func

        return out

    def to_expr(self, dd=None, context=False):
        assert (
            self.current_func is None
        ), f"current_func: {self.current_func} is not None"
        single_flag = False
        if self.name is not None:
            if not context:
                base_exprs = dd[self.name]
                if isinstance(base_exprs, Expr):
                    base_exprs = [base_exprs]
                    single_flag = True
                else:
                    base_exprs = base_exprs.exprs
            else:
                if isinstance(self.name, (list, tuple)):
                    base_exprs = [ct(expr) for expr in self.name]
                else:
                    base_exprs, single_flag = [ct(self.name)], True
        else:
            base_exprs = [None]
            single_flag = True

        def convert_one(sel, dd=None, context=False):
            _e = sel.to_expr(dd, context=context)
            if isinstance(_e, list):
                assert len(_e) == 1, "Selector should return only one Expr"
                return _e[0]
            else:
                return _e

        for j, lf in enumerate(self.lazy_funcs):
            if not hasattr(lf, "args"):
                continue
            # convert argument to Expr
            for i, arg in enumerate(lf.args):
                if isinstance(arg, Selector):
                    lf.args[i] = convert_one(arg, dd=dd, context=context)
                elif isinstance(arg, (list, tuple)) and any(
                    isinstance(a, Selector) for a in arg
                ):
                    if isinstance(arg, tuple):
                        arg = list(arg)
                    lf.args[i] = [
                        convert_one(e, dd=dd, context=context)
                        if isinstance(e, Selector)
                        else e
                        for e in arg
                    ]
            # convert keyword argument to Expr
            for k, v in lf.kwargs.items():
                if isinstance(v, Selector):
                    lf.kwargs[k] = convert_one(v, dd=dd, context=context)
                elif isinstance(v, (list, tuple)) and any(
                    isinstance(e, Selector) for e in v
                ):
                    if isinstance(v, tuple):
                        v = list(v)
                    lf.kwargs[k] = [
                        convert_one(e, dd=dd, context=context)
                        if isinstance(e, Selector)
                        else e
                        for e in v
                    ]
            self.lazy_funcs[j] = lf
        res = []
        for e in base_exprs:
            for lf in self.lazy_funcs:
                if not lf.mod_func:
                    if hasattr(lf, "args"):
                        e = getattr(e, lf.name)(*lf.args, **lf.kwargs)
                    else:
                        e = getattr(e, lf.name)  # attribute, not function
                else:  # module function
                    import teapy as tp

                    mod_list = lf.name.split(".")
                    e = getattr(tp, mod_list[0])
                    for mod in mod_list[1:]:
                        e = getattr(e, mod)
                    e = e(*lf.args, **lf.kwargs)
            if self.current_func is not None:
                if not self.current_func.mod_func:
                    e = getattr(e, self.current_func.name)
                else:
                    raise ValueError(
                        "mod_func should not be used before a function call"
                    )
            res.append(e)
        return res if not single_flag else res[0]

    def __call__(self, *args, **kwargs):
        if self.current_func is not None:
            lazy_funcs = [*self.lazy_funcs, self.current_func(*args, **kwargs)]
            return self._new_with_func(None, lazy_funcs=lazy_funcs)
        else:
            raise RuntimeError("No function to call")


def magic_func(name):
    def _func(self, *args, **kwargs):
        return self._new_with_func(LazyFunc(name))(*args, **kwargs)

    return _func


for magic_func_name in [
    "add",
    "radd",
    "sub",
    "rsub",
    "mul",
    "rmul",
    "truediv",
    "rtruediv",
    "pow",
    "rpow",
    "and",
    "rand",
    "or",
    "ror",
    "neg",
    "getitem",
    "abs",
    "invert",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
]:
    setattr(Selector, f"__{magic_func_name}__", magic_func(f"__{magic_func_name}__"))


def selector_to_expr(exprs, dd=None, context=False):
    def process_one(e):
        if isinstance(e, Selector):
            return e.to_expr(dd=dd, context=context)
        elif isinstance(e, str):
            return dd[e] if not context else ct(e)
        else:
            return e

    if not isinstance(exprs, (list, tuple)):
        return process_one(exprs)
    res = []
    for e in exprs:
        temp = process_one(e)
        if isinstance(temp, list):
            res.extend(temp)
        else:
            res.append(temp)
    return res


if __name__ == "__main__":
    import numpy as np

    import teapy as tp
    from teapy import s

    dd = tp.DataDict(a=np.arange(10), b=np.arange(10))
    print(
        dd.with_columns(
            (2 + s("a")).ts_sum(3, min_periods=2).alias("s1"),
            s("a").ts_corr(s("b"), 3, min_periods=3).alias("s2"),
        ).eval()
    )
