from functools import wraps

from .selector import Selector
from .tears import arange, stack
from .tears import corr as _corr


def select_wrapper(func):
    name = func.__qualname__
    @wraps(func)
    def _wrap_func(exprs, *args, **kwargs):
        if len(exprs) and isinstance(exprs[0], Selector):
            return Selector().mod_func(name)(exprs, *args, **kwargs)
        else:
            return func(exprs, *args, **kwargs)
    return _wrap_func


@select_wrapper
def hcorr(exprs, method="pearson", stable=False):
    return _corr(exprs, method=method, stable=stable)
    # return PyDataDict(exprs).corr(method=method, stable=stable)


@select_wrapper
def hmax(exprs, par=False):
    """horizontal max"""
    return stack(exprs, axis=-1).max(axis=-1, par=par)


@select_wrapper
def hmin(exprs, par=False):
    """horizontal min"""
    return stack(exprs, axis=-1).min(axis=-1, par=par)


@select_wrapper
def hmean(exprs, par=False, min_periods=1):
    """horizontal mean"""
    return stack(exprs, axis=-1).mean(axis=-1, par=par, min_periods=min_periods)


@select_wrapper
def hstd(exprs, par=False, min_periods=3):
    """horizontal std"""
    return stack(exprs, axis=-1).std(axis=-1, par=par, min_periods=min_periods)


@select_wrapper
def hsum(exprs, par=False):
    return stack(exprs, axis=-1).sum(axis=-1, par=par)


def align_frames(dds, by, sort=True, rev=False, outer_df=False, with_by=True):
    suffix = "__align_by"
    if len(dds) <= 1:
        return dds
    if not isinstance(by, (list, tuple)):
        by = [by]

    dd_outer = dds[0].apply(lambda e: e.suffix(suffix + "0"), exclude=by)
    # s = 0
    for i, rdd in enumerate(dds[1:]):
        rdd = rdd.apply(lambda e, i=i: e.suffix(suffix + str(i + 1)), exclude=by)
        dd_outer.join(rdd, on=by, how="outer", sort=False, inplace=True)
        # # to avoid stack overflow
        # s += 1
        # if s == step:
        #     s = 0
        #     dd_outer.eval()
    if sort:
        dd_outer.sort(by=by, rev=rev, inplace=True)
    if outer_df:
        return dd_outer if with_by else dd_outer.exclude(by)
    if with_by:
        return [
            dd_outer[[*by, "^.*" + suffix + f"{i}$"]].apply(
                lambda e, i=i: e.alias(e.name.replace(suffix + str(i), ""))
            )
            for i in range(len(dds))
        ]
    else:
        return dd_outer[by], [
            dd_outer["^.*" + suffix + f"{i}$"].apply(
                lambda e, i=i: e.alias(e.name.replace(suffix + str(i), ""))
            )
            for i in range(len(dds))
        ]


def get_align_frames_idx(dds, by, sort=True, rev=False, return_by=False):
    if len(dds) <= 1:
        return dds
    if not isinstance(by, (list, tuple)):
        by = [by]
    by0 = dds[0][by].exprs
    out_idxs = [arange(by0[0].shape[0])]
    for i, rdd in enumerate(dds[1:]):
        left_other = by0[1:] if len(by) > 1 else None
        *by0, left_idx, right_idx = by0[0]._get_outer_join_idx(
            left_other=left_other, right=rdd[by].exprs, sort=sort, rev=rev
        )
        for i, idx in enumerate(out_idxs):
            out_idxs[i] = idx.select(left_idx, check=False).cast("opt<usize>")
        out_idxs.append(right_idx)
    return out_idxs if not return_by else (by0, out_idxs)
