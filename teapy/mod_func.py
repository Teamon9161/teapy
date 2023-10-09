from .datadict import DataDict
from .teapy import PyDataDict, stack


def hcorr(exprs, method="pearson", stable=False):
    return PyDataDict(exprs).corr(method=method, stable=stable)


def hmax(exprs, par=False):
    """horizontal max"""
    return stack(exprs, axis=-1).max(axis=-1, par=par)


def hmin(exprs, par=False):
    """horizontal min"""
    return stack(exprs, axis=-1).min(axis=-1, par=par)


def hmean(exprs, par=False, min_periods=1):
    """horizontal mean"""
    return stack(exprs, axis=-1).mean(axis=-1, par=par, min_periods=min_periods)


def hstd(exprs, par=False, min_periods=3):
    """horizontal std"""
    return stack(exprs, axis=-1).std(axis=-1, par=par, min_periods=min_periods)


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
        rdd = rdd.apply(lambda e: e.suffix(suffix + str(i + 1)), exclude=by)
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
            dd_outer[by + ["^.*" + suffix + f"{i}$"]].apply(
                lambda e: e.alias(e.name.replace(suffix + str(i), ""))
            )
            for i in range(len(dds))
        ]
    else:
        return dd_outer[by], [
            dd_outer["^.*" + suffix + f"{i}$"].apply(
                lambda e: e.alias(e.name.replace(suffix + str(i), ""))
            )
            for i in range(len(dds))
        ]
