from .teapy import stack


def hmax(exprs, par=False):
    """horizontal max"""
    return stack(exprs, axis=-1).max(axis=-1, par=par)


def hmin(exprs, par=False):
    """horizontal min"""
    return stack(exprs, axis=-1).min(axis=-1, par=par)


def hmean(exprs, par=False):
    """horizontal mean"""
    return stack(exprs, axis=-1).mean(axis=-1, par=par)


def hstd(exprs, par=False):
    """horizontal std"""
    return stack(exprs, axis=-1).std(axis=-1, par=par)


def hsum(exprs, par=False):
    return stack(exprs, axis=-1).sum(axis=-1, par=par)
