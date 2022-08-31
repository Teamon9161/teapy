from .teapy import *
from . import teapy as _tp
ts_decay_linear = _tp.ts_wma
ts_mean = _tp.ts_sma
ts_ema = _tp.ts_ewm

def rank(arr, axis=0, pct=False, par=False):
    if not pct:
        return _tp.rank(arr, axis=axis, par=par)
    else:
        return _tp.rank_pct(arr, axis=axis, par=par)