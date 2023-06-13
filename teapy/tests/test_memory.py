import numpy as np

import teapy as tp
from teapy import Expr


def test_view_func():
    a = np.random.randn(1000, 200)
    b = Expr(a).ts_sma(5, axis=1).insert_axis(1).swap_axes(0, 1).insert_axis(0).eview()
    assert b[0, 0, 0, 0] == a[0, 0]
