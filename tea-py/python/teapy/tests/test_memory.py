import numpy as np
import teapy as tp
from teapy import Expr
from teapy.testing import assert_allclose


def test_view_func():
    a = np.random.randn(1000, 200)
    b = Expr(a).ts_sma(5, axis=1).insert_axis(1).swap_axes(0, 1).insert_axis(0).eview()
    assert b[0, 0, 0, 0] == a[0, 0]


def test_viewmut_func():
    a = np.random.randn(1000, 200)
    b = Expr(a).swap_axes(0, 1).put_mask(a.T > 0, -100).eview()
    assert_allclose(b, np.where(a.T > 0, -100, a.T))
