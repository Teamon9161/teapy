import numpy as np
import teapy as tp
from teapy import s
from teapy.regression import Ols
from teapy.testing import assert_allclose


def test_base():
    c = s("b").ts_sum(2).alias("c")
    d = s("a").mean().alias("d")

    dd = tp.DataDict(a=[1.0, 2, 3, 4], b=[4, 3, 1])
    dd = dd.with_columns(c, d)
    # assert_allclose(c.eview(), [4, 7, 4])
    assert_allclose(dd["c"].view, [4, 7, 4])
    dd.eval()
    assert dd["d"].view == 2.5


def test_rolling_context():
    x = tp.Expr([1, 2, 3, 4, 5])
    y = tp.Expr([2, 3, 4, 5, 6])
    assert_allclose(x.rolling(2).apply(s(0).sum()).eview(), np.array([1, 3, 5, 7, 9]))
    assert_allclose(
        x.rolling(2, others=y).apply(-s(0).sum() + s(1).sum()).eview(),
        np.array([1, 2, 2, 2, 2]),
    )
    x = tp.Expr([1, 2, np.nan, 4, 5])
    y = tp.Expr([2, 3, 4, 5, 6])

    assert_allclose(
        x.rolling(4, others=y).apply(Ols(s(1), s(0)).params.last()).eview(),
        np.array([0, 1, 1, 1, 1]),
    )
