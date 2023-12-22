import numpy as np

import teapy as tp
from teapy import ct
from teapy.regression import Ols
from teapy.testing import assert_allclose


def test_base():
    c = ct("b").ts_sum(2)
    d = ct("a").mean().alias("d")

    dd = tp.DataDict(a=[1.0, 2, 3, 4], b=[4, 3, 1])
    dd = dd.with_columns(d)

    assert_allclose(c.eview(dd, freeze=False), [4, 7, 4])
    assert c.step > 0
    assert_allclose(c.eview(dd, freeze=True), [4, 7, 4])
    assert_allclose(c.view, [4, 7, 4])
    dd.eval(context=True)
    assert dd["d"].view == 2.5


def test_rolling_context():
    x = tp.Expr([1, 2, 3, 4, 5])
    y = tp.Expr([2, 3, 4, 5, 6])
    assert_allclose(x.rolling(2).apply(ct(0).sum()).eview(), np.array([1, 3, 5, 7, 9]))
    assert_allclose(
        x.rolling(2, others=y).apply(-ct(0).sum() + ct(1).sum()).eview(),
        np.array([1, 2, 2, 2, 2]),
    )
    x = tp.Expr([1, 2, np.nan, 4, 5])
    y = tp.Expr([2, 3, 4, 5, 6])

    assert_allclose(
        x.rolling(4, others=y).apply(Ols(ct(1), ct(0)).params.last()).eview(),
        np.array([0, 1, 1, 1, 1]),
    )
