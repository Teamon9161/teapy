import teapy as tp
from teapy import ct
from teapy.testing import assert_allclose


def test_base():
    c = ct("b").ts_sum(2)
    d = ct("a").mean().alias("d")

    dd = tp.DataDict(a=[1.0, 2, 3, 4], b=[4, 3, 1])
    dd = dd.with_columns([d])

    assert_allclose(c.eview(dd), [4, 7, 4])

    dd.eval()
    assert dd["d"].view == 2.5
