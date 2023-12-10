import numpy as np
import teapy as tp


def test_nat():
    for unit in ["ns", "us", "ms"]:
        a = np.array(
            [np.datetime64("nat", unit), np.datetime64("2020-01-01 00:00:02", unit)]
        )
        b = tp.Expr(a)
        assert np.isnat(b.view[0])
        assert b.view[1] == a[1]
        # assert (b[1] - b[0]).eview()
