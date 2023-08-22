import numpy as np

import teapy as tp
from teapy.testing import assert_allclose


def test_slice():
    a = np.random.randn(100, 39)
    e = tp.Expr(a, copy=False)
    assert_allclose(e[:10, :].eview(), a[:10, :])
    assert_allclose(e[-4:9, :].eview(), a[-4:9, :])
    assert_allclose(e[:, -2:-4].eview(), a[:, -2:-4])
    assert_allclose(e[None, :, :].eview(), a[None, :, :])
    assert_allclose(e[[-3, -5], 3:9].eview(), a[[-3, -5], 3:9])
    e = tp.Expr(a, copy=True)
    assert_allclose(e.argsort(axis=1)[:, :10].eview(), a.argsort(axis=1)[:, :10])
    # # currentyly the logic is not the same
    # assert_allclose(e[[-3, -5], [3, -2]].eview(), a[[-3, -5], [3, -2]])

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    e = tp.Expr(a)
    assert e[-1].eview() == 8
    assert e[-3].eview() == 6
    assert_allclose(e[[-1, -2, -3, 1, 0]].eview(), a[[-1, -2, -3, 1, 0]])


def test_unique():
    assert_allclose(tp.Expr([1, 3, 2, 1, 2]).unique().eview(), [1, 3, 2])
    e = tp.Expr(["b", "bb", "a", "ab", "ab", "bb"]).unique()
    assert e.eview().tolist() == ["b", "bb", "a", "ab"]

    # test sorted unique
    a = tp.Expr([2, 2, 3, 3, 4, 5, 5])
    e1 = a._get_sorted_unique_idx("first").eview()
    assert_allclose(e1, [0, 2, 4, 5])
    e2 = a._get_sorted_unique_idx("last").eview()
    assert_allclose(e2, [1, 3, 4, 6])
    e3 = a.sorted_unique().eview()
    assert_allclose(e3, [2, 3, 4, 5])
