import numpy as np
import teapy as tp
from teapy.testing import assert_allclose


def test_continuity():
    # test output continuity
    arr1 = np.array(np.random.randn(100, 20), order="c")
    assert tp.ts_sma(arr1, window=3).flags["C_CONTIGUOUS"]
    arr2 = np.array(np.random.randn(100, 20), order="f")
    assert tp.ts_sma(arr2, window=3).flags["F_CONTIGUOUS"]
    assert tp.ts_cov(arr2, arr1, window=5).flags["F_CONTIGUOUS"]

    # test argsort on discontinuous axis
    res1 = tp.argsort(arr1, axis=0)
    res2 = arr1.argsort(axis=0)
    assert_allclose(res1, res2)


def test_step():
    a = tp.Expr([1, 2, 3]).ts_mean(1).mean()
    assert a.step == 2


def test_high_dimensional():
    arr = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    expected_axis0 = np.array([[[0, 1], [2, 3]], [[4, 6], [8, 10]]])
    expected_axis1 = np.array([[[0, 1], [2, 4]], [[4, 5], [10, 12]]])
    expected_axis2 = np.array([[[0, 1], [2, 5]], [[4, 9], [6, 13]]])

    assert_allclose(tp.ts_sum(arr, 2, axis=0), expected_axis0)
    assert_allclose(tp.ts_sum(arr, 2, axis=1), expected_axis1)
    assert_allclose(tp.ts_sum(arr, 2, axis=2), expected_axis2)


def test_special():
    # test rank a array with 0 element
    assert tp.rank(np.array([])).tolist() == []
    assert tp.rank(np.array([2]))[0] == 1  # test rank a array with 1 element
    # test rank all nan array
    assert_allclose(tp.rank(np.array([np.nan, np.nan])), np.array([np.nan, np.nan]))


def test_register():
    @tp.register
    def mean_test(e, axis):
        return e.mean(axis=axis)

    assert tp.Expr([1, 2, 3]).mean_test(axis=0).eview() == 2


def test_eval():
    dd = tp.DataDict(a=[1, 2, 3, 4])
    dd1 = dd.with_columns(dd["a"].ts_mean(2).alias("b"))
    dd2 = dd.with_columns(dd["a"].ts_sum(3).alias("c"))

    # eval exprs
    tp.eval([dd1["b"], dd2["c"]])
    assert_allclose(dd2["c"].view, [1, 3, 6, 9])
    # eval datadicts
    dd1 = dd.with_columns(dd["a"].ts_mean(2).alias("b"))
    dd2 = dd.with_columns(dd["a"].ts_sum(3).alias("c"))
    tp.eval([dd1, dd2])
    assert_allclose(dd1["b"].view, [1, 1.5, 2.5, 3.5])
