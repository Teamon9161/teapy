from multiprocessing import cpu_count
from time import time

import numpy as np

import teapy as tp
from teapy.testing import assert_allclose


def test_continuity():
    arr1 = np.array(np.random.randn(100, 20), order="c")
    assert tp.ts_sma(arr1, window=3).flags["C_CONTIGUOUS"]
    arr2 = np.array(np.random.randn(100, 20), order="f")
    assert tp.ts_sma(arr2, window=3).flags["F_CONTIGUOUS"]
    assert tp.ts_cov(arr1, arr2, window=5).flags["C_CONTIGUOUS"]


def test_high_dimensional():
    arr = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    expected_axis0 = np.array([[[0, 1], [2, 3]], [[4, 6], [8, 10]]])
    expected_axis1 = np.array([[[0, 1], [2, 4]], [[4, 6], [10, 12]]])
    expected_axis2 = np.array([[[0, 1], [2, 5]], [[4, 9], [6, 13]]])

    assert_allclose(tp.ts_sum(arr, 0), expected_axis0)
    assert_allclose(tp.ts_sum(arr, 1), expected_axis1)
    assert_allclose(tp.ts_sum(arr, 2), expected_axis2)


def test_parallel():
    arr = np.random.randn(300, 100000)
    # no parallel
    start = time()
    tp.ts_std(arr, window=10, par=False)
    time1 = time() - start

    # parallel
    start = time()
    tp.ts_std(arr, window=10, par=True)
    time2 = time() - start

    if cpu_count() > 1:
        assert time1 < time2
