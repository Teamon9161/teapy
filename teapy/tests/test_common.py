from time import time

import numpy as np
import pandas as pd

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


def test_high_dimensional():
    arr = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    expected_axis0 = np.array([[[0, 1], [2, 3]], [[4, 6], [8, 10]]])
    expected_axis1 = np.array([[[0, 1], [2, 4]], [[4, 5], [10, 12]]])
    expected_axis2 = np.array([[[0, 1], [2, 5]], [[4, 9], [6, 13]]])

    assert_allclose(tp.ts_sum(arr, 2, axis=0), expected_axis0)
    assert_allclose(tp.ts_sum(arr, 2, axis=1), expected_axis1)
    assert_allclose(tp.ts_sum(arr, 2, axis=2), expected_axis2)


# def test_parallel():
#     from multiprocessing import cpu_count

#     arr = np.random.randn(16, 100000)
#     # no parallel
#     start = time()
#     tp.ts_std(arr, window=10, axis=1, par=False)
#     time1 = time() - start
#     # parallel
#     start = time()
#     tp.ts_std(arr, window=10, axis=1, par=True)
#     time2 = time() - start

#     if cpu_count() > 1:
#         assert time1 > time2


def test_special():
    # test rank a array with 0 element
    assert tp.rank(np.array([])).tolist() == []
    assert tp.rank(np.array([2]))[0] == 1  # test rank a array with 1 element
    # test rank all nan array
    assert_allclose(tp.rank(np.array([np.nan, np.nan])), np.array([np.nan, np.nan]))


# # currently only numpy array output is supported
# def test_array_func_input():
#     from pandas.testing import assert_frame_equal, assert_series_equal

#     value, expect = [3, 2, 1], [2, 1, 0]
#     # List input
#     assert tp.argsort(value).tolist() == expect
#     # Series input
#     sr = pd.Series(value, name="aa")
#     assert_series_equal(tp.argsort(sr), pd.Series(expect, name="aa"), check_dtype=False)
#     # DataFrame input
#     df = pd.DataFrame({"a": value})
#     assert_frame_equal(
#         tp.argsort(df, axis=0), pd.DataFrame({"a": expect}), check_dtype=False
#     )

# # currently only numpy array output is supported
# def test_ts_func_input():
#     from pandas.testing import assert_frame_equal, assert_series_equal

#     value, expect = [1, 1, 1], [1.0, 2, 3]
#     # list input
#     assert tp.ts_sum(value, 3).tolist() == expect
#     # series input
#     sr = pd.Series(value, name="aa")
#     assert_series_equal(tp.ts_sum(sr, 3), pd.Series(expect, name="aa"))
#     df = pd.DataFrame({"a": value})
#     assert_frame_equal(tp.ts_sum(df, 3, axis=0), pd.DataFrame({"a": expect}))

# # currently only numpy array output is supported
# def test_ts_func2_input():
#     from pandas.testing import assert_series_equal

#     value1, value2, expect = [1, 2, 3], [1, 2, 3], [np.nan, 1.0, 1.0]
#     # list input
#     assert_allclose(tp.ts_corr(value1, value2, 3), np.array(expect))
#     # series input
#     sr1, sr2 = pd.Series(value1, name="aa"), pd.Series(value1, name="bb")
#     assert_series_equal(tp.ts_corr(sr1, sr2, 3), pd.Series(expect, name="aa"))
