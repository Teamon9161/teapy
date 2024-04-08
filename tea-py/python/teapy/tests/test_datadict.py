import numpy as np
import pandas as pd
import teapy as tp
from numpy.testing import assert_array_equal
from teapy import Expr, get_align_frames_idx, s
from teapy.py_datadict import DataDict
from teapy.testing import assert_allclose, assert_allclose3


def test_memory():
    a = np.random.randn(1000, 1000)
    b = a.copy()
    dd = DataDict(a=a)
    dd = dd.with_columns(dd["a"].t().t().insert_axis(1)[:, 0, :].alias("b"))
    del a
    assert_allclose(dd["a"].view, b)
    e = dd["b"]
    del dd
    e = e.eval()
    assert_allclose(e.view, b)


def test_init():
    name_auto_prefix = "column_"
    dd = DataDict()
    dd = DataDict({"a": [2], "b": [45]})
    assert dd.columns == ["a", "b"]
    dd = DataDict({"a": [2], "b": [45]}, columns=["c", "d"])
    assert dd.columns == ["c", "d"]  # override
    dd = DataDict([[5], [7]])
    assert dd.columns == [name_auto_prefix + "0", name_auto_prefix + "1"]
    dd = DataDict([[5], [7]], columns=["0", "1"], a=[34, 5], b=[3])
    assert dd.columns == ["0", "1", "a", "b"]
    dd = DataDict({"a": [2], "b": [45]}, c=[34, 5])
    assert dd.columns == ["a", "b", "c"]
    ea, eb = Expr([1]).alias("a"), Expr([2]).alias("b")
    assert DataDict([ea, eb]).columns == ["a", "b"]
    assert DataDict([ea, eb], columns=["c", "d"]).columns == ["c", "d"]


def test_rename():
    dd = DataDict({"a": [2], "b": [45]})
    dd.rename({"a": "c"}, inplace=True)
    assert set(dd.columns) == {"c", "b"}
    dd = dd.rename({"b": "a"})
    assert set(dd.columns) == {"c", "a"}
    dd = dd.rename(["a", "b"])
    assert dd.columns == ["a", "b"]


def test_get_and_set_item():
    dd = DataDict()
    a = np.random.randn(100)
    dd["a"] = a
    assert_allclose3(a, dd["a"].view, dd[0].view)
    b = np.random.randn(100)
    dd["b"] = b
    assert dd[["a", "b"]].columns == dd[[0, 1]].columns == ["a", "b"]

    dd[0] = b
    assert_allclose(dd[0].view, b)
    # Expr in column 0 will be renamed
    assert dd["a"].name == "a"
    dd[["cdsf", "adf"]] = [[4], [2]]
    assert dd["^a.*$"].columns == ["a", "adf"]
    assert dd[["b", "^a.*$"]].columns == ["b", "a", "adf"]

    dd["^a.*$"] = dd["^a.*$"].apply(lambda e: e * 2)
    dd.eval()
    assert dd["adf"].view == 4
    assert_allclose(dd["a"].view, 2 * b)
    dd[["^a.*$", "cdsf"]] = dd[["^a.*$", "cdsf"]].apply(lambda e: e / 2)
    assert dd["cdsf"].eview() == 2


def test_drop():
    dd = DataDict([np.random.randn(10), np.random.randn(10)], columns=["a", "b"])
    assert dd.drop("a").columns == ["b"]
    assert dd.drop(["a", "b"]).columns == []
    dd.drop("b", inplace=True)
    assert dd.columns == ["a"]
    del dd["a"]
    assert dd.columns == []


def test_to_dict():
    data = {"a": 1, "b": 2}
    dd = DataDict(data)
    assert dd.to_dict() == data
    dd = dd.with_columns((s("a") * 2).alias("c"))
    assert dd.to_dict() == {"a": 1, "b": 2, "c": 2}


def test_dropna():
    dd = DataDict(
        {
            "a": [1, 2, np.nan, 3, 2, np.nan],
            "b": [np.nan, 3, np.nan, 4, np.nan, 5],
        }
    )

    # dropna and return a new datadict
    new_dd = dd.dropna(how="all").eval(inplace=False)
    assert_allclose(new_dd["b"].view, np.array([np.nan, 3, 4, np.nan, 5]))
    new_dd = dd.dropna(how="any").eval(inplace=False)
    assert_allclose(new_dd["a"].view, [2, 3])
    new_dd = dd.dropna(subset=1, how="any").eval(inplace=False)
    assert_allclose(new_dd["b"].view, [3, 4, 5])

    # dropna inplace
    dd.dropna(how="any", inplace=True)
    assert_allclose(dd["a"].eview(), [2, 3])


def test_copy():
    dd = DataDict(a=[4, 2, 3, 1], b=[1, 2, 3, 4])
    dd1 = dd.copy()
    dd1.sort(["a"], inplace=True)
    dd1.eval(inplace=True)
    assert_allclose(dd1["b"].view, [4, 2, 3, 1])
    assert_allclose(dd["b"].view, [1, 2, 3, 4])


def test_sort():
    from teapy import nan

    dd = DataDict(a=[4, 2, 3, 1], b=[1, 2, 3, 4])
    assert_allclose(dd["b"].sort(dd["a"]).eview(), [4, 2, 3, 1])

    dd = dd.sort(["a"]).eval(inplace=False)
    assert_allclose(dd["b"].view, [4, 2, 3, 1])

    dd = dd.sort(["a"], rev=True).eval(inplace=False)
    assert_allclose(dd["b"].view, [1, 3, 2, 4])

    dd = DataDict(a=[4, 2, nan, 1], b=[1, 2, 3, 4])
    dd.sort(["a"], inplace=True)
    dd.eval(inplace=True)
    assert_allclose(dd["b"].view, [4, 2, 1, 3])


def test_mean():
    dd = DataDict(a=[1, 2, 3, 4], b=[3, 4, 5, 6])
    assert_allclose(dd.mean(axis=-1).eview(), np.array([2, 3, 4, 5]))
    assert dd.mean(axis=0)["a"].eview() == 2.5


def test_sum():
    dd = DataDict(a=[1, 2, 3, 4], b=[3, 4, 5, 6])
    assert_allclose(dd.sum(axis=-1).eview(), np.array([4, 6, 8, 10]))
    assert dd.sum(axis=0)["a"].eview() == 10


def test_min():
    dd = DataDict(a=[1, 2, 3, 4], b=[3, 4, 5, 6])
    assert_allclose(dd.min(axis=-1).eview(), np.array([1, 2, 3, 4]))
    assert dd.min(axis=0)["a"].eview() == 1


def test_max():
    dd = DataDict(a=[1, 2, 3, 4], b=[3, 4, 5, 6])
    assert_allclose(dd.max(axis=-1).eview(), np.array([3, 4, 5, 6]))
    assert dd.max(axis=0)["a"].eview() == 4


def test_corr():
    for method in ["pearson", "spearman"]:
        dd = DataDict({"a": np.random.randn(100), "b": np.random.randn(100)})
        df = dd.to_pd()
        assert_allclose(dd.corr(method=method).eview(), df.corr(method=method).values)


def test_dtypes():
    dd = DataDict(
        a=np.random.randint(1, 3, 3).astype(np.int32),
        b=[1.0, 2.0, 3.0],
        c=["df", "134", "231"],
    )
    assert dd.dtypes == {"a": "I32", "b": "F64", "c": "String"}


def test_join():
    ldd = DataDict({"left_on": ["a", "b", "a", "c"], "va": [1, 2, 3, 4]})
    rdd = DataDict({"right_on": ["b", "b", "c"], "vb": [10, 20, 30]})
    dd = ldd.join(rdd, left_on="left_on", right_on="right_on", how="left")
    dd2 = ldd.join(rdd, left_on="left_on", right_on="right_on", how="right")

    assert_allclose(dd["vb"].eview(), np.array([np.nan, 20, np.nan, 30]))
    assert_allclose(dd2["va"].eview(), np.array([2, 2, 4]))
    ldd = ldd.rename({"left_on": "on"})
    rdd = rdd.rename({"right_on": "on"})
    dd = ldd.join(rdd, on="on", how="left")
    assert_allclose(dd["vb"].eview(), np.array([np.nan, 20, np.nan, 30]))
    # inplace join
    ldd.join(rdd, on="on", how="left", inplace=True)
    assert_allclose(ldd["vb"].eview(), np.array([np.nan, 20, np.nan, 30]))
    ldd = DataDict({"left_on": ["a", "b", "a", "c"], "va": [1, 2, 3, 4]})
    rdd = DataDict({"right_on": ["b", "b", "c"], "vb": [10, 20, 30]})
    ldd.join(rdd, left_on="left_on", right_on="right_on", how="right", inplace=True)
    assert_allclose(rdd["va"].eview(), np.array([2, 2, 4]))

    ldd = DataDict({"on": ["a", "b", "d", "c"], "va": [1, 2, 3, 4]})
    rdd = DataDict({"on": ["b", "a", "e"], "vb": [10, 20, 30]})
    dd = ldd.join(rdd, how="outer", on="on").eval()
    assert_allclose(dd["va"].eview(), np.array([1, 2, 4, 3, np.nan]))
    assert_allclose(dd["vb"].eview(), np.array([20, 10, np.nan, np.nan, 30]))
    assert_array_equal(dd["on"].eview(), np.array(["a", "b", "c", "d", "e"]))

    by, idxs = get_align_frames_idx([ldd, rdd], by="on", return_by=True)
    assert_allclose(ldd["va"].select(idxs[0]).eview(), np.array([1, 2, 4, 3, np.nan]))
    assert_allclose(
        rdd["vb"].select(idxs[1]).eview(), np.array([20, 10, np.nan, np.nan, 30])
    )
    assert_array_equal(by[0].eview(), np.array(["a", "b", "c", "d", "e"]))


def test_groupby():
    n = 100
    dd = DataDict(
        {
            "g": ["e", "e"] + ["a", "b", "a", "a", "c"] * n + ["d"],
            "v": [-0.1, -5, *np.random.randn(5 * n).tolist(), 1],
        }
    )
    df = dd.to_pd()
    assert_allclose(
        dd.groupby("g").agg(s(1).max().alias("v"))["v"].eview(),
        df.groupby("g", sort=False).v.max(),
    )

    dd = DataDict(
        {
            "a": ["a", "b", "a", "b", "b", "c"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )

    res = dd.groupby("a").agg(
        s("c").sum().alias("c1"),
        c="max",
    )
    assert_allclose(res["c1"].eview(), np.array([10, 10, 1]))
    assert_allclose(res["c"].eview(), np.array([6, 5, 1]))


def test_unique():
    dd = DataDict(
        a=[1, 1, 3, 4, 5, 3],
        b=["a", "b", "b", "c", "d", "b"],
        c=[1.23, -4.234, 4.234, 2.13, -4.234, 1.23],
        v=[1, 2, 3, 4, 5, 6],
    )
    assert_allclose(dd.unique("c")["v"].eview(), [1, 2, 3, 4])
    assert_allclose(dd.unique("a")["v"].eview(), [1, 3, 4, 5])
    assert_allclose(dd.unique(["a", "b"], keep="last")["v"].eview(), [1, 2, 4, 5, 6])
    assert_allclose(dd.unique(["a", "b"], keep="first")["v"].eview(), [1, 2, 3, 4, 5])
