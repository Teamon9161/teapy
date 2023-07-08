import numpy as np
import pandas as pd

from teapy import DataDict, Expr
from teapy.testing import assert_allclose, assert_allclose3


def test_init():
    dd = DataDict()
    dd = DataDict({"a": [2], "b": [45]})
    assert dd.columns == ["a", "b"]
    dd = DataDict({"a": [2], "b": [45]}, columns=["c", "d"])
    assert dd.columns == ["c", "d"]  # override
    dd = DataDict([[5], [7]])
    assert dd.columns == ["0", "1"]
    dd = DataDict([[5], [7]], columns=["0", "1"], a=[34, 5], b=[3])
    assert dd.columns == ["0", "1", "a", "b"]
    dd = DataDict({"a": [2], "b": [45]}, c=[34, 5])
    assert dd.columns == ["a", "b", "c"]
    ea, eb = Expr([1]).alias("a"), Expr([2]).alias("b")
    assert DataDict([ea, eb]).columns == ["a", "b"]
    assert DataDict([ea, eb], columns=["c", "d"]).columns == ["c", "d"]


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


def test_dtypes():
    dd = DataDict(
        a=np.random.randint(1, 3, 3).astype(np.int32),
        b=[1.0, 2.0, 3.0],
        c=["df", "134", "231"],
    )
    assert dd.dtypes == {"a": "Int32", "b": "Float64", "c": "String"}


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


def test_groupby():

    dd = DataDict(
        {
            "g": ["e", "e"] + ["a", "b", "a", "a", "c"] * 100 + ["d"],
            "v": [-0.1, -5] + np.random.randn(500).tolist() + [1],
        }
    )
    df = pd.DataFrame(dd.to_dict())
    assert_allclose(
        dd.groupby("g").apply(lambda df: df["v"].max())["v"].eview(),
        df.groupby("g", sort=False).v.max(),
    )

    dd = DataDict(
        {
            "a": ["a", "b", "a", "b", "b", "c"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )
    # res = dd.groupby("a").apply(lambda df: df["c"].sum())
    # assert_allclose(res["c"].eview(), np.array([10, 10, 1]))

    res = dd.groupby("a").apply(
        lambda df: [df["c"].sum().alias("c1"), df["c"].max().alias("c2")]
    )
    assert_allclose(res["c2"].eview(), np.array([6, 5, 1]))
