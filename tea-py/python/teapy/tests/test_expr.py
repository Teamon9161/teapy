import numpy as np
import pandas as pd
import teapy as tp
from teapy.testing import assert_allclose, assert_series_equal


def test_name():
    e = tp.Expr([1, 2])
    assert e.name is None
    e = tp.Expr([1, 2], name="a")
    assert e.name == "a"
    e.alias("b", inplace=True)
    assert e.name == "b"
    e = e.alias("c")
    assert e.name == "c"
    e1 = e.suffix("_suffix")
    assert e1.name == "c_suffix"
    e.prefix("prefix_", inplace=True)
    assert e.name == "prefix_c"


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
    e[0] = 100
    assert_allclose(e.eview(), np.array([100, 2, 3, 4, 5, 6, 7, 8]))


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


def test_isin():
    e = tp.Expr(["a", "bd", "sdf", "bdfd", "ab"])
    assert e.filter(~e.is_in(["ab", "a"])).eview().tolist() == ["bd", "sdf", "bdfd"]


def test_rolling():
    time = tp.Expr(
        pd.date_range("2020-01-01 04:00:00", "2020-01-5 00:00:00", freq="4H").values
    )
    value = tp.Expr(
        [5, 5, 9, 5, 8, 8, 2, 7, 3, 3, 8, 6, 4, 7, 8, 3, 1, 5, 3, 4, 4, 7, 9, 3]
    )
    res1 = value.rolling("12h", time_expr=time).min().eval()
    expect1 = [5, 5, 5, 5, 5, 5, 2, 2, 2, 3, 3, 3, 4, 4, 4, 3, 1, 1, 1, 3, 3, 4, 4, 3]
    assert_allclose(res1.view, expect1)
    res2 = value.rolling("12h", time_expr=time, start_by="duration_start").min().eval()
    expect2 = [5, 5, 9, 5, 5, 8, 2, 2, 3, 3, 3, 6, 4, 4, 8, 3, 1, 5, 3, 3, 4, 4, 4, 3]
    assert_allclose(res2.view, expect2)

    time = tp.Expr(
        pd.concat(
            [
                pd.date_range(
                    "2020-01-01 04:00:00", "2020-01-03 00:00:00", freq="4H"
                ).to_series(),
                pd.date_range(
                    "2020-01-05 08:00:00", "2020-01-07 00:00:00", freq="4H"
                ).to_series(),
            ]
        ).values
    )

    value = tp.Expr(
        [5, 5, 9, 5, 8, 8, 2, 7, 3, 3, 8, 6, 4, 7, 8, 3, 1, 5, 3, 4, 4, 2, 4]
    )

    res3 = value.rolling(time_expr=time, window="3d", offset="1d").sum().eval()
    expect3 = [
        5,
        5,
        9,
        5,
        8,
        8,
        7,
        12,
        12,
        8,
        16,
        14,
        11,
        10,
        11,
        11,
        7,
        5,
        7,
        11,
        12,
        5,
        5,
    ]
    assert_allclose(res3.view, expect3)

    time = tp.Expr(pd.date_range("2020-01-01", "2020-05-05", freq="8d").values)
    value = tp.Expr([1, 9, 3, 1, 5, 4, 3, 8, 1, 1, 5, 2, 6, 8, 7, 9])
    res4 = value.rolling("1mo", time_expr=time).min().eval()
    expect4 = [1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 2, 6]
    assert_allclose(res4.view, expect4)
    res5 = value.rolling("1mo", time_expr=time, start_by="duration_start").max().eval()
    expect5 = [1, 9, 9, 9, 5, 5, 5, 8, 1, 1, 5, 5, 6, 8, 8, 9]
    assert_allclose(res5.view, expect5)


def test_group_by_time():
    time = tp.Expr(
        pd.date_range("2020-01-01 04:00:00", "2020-01-5 00:00:00", freq="4H").values
    )
    value = tp.Expr(
        [5, 5, 9, 5, 8, 8, 2, 7, 3, 3, 8, 6, 4, 7, 8, 3, 1, 5, 3, 4, 4, 7, 9, 3]
    )
    for closed in ["left", "right"]:
        df = pd.DataFrame({"time": time.view, "value": value.view})
        df_pd = df.set_index("time").resample("12h", closed=closed).sum()

        label, v = value.groupby("12h", time_expr=time, closed=closed).agg(
            tp.s(0).sum()
        )
        label = label.eview()
        v = v.eview()
        assert_allclose(v, df_pd["value"])
        assert_series_equal(pd.Series(label), df_pd.index.to_series())
