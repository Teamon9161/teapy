from functools import partial

import teapy as tp
from teapy import Expr, asexprs, nan

from .selector import Selector
from .tears import get_newey_west_adjust_s as _get_newey_west_adjust_s


def get_newey_west_adjust_s(x, resid, lag):
    if isinstance(x, Selector):
        return x.mod_func("regression.get_newey_west_adjust_s")(x, resid, lag)
    return _get_newey_west_adjust_s(x, resid, lag)


def mark_star(value_to_mark, t_value, p_value, precision=2, split="\r\n"):
    """用于标显著程度"""
    value_to_mark, t_value, p_value = asexprs((value_to_mark, t_value, p_value))
    value_to_mark = value_to_mark.round_string(precision)
    t_value = t_value.round_string(precision)
    value_to_mark = (
        value_to_mark.if_then((p_value <= 0.1) & (p_value > 0.05), value_to_mark + "*")
        .if_then((p_value <= 0.05) & (p_value > 0.01), value_to_mark + "**")
        .if_then((p_value <= 0.01) & (p_value >= 0), value_to_mark + "***")
        .if_then((p_value > 1) | (p_value < 0), "Invalid")
    )
    value_to_mark += f"{split}(" + t_value + ")"
    return value_to_mark


class Ols:
    def __init__(
        self,
        y,
        x,
        constant=True,
        dropna=True,
        calc_t=False,
        keep_shape=True,
        adjust_t=False,
        lag=None,
    ):
        """
        线性回归
        y: 因变量, 需是一维, 支持的类型: 能被转为Expr的所有类型
        x: 自变量, 支持的类型: 能被转为Expr的所有类型
        constant: 是否在回归时给x加上常数项
        dropna: 是否需要去掉nan, 如果无nan却去除nan会略微降低效率
        calc_t: 是否需要计算t值和p值, 由于不需要计算时可以直接使用lapack求解, 因此可以显著提高速度
        keep_shape: 返回的残差序列resid是否保留原有序列的长度(默认会先去除nan, 序列长度可能会改变
        adjust_t: 是否需要对回归的t值和p值进行Newey-West调整, 如果会True则默认calc_t也为True
        lag: Newey-West调整时的最大滞后阶数, 为None时使用最优滞后阶数的公式进行计算.
        """
        if not isinstance(y, tp.Selector) and not isinstance(x, tp.Selector):
            y, x = Expr(y), Expr(x)
        self.n_ori = y.shape[0]
        self.keep_shape = keep_shape
        self.dropna = dropna
        x = x.if_then(x.ndim() == 1, x.insert_axis(1))
        if dropna:
            nan_mask = y.is_nan() | x.is_nan().any(axis=1)
            y, x = y.filter(~nan_mask), x.filter(~nan_mask)
            if keep_shape:
                self.nan_mask = nan_mask
        self.y = y
        self.x = (
            tp.full(x.shape[0], 1.0).insert_axis(1).concat(x, axis=1) if constant else x
        )
        del y, x
        self.n = self.x.shape[0]
        self.k = self.x.shape[1]
        self.df = self.n - self.k
        if not calc_t and not adjust_t:
            self.ols_res = self.x.lstsq(self.y)
            self.rank = self.ols_res.ols_rank()
            self.params = self.ols_res.params()
            self.fitted_values = self.ols_res.fitted_values()
            self.SSE = self.ols_res.sse()
            self.resid = self.resid()
        else:
            pinv_x, s = self.x.pinv(return_s=True)
            self.rank = (s != 0).cast(int).sum()
            self.params = pinv_x @ self.y
            self.fitted_values = self.x @ self.params
            self.resid = self.resid()
            resid = self.y - self.fitted_values
            self.SSE = resid.t() @ resid

            if not adjust_t:
                self.tvalues = (
                    self.params
                    / ((pinv_x @ pinv_x.t()).diag() * self.SSE / self.df).sqrt()
                )
                self.pvalues = (1 - self.tvalues.abs().t_cdf(self.df)) * 2
            else:
                # Calculate Newey West t
                if lag is None:
                    lag = (4 * (self.n_ori / 100) ** (2 / 9)).ceil().cast(int)
                X = pinv_x @ pinv_x.t()
                # 计算Q矩阵的渐进估计S矩阵
                # 详细算法说明可参考 多因子回归检验中的 Newey-West调整
                # (https://zhuanlan.zhihu.com/p/54913149)
                S = get_newey_west_adjust_s(self.x, resid, Expr(lag)) / self.n
                # 计算t统计量和p统计量
                self.tvalues = self.params / (self.n * (X @ S @ X).diag()).sqrt()
                self.pvalues = (1 - self.tvalues.abs().norm_cdf()) * 2
            self.resid = (
                tp.full(self.nan_mask.shape, nan).put_mask(~self.nan_mask, resid)
                if keep_shape and dropna
                else resid
            )

    def resid(self):
        if self.keep_shape and self.dropna:
            resid = tp.full(self.nan_mask.shape, nan)
            return resid.put_mask(~self.nan_mask, self.y - self.fitted_values)
        else:
            return self.y - self.fitted_values

    def result(self, i=1, mark=False, multiplier=1, precision=4, split="\r\n"):
        """
        返回第i个位置的回归结果
        i: 返回第i个位置(从0开始)的自变量的回归结果
        mark: 是否需要直接返回标记星号的结果
        """
        if hasattr(self, "tvalues"):
            ret = [self.params[i] * multiplier, self.tvalues[i], self.pvalues[i]]
            for i, v in enumerate(ret):
                ret[i] = v.round(precision)
            return mark_star(*ret, precision=precision, split=split) if mark else ret
        else:
            return (self.params[i] * multiplier).round(precision)


# t统计量经过Newey West调整的线性回归
NwOls = partial(Ols, calc_t=True, adjust_t=True)

# 旧版本的api, 后面可能会弃用
sp_ols = partial(Ols, adjust_t=False)
nw_ols = NwOls


class ChowTest:
    """邹检验, 参考chowtest库, 注意原库进行f检验时自由度似乎写错了, 本函数已进行修正"""

    def __init__(self, y, x, x1_idx, x2_idx, constant=True):
        y, x, x1_idx, x2_idx = asexprs((y, x, x1_idx, x2_idx))
        res_all = Ols(y, x, constant=True)  # 全部数据的回归结果
        rss_all = res_all.resid.pow(2).sum()  # 总残差平方和rss
        x1, y1 = x[x1_idx], y[x1_idx]
        x2, y2 = x[x2_idx], y[x2_idx]
        res1, res2 = Ols(y1, x1, constant=constant), Ols(y2, x2, constant=constant)
        rss1, rss2 = res1.resid.pow(2).sum(), res2.resid.pow(2).sum()
        k = res_all.k  # 自变量个数
        N1 = res1.n  # 时间1的观测值数目
        N2 = res2.n  # 时间2的观测值数目
        numerator = (rss_all - (rss1 + rss2)) / k  # 邹统计量的分子
        denominator = (rss1 + rss2) / (N1 + N2 - 2 * k)  # 邹统计量的分母
        self.params = res1.params - res2.params
        self.chowvalues = numerator / denominator
        self.pvalues = 1 - self.chowvalues.f_cdf(df1=k, df2=(N1 + N2 - 2 * k))

    def result(self, i=1, mark=False, multiplier=1, precision=4, split="\r\n"):
        """
        返回第i个位置的回归结果
        i: 返回第i个位置(从0开始)的自变量的回归结果
        mark: 是否需要直接返回标记星号的结果
        """
        ret = [self.params[i] * multiplier, self.chowvalues, self.pvalues]
        for i, v in enumerate(ret):
            ret[i] = v.round(precision)
        return mark_star(*ret, precision=precision, split=split) if mark else ret


# =============================================================================
# test
# =============================================================================
if __name__ == "__main__":
    import numpy as np
    import statsmodels.api as sm
    from numpy.testing import assert_allclose

    def sp_ols_sm(y, x, constant=True):
        """statsmodels的ols回归函数, 效率较低, constant为True则加常数项"""
        x = np.asanyarray(x)
        x = np.vstack([np.ones(x.shape[0]), x.T]).T if constant else x
        return sm.OLS(np.asanyarray(y), x, missing="drop").fit()

    def nw_ols_sm(y, x, lag=None, constant=True):
        """statsmodels实现的ols函数, 效率较低, constant为True则加常数项, lag: 滞后阶数"""
        nlag = int(np.ceil(4 * (y.size / 100) ** (2 / 9))) if lag is None else lag
        x = np.asanyarray(x)
        x = np.vstack([np.ones(x.shape[0]), x.T]).T if constant else x
        return sm.OLS(np.asanyarray(y), x, missing="drop").fit(
            cov_type="HAC", cov_kwds={"maxlags": nlag}
        )

    # 结果准确性测试
    y = np.array([1, 4, 6, 1, np.nan, 43])
    x = np.array([[5, 6, 12, 5, 7, 2], [7, np.nan, 7, 1, 4, 8]]).T

    def test_ols_result(res1, res2):
        assert_allclose(res1.params.eview(), res2.params)
        assert_allclose(res1.resid.eview(), res2.resid)
        assert_allclose(res1.fitted_values.eview(), res2.fittedvalues)
        assert_allclose(res1.tvalues.eview(), res2.tvalues)

    res1 = sp_ols(y, x, constant=True, calc_t=True, keep_shape=False)
    res2 = sp_ols_sm(y, x, constant=True)
    test_ols_result(res1, res2)

    res3 = sp_ols(y, x, constant=False, calc_t=True, keep_shape=False)
    res4 = sp_ols_sm(y, x, constant=False)
    test_ols_result(res3, res4)

    res5 = nw_ols(y, x, constant=True, keep_shape=False)
    res6 = nw_ols_sm(y, x, constant=True)
    test_ols_result(res5, res6)

    res7 = nw_ols(y, x, constant=False, keep_shape=False)
    res8 = nw_ols_sm(y, x, constant=False)
    test_ols_result(res7, res8)

    res9 = nw_ols(y, x, constant=True, lag=2, keep_shape=False)
    res10 = nw_ols_sm(y, x, constant=True, lag=2)
    test_ols_result(res7, res8)
    print("结果准确性检验通过")
