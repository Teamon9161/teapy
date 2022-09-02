import teapy as tp
import numpy as np
import pandas as pd
# import polars as pl
# from teamon import tease as ts

# a = np.array([
#     [1.,2,np.nan,np.nan,5,5,7,8,9],
#     [1.,2,np.nan,4,5,5,7,8,9]
# ], order='f')

# a = np.array([np.nan, np.nan, np.nan, 0, 5, 6, 4, np.nan, 2, 6, 123, 0, 4, 5, 1, np.nan, np.nan, np.nan])
# b = np.array(np.random.randn(8, 1000000), order='c').astype(float)
# bb = pd.DataFrame(b)

#  c = pl.DataFrame({'a': a}).fill_nan(None).lazy()
# a=np.array([1,1,5,np.nan,1,1,1]).astype(float)

# a = np.array([4,6,np.nan,5,3,4,np.nan, 55, 21, 123, 325, 1233, 5])

# tp.rank_pct(a, axis=1, par=True)
# ts.rank(a)
# pd.Series(a).rank(pct=True)
# a.argsort()


# b = b[0, :]
# bb = pd.DataFrame(b)
# timeit tp.ts_sma(b, 10, axis=1, par=False)
# timeit ts.ts_sma(b, 10, axis=1)
# timeit b.argsort(axis=1)


'''
if __name__ == '__main__':
    from numpy.testing import assert_allclose
    from functools import partial
    assert_allclose = partial(assert_allclose, rtol=1e-5, atol=1e-6)

    
    def get_arr(length: int, nan_ret: float=0.1):
        arr = np.random.randn(length)
        nan_idx = np.random.randint(0, length, int(nan_ret*length)) # 1000 nan idx
        arr[nan_idx] = np.nan
        return arr
    arr_list = [get_arr(3000), get_arr(3000), np.random.randint(0, 1000, 3000)]

    for window in [4, 10, 20, 50]:
        for min_periods in [1, 2, window]:
            for arr in arr_list:
                
                # 测试argsort(如果有重复值或nan的话排序不稳定，最后的排序不会一致)
                if arr.dtype != np.int32:
                    res1 = tp.argsort(arr[~np.isnan(arr)])
                    res2 = arr[~np.isnan(arr)].argsort()
                    assert_allclose(res1, res2)
                
                # 测试rank
                res1 = tp.rank(arr)
                res2 = pd.Series(arr).rank().values
                assert_allclose(res1, res2)
                
                res1 = tp.rank(arr, pct=True)
                res2 = pd.Series(arr).rank(pct=True).values
                assert_allclose(res1, res2)
                
                # 测试移动平均
                res1 = tp.ts_sma(arr, window, min_periods=min_periods)
                res2 = pd.Series(arr).rolling(window, min_periods=min_periods).mean().values
                assert_allclose(res1, res2)
                
                # 测试移动求和
                res1 = tp.ts_sum(arr, window, min_periods=min_periods)
                res2 = pd.Series(arr).rolling(window, min_periods=min_periods).sum().values
                assert_allclose(res1, res2)
            
                # 测试移动连乘
                res1 = tp.ts_prod(arr, window, min_periods=min_periods)
                res2 = pd.Series(arr).rolling(window, min_periods=min_periods).apply(lambda x: x.prod()).values
                assert_allclose(res1, res2)
                
                # 测试移动几何平均
                res1 = tp.ts_prod_mean(arr, window, min_periods=min_periods)
                res2 = pd.Series(arr).rolling(window, min_periods=min_periods).apply(
                    lambda x: x.prod() ** (1 / x.notnull().sum())).values
                assert_allclose(res1, res2)
                
                # 测试移动最大值
                res1 = tp.ts_max(arr, window, min_periods=min_periods)
                res2 = pd.Series(arr).rolling(window, min_periods=min_periods).max().values
                assert_allclose(res1, res2)
                
                # 测试移动最小值
                res1 = tp.ts_min(arr, window, min_periods=min_periods)
                res2 = pd.Series(arr).rolling(window, min_periods=min_periods).min().values
                assert_allclose(res1, res2)
                
                # 测试移动最小值索引(在window中)
                if arr.dtype != np.int32: # 整数arr可能有重复值，对于重复值总是取最后一个，而pandas取的是第一个
                    res1 = tp.ts_argmin(arr, window, min_periods=min_periods)
                    res2 = pd.Series(arr).rolling(window, min_periods=min_periods).apply(
                        lambda x: x.idxmin() - x.index[0] + 1
                    ).values
                    assert_allclose(res1, res2)
                
                # 测试移动最大值索引(在window中)
                if arr.dtype != np.int32: # 整数arr可能有重复值，对于重复值总是取最后一个，而pandas取的是第一个
                    res1 = tp.ts_argmax(arr, window, min_periods=min_periods)
                    res2 = pd.Series(arr).rolling(window, min_periods=min_periods).apply(
                        lambda x: x.idxmax() - x.index[0] + 1
                    ).values
                    assert_allclose(res1, res2)  
                
                # 测试移动标准差
                res1 = tp.ts_std(arr, window, min_periods=min_periods)
                res2 = pd.Series(arr).rolling(window, min_periods=min_periods).std().values
                assert_allclose(res1, res2)  
                
                # 测试移动偏度
                res1 = tp.ts_skew(arr, window, min_periods=min_periods)
                res2 = pd.Series(arr).rolling(window, min_periods=min_periods).skew().values
                assert_allclose(res1, res2)
                
                for i in range(3000):
                    if abs(res1[i]-res2[i]) >= 1e-5:
                        print(i)
                
                # 测试移动峰度
                res1 = tp.ts_kurt(arr, window, min_periods=min_periods)
                res2 = pd.Series(arr).rolling(window, min_periods=min_periods).kurt().values
                assert_allclose(res1, res2) 
                
                # 测试移动排名(pandas is slow, so only 1000)
                res1 = tp.ts_rank(arr[:1000], window, min_periods=min_periods)
                res2 = pd.Series(arr[:1000]).rolling(window, min_periods=min_periods).apply(lambda x: x.rank().iloc[-1]).values
                assert_allclose(res1, res2) 
                
                # 测试移动stable标准化
                res1 = tp.ts_stable(arr[:1000], window, min_periods=min_periods)
                res2 = pd.Series(arr[:1000]).rolling(window, min_periods=min_periods).apply(
                    lambda x: x.mean() / x.std()
                ).values
                assert_allclose(res1, res2) 
                
                # 测试移动meanstd标准化
                res1 = tp.ts_meanstdnorm(arr[:1000], window, min_periods=min_periods)
                res2 = pd.Series(arr[:1000]).rolling(window, min_periods=min_periods).apply(
                    lambda x: (x.iloc[-1]-x.mean()) / x.std()
                ).values
                assert_allclose(res1, res2) 
                
                # 测试移动minmax标准化
                res1 = tp.ts_minmaxnorm(arr[:1000], window, min_periods=min_periods)
                res2 = pd.Series(arr[:1000]).rolling(window, min_periods=min_periods).apply(
                    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min())
                ).values
                assert_allclose(res1, res2)
                
                # 测试ewm和wma
                def ewm(s):
                    alpha = 2 / window
                    n = s.count()
                    if n > 0:
                        weight = np.logspace(n-1, 0, num=n, base=(1-alpha))
                        weight /= weight.sum()
                        return (weight * s[~s.isna()]).sum()
                    else:
                        return np.nan
                res1 = tp.ts_ewm(arr, window, min_periods=min_periods)
                res2 = pd.Series(arr).rolling(window, min_periods=min_periods).apply(ewm)
                
                def wma(s):
                    n = s.count()
                    if n > 0:
                        weight = np.arange(n) + 1
                        weight = weight / weight.sum()
                        return (weight * s[~s.isna()]).sum()
                    else:
                        return np.nan
                res1 = tp.ts_wma(arr, window, min_periods=min_periods)
                res2 = pd.Series(arr).rolling(window, min_periods=min_periods).apply(wma)    
                
                # 测试移动回归
                import statsmodels.api as sm
                
                def ts_reg(s):
                    s = s.dropna()
                    if s.size > 1:
                        reg = sm.OLS(s, sm.add_constant(np.arange(s.size)+1)).fit()
                        return reg.params[0] + reg.params[1] * s.size
                    else:
                        return np.nan
                res1 = tp.ts_minmaxnorm(arr[:100], window, min_periods=min_periods)
                res2 = pd.Series(arr[:100]).rolling(window, min_periods=min_periods).apply(ts_reg)
                
                def ts_reg_intercept(s):
                    s = s.dropna()
                    if s.size > 1:
                        reg = sm.OLS(s, sm.add_constant(np.arange(s.size)+1)).fit()
                        return reg.params[0]
                    else:
                        return np.nan
                res1 = tp.ts_reg_intercept(arr[:100], window, min_periods=min_periods)
                res2 = pd.Series(arr[:100]).rolling(window, min_periods=min_periods).apply(ts_reg_intercept)
                
                def ts_reg_slope(s):
                    s = s.dropna()
                    if s.size > 1:
                        reg = sm.OLS(s, sm.add_constant(np.arange(s.size)+1)).fit()
                        return reg.params[1]
                    else:
                        return np.nan
                res1 = tp.ts_reg_slope(arr[:100], window, min_periods=min_periods)
                res2 = pd.Series(arr[:100]).rolling(window, min_periods=min_periods).apply(ts_reg_slope)
                
                def ts_tsf(s):
                    s = s.dropna()
                    if s.size > 1:
                        reg = sm.OLS(s, sm.add_constant(np.arange(s.size)+1)).fit()
                        return reg.params[0] + reg.params[1] * (s.size + 1)
                    else:
                        return np.nan
                res1 = tp.ts_tsf(arr[:100], window, min_periods=min_periods)
                res2 = pd.Series(arr[:100]).rolling(window, min_periods=min_periods).apply(ts_tsf)
            
            # 测试移动协方差
            res1 = tp.ts_cov(arr_list[0], arr_list[1], window, min_periods=min_periods)
            res2 = pd.Series(arr_list[0]).rolling(window, min_periods=min_periods).cov(pd.Series(arr_list[1]))
            assert_allclose(res1, res2.replace([np.inf, -np.inf], np.nan).values) 
            
            # 测试移动相关性
            res1 = tp.ts_corr(arr_list[0], arr_list[1], window, min_periods=min_periods)
            res2 = pd.Series(arr_list[0]).rolling(window, min_periods=min_periods).corr(pd.Series(arr_list[1]))
            assert_allclose(res1, res2.replace([np.inf, -np.inf], np.nan).values) 
    
    # 测试连续性
    a = np.array(np.random.randn(100, 20), order='c')
    assert tp.ts_sma(a, window=3).flags['C_CONTIGUOUS'] == True
    a = np.array(np.random.randn(100, 20), order='f')
    assert tp.ts_sma(a, window=3).flags['F_CONTIGUOUS'] == True
    print('测试通过')
'''