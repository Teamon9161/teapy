use core::prelude::*;
// use core::datatype::Number;

#[cfg(all(feature = "agg", feature = "arr_func"))]
use crate::WinsorizeMethod;
#[cfg(feature = "agg")]
use crate::{CorrMethod, QuantileMethod};

#[cfg(any(feature = "agg", feature = "arr_func"))]
macro_rules! auto_impl_view {
    (in1, [$($func: ident),* $(,)?], $other: tt) => {
        $(auto_impl_view!(in1, $func, $other);)*
    };
    (in1, $func:ident, ($($p:ident: $p_ty:ty),* $(,)?)) => {
        impl<'a> Expr<'a> {
            pub fn $func(&mut self $(, $p: $p_ty)*) -> &mut Self {
                self.chain_f_ctx(move |(data, ctx)| {
                    let arr = data.view_arr(ctx.as_ref())?;
                    match_arrok!(numeric arr, a, { Ok((a.view().$func($($p),*).into(), ctx)) })
                });
                self
            }
        }
    };
    (in2, [$($func: ident),* $(,)?], $other: tt) => {
        $(auto_impl_view!(in2, $func, $other);)*
    };
    (in2, $func:ident, ($($p:ident: $p_ty:ty),* $(,)?)) => {
        impl<'a> Expr<'a> {
            pub fn $func(&mut self, other: Expr<'a>, $($p: $p_ty),*) -> &mut Self {
                self.chain_f_ctx(move |(data, ctx): FuncOut<'a>| {
                    // other.eval_inplace(ctx.clone())?;
                    let other_arr = other.view_arr(ctx.as_ref())?;
                    let arr = data.view_arr(ctx.as_ref())?;
                    match_arrok!(
                        (arr, a, F64, F32, I64, I32),
                        (other_arr, o, F64, F32, I64, I32),
                        {
                            Ok((a.view().$func(&o.view(), $($p),*).into(), ctx))
                        }
                    )
                });
                self
            }
        }
    };
}

#[cfg(feature = "arr_func")]
macro_rules! auto_impl_viewmut {
    (in1, [$($func: ident),* $(,)?], $other: tt) => {
        $(auto_impl_viewmut!(in1, $func, $other);)*
    };
    (in1, $func:ident, ($($p:ident: $p_ty:ty),* $(,)?)) => {
        impl<'a> Expr<'a> {
            pub fn $func(&mut self, $($p: $p_ty),*) -> &mut Self {
                self.chain_f_ctx(move |(data, ctx)| {
                    let mut arr = data.into_arr(ctx.clone())?;
                    match_arrok!(numeric &mut arr, a, {
                        a.view_mut().$func($($p),*);
                    });
                    Ok((arr.into(), ctx))
                });
                self
            }
        }
    };
}

#[cfg(feature = "arr_func")]
macro_rules! auto_impl_f64_func {
    ([$($func: ident),* $(,)?], $other: tt) => {
        $(auto_impl_f64_func!($func, $other);)*
    };
    ($func:ident, ($($p:ident: $p_ty:ty),* $(,)?)) => {
        impl<'a> Expr<'a> {
            pub fn $func(&mut self $(, $p: $p_ty)*) -> &mut Self{
                self.chain_f_ctx(move |(data, ctx)| {
                    let arr = data.view_arr(ctx.as_ref())?;
                    match_arrok!(numeric arr, a, { Ok((a.view().map(|v| v.f64().$func($($p),*)).into(), ctx)) })
                });
                self
            }
        }
    };
}

#[cfg(feature = "arr_func")]
auto_impl_f64_func!(
    [
        sqrt,
        cbrt,
        ln,
        ln_1p,
        log2,
        log10,
        exp,
        exp2,
        exp_m1,
        acos,
        asin,
        atan,
        sin,
        cos,
        tan,
        ceil,
        floor,
        fract,
        trunc,
        is_finite,
        is_infinite,
    ],
    ()
);
#[cfg(feature = "arr_func")]
auto_impl_f64_func!([log], (base: f64));
#[cfg(feature = "arr_func")]
auto_impl_view!(in1, [is_nan, not_nan], ());
#[cfg(feature = "agg")]
auto_impl_view!(in1, [ndim], ());
// auto_impl_view!(in1, [abs], (par: bool));
#[cfg(feature = "arr_func")]
auto_impl_view!(in1, [diff, pct_change], (n: i32, axis: i32, par: bool));
#[cfg(feature = "agg")]
auto_impl_view!(in1, 
    [
        count_nan, count_notnan, median, max, min, prod, 
        valid_last, valid_first, first, last,
    ], (axis: i32, par: bool));
#[cfg(feature = "arr_func")]
auto_impl_view!(in1, [cumprod], (axis: i32, par: bool));
#[cfg(feature = "arr_func")]
auto_impl_view!(in1, [cumsum], (stable: bool, axis: i32, par: bool));
#[cfg(feature = "agg")]
auto_impl_view!(in1, [sum], (stable: bool, axis: i32, par: bool));

#[cfg(feature = "agg")]
auto_impl_view!(in1, [mean, var, std, skew, kurt], (min_periods: usize, stable: bool, axis: i32, par: bool));
#[cfg(feature = "agg")]
auto_impl_view!(in1, quantile, (q: f64, method: QuantileMethod, axis: i32, par: bool));
#[cfg(all(feature = "arr_func", feature = "agg"))]
auto_impl_view!(in1, rank, (pct: bool, rev: bool, axis: i32, par: bool));
#[cfg(feature = "arr_func")]
auto_impl_view!(in1, argsort, (rev: bool, axis: i32, par: bool));
#[cfg(all(feature = "arr_func", feature = "agg"))]
auto_impl_view!(in1, split_group, (group: usize, rev: bool, axis: i32, par: bool));
#[cfg(all(feature = "arr_func", feature = "agg"))]
auto_impl_view!(in1, [arg_partition, partition], (kth: usize, sort: bool, rev: bool, axis: i32, par: bool));
#[cfg(all(feature = "arr_func", feature = "agg"))]
auto_impl_viewmut!(in1, [zscore_inplace], (min_periods: usize, stable: bool, axis: i32, par: bool));
#[cfg(all(feature = "agg", feature = "arr_func"))]
auto_impl_viewmut!(in1, [winsorize_inplace], (method: WinsorizeMethod, method_params: Option<f64>, stable: bool, axis: i32, par: bool));
#[cfg(feature = "agg")]
auto_impl_view!(in2, [corr], (method: CorrMethod, min_periods: usize, stable: bool, axis: i32, par: bool));
#[cfg(feature = "agg")]
auto_impl_view!(in2, [cov], (min_periods: usize, stable: bool, axis: i32, par: bool));

// === window expression ===
// window function without stable argument
#[cfg(feature = "window_func")]
auto_impl_view!(in1,
    [
        ts_argmin, ts_argmax, ts_min, ts_max, ts_rank, ts_rank_pct,
        ts_prod, ts_prod_mean, ts_minmaxnorm,
    ],
    (window: usize, min_periods: usize, axis: i32, par: bool)
);
#[cfg(feature = "window_func")]
// window corr and cov
auto_impl_view!(in2, [ts_cov, ts_corr, ts_regx_alpha, ts_regx_beta],
    (window: usize, min_periods: usize, stable: bool, axis: i32, par: bool)
);
#[cfg(feature = "window_func")]
// window corr and cov
auto_impl_view!(in2, [ts_regx_resid_std, ts_regx_resid_skew],
    (window: usize, min_periods: usize, axis: i32, par: bool)
);

// window function with stable argument
#[cfg(feature = "window_func")]
auto_impl_view!(in1,
    [
        ts_sum, ts_sma, ts_ewm, ts_wma, ts_std, ts_var, ts_skew, ts_kurt,
        ts_stable, ts_meanstdnorm, ts_reg, ts_tsf, ts_reg_slope, ts_reg_intercept,
    ],
    (window: usize, min_periods: usize, stable: bool, axis: i32, par: bool)
);
