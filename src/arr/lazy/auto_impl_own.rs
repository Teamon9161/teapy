use num::traits::AsPrimitive;

use super::super::{CorrMethod, Number, QuantileMethod, WinsorizeMethod};
use super::{ArbArray, Expr, ExprElement, RefType};

macro_rules! impl_view_lazy {
    (in1, $func:ident -> $otype:ident, ($($p:ident: $p_ty:ty),* $(,)?)) => {
        pub fn $func (self $(, $p: $p_ty)*) -> Expr<'a, $otype> {
            self.chain_view_f(move |arr| arr.$func($($p),*).into(), RefType::False)
        }
    };
    (in1, [$($func: ident -> $otype:ident),* $(,)?], $other: tt) => {
        $(impl_view_lazy!(in1, $func -> $otype, $other);)*
    };
    (in1-inplace, $func:ident, $func_inplace: ident -> $otype:ident, ($($p:ident: $p_ty:ty),* $(,)?)) => {
        pub fn $func (self $(, $p: $p_ty)*) -> Expr<'a, $otype> {
            self.chain_arr_f(move |arb_arr| {
                use ArbArray::*;
                match arb_arr {
                    View(arr) => arr.$func($($p),*).into(),
                    ViewMut(mut arr) => {
                        arr.$func_inplace($($p),*);
                        ViewMut(arr)
                    },
                    Owned(mut arr) => {
                        arr.view_mut().$func_inplace($($p),*);
                        Owned(arr)
                    },
                }
            }, RefType::Keep)
        }
    };
    (in1-inplace, [$($func: ident, $func_inplace: ident -> $otype:ident),* $(,)?], $other: tt) => {
        $(impl_view_lazy!(in1-inplace, $func, $func_inplace -> $otype, $other);)*
    };

    (in2, $func:ident -> $otype:ty, ($($p:ident: $p_ty:ty),* $(,)?)) => {
        pub fn $func<T2> (self, other: Expr<'a, T2> $(, $p: $p_ty)*) -> Expr<'a, $otype>
        where
            T2: Number + ExprElement,
        {
            self.chain_view_f(move |arr| arr.$func(&other.eval().view_arr(), $($p),*).into(), RefType::False)
        }
    };
    (in2, [$($func: ident -> $otype:ident),* $(,)?], $other: tt) => {
        $(impl_view_lazy!(in2, $func -> $otype, $other);)*
    };
}

impl<'a, T> Expr<'a, T>
where
    T: Number + ExprElement,
    f64: AsPrimitive<T>,
{
    // in1: lazy function with one input array,
    // mean -> f64: eager function name in Arr, and the dtype of output is f64
    // (stable: bool): other arguments of the function
    // impl_view_lazy!(in1, [remove_nan_1d -> f64], ());
    impl_view_lazy!(in1, [is_nan -> bool, not_nan -> bool], ());
    impl_view_lazy!(in1, [diff -> f64, pct_change -> f64], (n: i32, axis: i32, par: bool));
    impl_view_lazy!(in1,
        [
            count_notnan -> i32, count_nan -> i32, median -> f64, max -> T,
            min -> T, prod -> T, cumprod -> T,
        ],
        (axis: i32, par: bool)
    );
    impl_view_lazy!(in1,
        [
            sum -> T, mean -> f64, var -> f64, std -> f64,
            skew -> f64, kurt -> f64, cumsum -> T,
        ],
        (stable: bool, axis: i32, par: bool)
    );
    impl_view_lazy!(in1-inplace,
        [
            zscore, zscore_inplace -> T,
        ],
        (stable: bool, axis: i32, par: bool)
    );
    impl_view_lazy!(in1-inplace,
        [
            winsorize, winsorize_inplace -> T,
        ],
        (method: WinsorizeMethod, method_params: Option<f64>, stable: bool, axis: i32, par: bool)
    );

    impl_view_lazy!(in1, quantile -> f64, (q: f64, method: QuantileMethod, axis: i32, par: bool));
    impl_view_lazy!(in1, rank -> f64, (pct: bool, rev: bool, axis: i32, par: bool));
    impl_view_lazy!(in1, argsort -> i32, (rev: bool, axis: i32, par: bool));
    impl_view_lazy!(in1, split_group -> i32, (group: usize, rev: bool, axis: i32, par: bool));
    impl_view_lazy!(in1, arg_partition -> i32, (kth: usize, sort: bool, rev: bool, axis: i32, par: bool));
    impl_view_lazy!(in2, cov -> f64, (stable: bool, axis: i32, par: bool));
    impl_view_lazy!(in2, corr -> f64, (method: CorrMethod, stable: bool, axis: i32, par: bool));

    // === window expression ===
    // window function without stable argument
    #[cfg(feature = "window_func")]
    impl_view_lazy!(in1,
        [
            ts_argmin -> f64, ts_argmax -> f64, ts_min -> f64,
            ts_max -> f64, ts_rank -> f64, ts_rank_pct -> f64,
            ts_prod -> f64, ts_prod_mean -> f64, ts_minmaxnorm -> f64,
        ],
        (window: usize, min_periods: usize, axis: i32, par: bool)
    );
    #[cfg(feature = "window_func")]
    // window corr and cov
    impl_view_lazy!(in2, [ts_cov -> f64, ts_corr -> f64],
        (window: usize, min_periods: usize, stable: bool, axis: i32, par: bool)
    );

    // window function with stable argument
    #[cfg(feature = "window_func")]
    impl_view_lazy!(in1,
        [
            ts_sum -> f64, ts_sma -> f64, ts_ewm -> f64, ts_wma -> f64,
            ts_std -> f64, ts_var -> f64, ts_skew -> f64, ts_kurt -> f64,
            ts_stable -> f64, ts_meanstdnorm -> f64, ts_reg -> f64,
            ts_tsf -> f64, ts_reg_slope -> f64, ts_reg_intercept -> f64,
        ],
        (window: usize, min_periods: usize, stable: bool, axis: i32, par: bool)
    );
}
