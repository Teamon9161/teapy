#[cfg(feature = "lazy")]
use lazy::Expr;

#[cfg(feature = "lazy")]
macro_rules! auto_impl_rolling_view {
    (
        $(in1, [$($func: ident),* $(,)?], $other: tt);*
        $(;in2, [$($func2: ident),* $(,)?], $other2: tt)*
        $(;)?

    ) => {
        #[ext_trait]
        impl<'a> ExprRollingExt for Expr<'a> {
            $($(auto_impl_view!(in1, $func, $other);)*)*
            $($(auto_impl_view!(in2, $func2, $other2);)*)*
        }
    };
}

auto_impl_rolling_view!(
    in1, [
        ts_argmin, ts_argmax, ts_min, ts_max, ts_rank, ts_rank_pct,
        ts_prod, ts_prod_mean, ts_minmaxnorm,
    ],
    (window: usize, min_periods: usize, axis: i32, par: bool);
    in1, [
        ts_sum, ts_sma, ts_ewm, ts_wma, ts_std, ts_var, ts_skew, ts_kurt,
        ts_stable, ts_meanstdnorm, ts_reg, ts_tsf, ts_reg_slope, ts_reg_intercept,
    ],
    (window: usize, min_periods: usize, stable: bool, axis: i32, par: bool);
    in2, [ts_regx_resid_std, ts_regx_resid_skew], (window: usize, min_periods: usize, axis: i32, par: bool);
);
