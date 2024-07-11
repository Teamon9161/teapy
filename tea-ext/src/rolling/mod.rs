// mod corr;
#[cfg(feature = "lazy")]
mod impl_lazy;
// mod reg;

// pub use corr::*;
#[cfg(feature = "lazy")]
pub use impl_lazy::*;
// pub use reg::*;

#[cfg(feature = "lazy")]
use lazy::Expr;
use ndarray::{Array1, Data, DataMut, DimMax, Dimension, Ix1, ShapeBuilder};
use std::mem::MaybeUninit;
use tea_core::prelude::*;

macro_rules! auto_define_rolling_funcs {
    ($feature: ident:
        $($func: ident ($($param: ident: $ty: ty),*) -> $out: ty {$tv_func: ident}),* $(,)?
    ) => {
        #[arr_map_ext(lazy = "view", type = "Numeric")]
        impl<T: IsNone + Send + Sync, S: Data<Elem = T>, D: Dimension> $feature for ArrBase<S, D>
        {
            $(#[inline]
            fn $func<SO>(
                &self,
                out: &mut ArrBase<SO, Ix1>,
                $( $param: $ty, )*
            ) -> $out
            where
                SO: DataMut<Elem = MaybeUninit<$out>>,
                T::Inner: Number,
                Option<T::Inner>: Cast<$out>,
                f64: Cast<$out>
            {
                self.as_dim1()
                    .0
                    .$tv_func::<Array1<$out>, _>($($param,)* Some(out.0.view_mut()));
            })*
        }
    };
}

macro_rules! auto_define_rolling2_funcs {
    ($feature: ident:
        $($func: ident ($($param: ident: $ty: ty),*) -> $out: ty {$tv_func: ident}),* $(,)?
    ) => {
        #[arr_map2_ext(lazy = "view2", type = "PureNumeric", type2 = "PureNumeric")]
        impl<T: IsNone + Send + Sync, S: Data<Elem = T>, D: Dimension> $feature for ArrBase<S, D>
        {
            $(#[inline]
            fn $func<S2, D2, T2, SO>(
                &self,
                other: &ArrBase<S2, D2>,
                out: &mut ArrBase<SO, Ix1>,
                $( $param: $ty, )*
            ) -> $out
            where
                SO: DataMut<Elem = MaybeUninit<$out>>,
                S2: Data<Elem = T2>,
                D2: Dimension,
                D: DimMax<D2>,
                T2: IsNone + Send + Sync,
                T::Inner: Number,
                T2::Inner: Number,
                Option<T::Inner>: Cast<$out>,
                f64: Cast<$out>
            {
                self.as_dim1().0
                    .$tv_func::<Array1<$out>, _, _, _>(
                        &other.as_dim1().0,
                        $($param,)*
                        Some(out.0.view_mut())
                    );
            })*
        }
    };
}

auto_define_rolling_funcs!(
    FeatureTs:
    ts_sum(window: usize, min_periods: Option<usize>) -> f64 {ts_vsum_to},
    ts_mean(window: usize, min_periods: Option<usize>) -> f64 {ts_vmean_to},
    ts_ewm(window: usize, min_periods: Option<usize>) -> f64 {ts_vewm_to},
    ts_wma(window: usize, min_periods: Option<usize>) -> f64 {ts_vwma_to},
    ts_std(window: usize, min_periods: Option<usize>) -> f64 {ts_vstd_to},
    ts_var(window: usize, min_periods: Option<usize>) -> f64 {ts_vvar_to},
    ts_skew(window: usize, min_periods: Option<usize>) -> f64 {ts_vskew_to},
    ts_kurt(window: usize, min_periods: Option<usize>) -> f64 {ts_vkurt_to}
);

auto_define_rolling_funcs!(
    CmpTs:
    ts_min(window: usize, min_periods: Option<usize>) -> f64 {ts_vmin_to},
    ts_max(window: usize, min_periods: Option<usize>) -> f64 {ts_vmax_to},
    ts_argmin(window: usize, min_periods: Option<usize>) -> f64 {ts_vargmin_to},
    ts_argmax(window: usize, min_periods: Option<usize>) -> f64 {ts_vargmax_to},
    ts_rank(window: usize, min_periods: Option<usize>, pct: bool, rev: bool) -> f64 {ts_vrank_to}
);

auto_define_rolling_funcs!(
    NormTs:
    ts_zscore(window: usize, min_periods: Option<usize>) -> f64 {ts_vzscore_to},
    ts_minmaxnorm(window: usize, min_periods: Option<usize>) -> f64 {ts_vminmaxnorm_to}
);

auto_define_rolling_funcs!(
    RegTs:
    ts_reg(window: usize, min_periods: Option<usize>) -> f64 {ts_vreg_to},
    ts_tsf(window: usize, min_periods: Option<usize>) -> f64 {ts_vtsf_to},
    ts_reg_slope(window: usize, min_periods: Option<usize>) -> f64 {ts_vreg_slope_to},
    ts_reg_intercept(window: usize, min_periods: Option<usize>) -> f64 {ts_vreg_intercept_to},
    ts_reg_resid_mean(window: usize, min_periods: Option<usize>) -> f64 {ts_vreg_resid_mean_to},
);

auto_define_rolling2_funcs!(
    BinaryTs:
    ts_cov(window: usize, min_periods: Option<usize>) -> f64 {ts_vcov_to},
    ts_corr(window: usize, min_periods: Option<usize>) -> f64 {ts_vcorr_to},
    ts_regx_beta(window: usize, min_periods: Option<usize>) -> f64 {ts_vregx_beta_to},
    ts_regx_alpha(window: usize, min_periods: Option<usize>) -> f64 {ts_vregx_alpha_to},
    ts_regx_resid_mean(window: usize, min_periods: Option<usize>) -> f64 {ts_vregx_resid_mean_to},
    ts_regx_resid_std(window: usize, min_periods: Option<usize>) -> f64 {ts_vregx_resid_std_to},
    ts_regx_resid_skew(window: usize, min_periods: Option<usize>) -> f64 {ts_vregx_resid_skew_to},
);
