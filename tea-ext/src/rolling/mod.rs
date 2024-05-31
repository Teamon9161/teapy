mod cmp;
mod corr;
// mod feature;
#[cfg(feature = "lazy")]
mod impl_lazy;
// mod norm;
mod reg;

// pub use cmp::*;
pub use corr::*;
// pub use feature::*;
#[cfg(feature = "lazy")]
pub use impl_lazy::*;
// pub use norm::*;
pub use reg::*;

#[cfg(feature = "lazy")]
use lazy::Expr;
use ndarray::{Array1, Data, DataMut, Dimension, Ix1, ShapeBuilder};
use std::mem::MaybeUninit;
use tea_core::prelude::*;
use tevec::rolling::*;

macro_rules! auto_define_rolling_funcs {
    ($feature: ident:
        $($func: ident ($($param: ident: $ty: ty),*) -> $out: ty {$tv_func: ident}),*
    ) => {
        #[arr_map_ext(lazy = "view", type = "numeric")]
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
    // ts_prod -> ts_vprod_to,
    // ts_prod_mean -> ts_vprod_mean_to
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
