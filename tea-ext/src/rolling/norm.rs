#[cfg(feature = "lazy")]
use lazy::Expr;
use ndarray::{Data, DataMut, Dimension, Ix1, ShapeBuilder};
use std::mem::MaybeUninit;
use tea_core::prelude::*;
use ndarray::Array1;
use tevec::rolling::*;

macro_rules! auto_define_feature_norm_funcs {
    ($($func: ident->$tv_func: ident),*) => {
        #[arr_map_ext(lazy = "view", type = "numeric")]
        impl<T: IsNone + Clone + Send + Sync, S: Data<Elem = T>, D: Dimension> NormTs for ArrBase<S, D>
        where T::Cast<f64>: Send + Sync,
        {
            $(#[inline]
            fn $func<SO>(
                &self,
                out: &mut ArrBase<SO, Ix1>,
                window: usize,
                min_periods: Option<usize>,
                // _ignore_na: bool,
            ) -> T::Cast<f64>
            where
                SO: DataMut<Elem = MaybeUninit<T::Cast<f64>>>,
                T::Inner: Number,
            {
                self.as_dim1()
                    .0
                    .$tv_func::<Array1<_>>(window, min_periods, Some(out.0.view_mut()));
            })*
        }
    };
}

auto_define_feature_norm_funcs!(
    ts_zscore -> ts_vzscore_to,
    ts_minmaxnorm -> ts_vminmaxnorm_to
);
