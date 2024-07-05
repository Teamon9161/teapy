#[cfg(feature = "lazy")]
use lazy::Expr;
use ndarray::Array1;
use ndarray::{Data, DataMut, Dimension, Ix1, ShapeBuilder};
use std::mem::MaybeUninit;
use tea_core::prelude::*;

#[arr_map_ext(lazy = "view", type = "Numeric")]
impl<T: IsNone + Send + Sync, S: Data<Elem = T>, D: Dimension> CmpTs for ArrBase<S, D> {
    #[inline]
    fn ts_argmin<SO>(
        &self,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: Option<usize>,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T::Inner: Number,
    {
        self.as_dim1()
            .0
            .ts_vargmin_to::<Array1<_>, _>(window, min_periods, Some(out.view_mut().0))
            .unwrap();
    }

    #[inline]
    fn ts_argmax<SO>(
        &self,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: Option<usize>,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T::Inner: Number,
    {
        self.as_dim1()
            .0
            .ts_vargmax_to::<Array1<_>, _>(window, min_periods, Some(out.view_mut().0))
            .unwrap();
    }

    #[inline]
    fn ts_min<SO>(
        &self,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: Option<usize>,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T::Inner: Number,
        Option<T::Inner>: Cast<f64>,
    {
        self.as_dim1()
            .0
            .ts_vmin_to::<Array1<_>, _>(window, min_periods, Some(out.view_mut().0))
            .unwrap();
    }

    #[inline]
    fn ts_max<SO>(
        &self,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: Option<usize>,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T::Inner: Number,
        Option<T::Inner>: Cast<f64>,
    {
        self.as_dim1()
            .0
            .ts_vmax_to::<Array1<_>, _>(window, min_periods, Some(out.view_mut().0))
            .unwrap();
    }

    #[inline]
    fn ts_rank<SO>(
        &self,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: Option<usize>,
        pct: bool,
        rev: bool,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T::Inner: Number,
    {
        self.as_dim1()
            .0
            .ts_vrank_to::<Array1<_>, _>(window, min_periods, pct, rev, Some(out.view_mut().0))
            .unwrap();
    }
}
