use crate::prelude::ArrBase;
use ndarray::{Data, IxDyn, LinalgScalar};
#[cfg(feature = "ops")]
use num::traits::{abs, real::Real, Signed};
// use std::cmp::PartialOrd;

#[cfg(feature = "ops")]
use crate::prelude::{Arr, ArrD, ArrView, ArrViewMut, TpResult, WrapNdarray};
#[cfg(feature = "ops")]
use ndarray::{DataMut, DimMax, Dimension, Ix2, Zip};

#[cfg(feature = "ops")]
macro_rules! impl_cmp {
    ($func: ident, $func_impl: expr $(,T: $trait1: path)? $(, T2: $trait2: path)?) => {
        pub fn $func<S2, D2, T2>(&self, rhs: &ArrBase<S2, D2>, par: bool) -> Arr<bool, <D as DimMax<D2>>::Output>
        where
            D2: Dimension,
            D: DimMax<D2>,
            T: Send + Sync $(+ $trait1)?,
            T2: Send + Sync $(+ $trait2)?,
            S2: Data<Elem = T2>,
        {
            let (lhs, rhs) = if self.ndim() == rhs.ndim() && self.shape() == rhs.shape() {
                let lhs = self.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
                let rhs = rhs.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
                (lhs, rhs)
            } else {
                self.broadcast_with(rhs).unwrap()
            };
            if !par {
                Zip::from(lhs.0).and(rhs.0).map_collect($func_impl).wrap()
            } else {
                Zip::from(lhs.0).and(rhs.0).par_map_collect($func_impl).wrap()
            }
        }
    };

    (opf $func: ident, $operator: tt $(, T: $trait1: path)? $(, T2: $trait2: path)? ) => {
        impl_cmp!($func, |a, b| a $operator b $(,T: $trait1)? $(, T2: $trait2)?);
    }

}

#[cfg(feature = "ops")]
impl<T, S, D> ArrBase<S, D>
where
    S: Data<Elem = T>,
    D: Dimension,
{
    impl_cmp!(opf gt, >, T: PartialOrd<T2>);
    impl_cmp!(opf ge, >=, T: PartialOrd<T2>);
    impl_cmp!(opf lt, <, T: PartialOrd<T2>);
    impl_cmp!(opf le, <=, T: PartialOrd<T2>);
    impl_cmp!(opf eq, ==, T: PartialEq<T2>);
    impl_cmp!(opf ne, !=, T: PartialEq<T2>);
}

impl<T, S> ArrBase<S, IxDyn>
where
    S: Data<Elem = T>,
    T: LinalgScalar,
{
    #[cfg(feature = "ops")]
    // #[allow(clippy::useless_conversion)]
    pub fn dot<S2>(&self, other: &ArrBase<S2, IxDyn>) -> TpResult<ArrD<T>>
    where
        S2: Data<Elem = T>,
    {
        match (self.ndim(), other.ndim()) {
            (1, 1) => Ok(self
                .view()
                .to_dim1()?
                .0
                .dot(&other.view().to_dim1().unwrap().0)
                .into()),
            (1, 2) => Ok(self
                .view()
                .to_dim1()?
                .0
                .dot(&other.view().to_dim::<Ix2>().unwrap().0)
                .wrap()
                .to_dimd()),
            (2, 1) => Ok(self
                .view()
                .to_dim2()?
                .0
                .dot(&other.view().to_dim1().unwrap().0)
                .wrap()
                .to_dimd()),
            (2, 2) => Ok(self
                .view()
                .to_dim2()?
                .0
                .dot(&other.view().to_dim::<Ix2>().unwrap().0)
                .wrap()
                .to_dimd()),
            _ => Err(error::StrError::from("dot for this dim is not suppported")),
        }
        // .into()
    }
}

#[cfg(feature = "ops")]
macro_rules! impl_pow {
    ($T: ty) => {
        impl<'a, D> ArrView<'a, $T, D>
        where
            D: Dimension,
        {
            pub fn pow<S2, D2>(
                &self,
                rhs: &ArrBase<S2, D2>,
                par: bool,
            ) -> Arr<$T, <D as DimMax<D2>>::Output>
            where
                D2: Dimension,
                D: DimMax<D2>,
                S2: Data<Elem = usize>,
            {
                let (lhs, rhs) = if self.ndim() == rhs.ndim() && self.shape() == rhs.shape() {
                    let lhs = self.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
                    let rhs = rhs.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
                    (lhs, rhs)
                } else {
                    self.broadcast_with(rhs).unwrap()
                };
                if !par {
                    Zip::from(lhs.0)
                        .and(rhs.0)
                        .map_collect(|a, b| a.pow(*b as u32))
                        .wrap()
                } else {
                    Zip::from(lhs.0)
                        .and(rhs.0)
                        .par_map_collect(|a, b| a.pow(*b as u32))
                        .wrap()
                }
            }
        }

        impl<'a, D> ArrViewMut<'a, $T, D>
        where
            D: Dimension,
        {
            #[inline(always)]
            pub fn pow<S2, D2>(
                &self,
                rhs: &ArrBase<S2, D2>,
                par: bool,
            ) -> Arr<$T, <D as DimMax<D2>>::Output>
            where
                D2: Dimension,
                D: DimMax<D2>,
                S2: Data<Elem = usize>,
            {
                self.view().pow(rhs, par)
            }
        }

        impl<D> Arr<$T, D>
        where
            D: Dimension,
        {
            #[inline(always)]
            pub fn pow<S2, D2>(
                &self,
                rhs: &ArrBase<S2, D2>,
                par: bool,
            ) -> Arr<$T, <D as DimMax<D2>>::Output>
            where
                D2: Dimension,
                D: DimMax<D2>,
                S2: Data<Elem = usize>,
            {
                self.view().pow(rhs, par)
            }
        }
    };
}

#[cfg(feature = "ops")]
impl_pow!(i32);
#[cfg(feature = "ops")]
impl_pow!(i64);
#[cfg(feature = "ops")]
impl_pow!(usize);

#[cfg(feature = "ops")]
impl<T, S, D> ArrBase<S, D>
where
    S: Data<Elem = T>,
    D: Dimension,
{
    #[inline(always)]
    pub fn abs_inplace(&mut self, par: bool)
    where
        T: Signed + Send + Sync,
        S: DataMut<Elem = T>,
    {
        if !par {
            self.map_inplace(|v| *v = v.abs());
        } else {
            self.par_map_inplace(|v| *v = v.abs());
        }
    }

    #[inline(always)]
    pub fn abs(&self) -> Arr<T, D>
    where
        T: Signed + Send + Sync + Clone,
    {
        self.map(|v| abs(v.clone()))
    }

    #[inline(always)]
    pub fn sign(&self) -> Arr<T, D>
    where
        T: Signed + Clone,
    {
        self.map(|v| v.signum())
    }

    pub fn powi<S2, D2>(
        &self,
        rhs: &ArrBase<S2, D2>,
        par: bool,
    ) -> Arr<T, <D as DimMax<D2>>::Output>
    where
        D2: Dimension,
        D: DimMax<D2>,
        T: Send + Sync + Real,
        S2: Data<Elem = i32>,
    {
        let (lhs, rhs) = if self.ndim() == rhs.ndim() && self.shape() == rhs.shape() {
            let lhs = self.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
            let rhs = rhs.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
            (lhs, rhs)
        } else {
            self.broadcast_with(rhs).unwrap()
        };
        if !par {
            Zip::from(lhs.0)
                .and(rhs.0)
                .map_collect(|a, b| a.powi(*b))
                .wrap()
        } else {
            Zip::from(lhs.0)
                .and(rhs.0)
                .par_map_collect(|a, b| a.powi(*b))
                .wrap()
        }
    }

    pub fn powf<S2, D2>(
        &self,
        rhs: &ArrBase<S2, D2>,
        par: bool,
    ) -> Arr<T, <D as DimMax<D2>>::Output>
    where
        D2: Dimension,
        D: DimMax<D2>,
        T: Send + Sync + Real,
        S2: Data<Elem = T>,
    {
        let (lhs, rhs) = if self.ndim() == rhs.ndim() && self.shape() == rhs.shape() {
            let lhs = self.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
            let rhs = rhs.view().to_dim::<<D as DimMax<D2>>::Output>().unwrap();
            (lhs, rhs)
        } else {
            self.broadcast_with(rhs).unwrap()
        };
        if !par {
            Zip::from(lhs.0)
                .and(rhs.0)
                .map_collect(|a, b| a.powf(*b))
                .wrap()
        } else {
            Zip::from(lhs.0)
                .and(rhs.0)
                .par_map_collect(|a, b| a.powf(*b))
                .wrap()
        }
    }
}
