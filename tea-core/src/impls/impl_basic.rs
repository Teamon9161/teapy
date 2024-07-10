// use crate::impl_reduce_nd;
use crate::prelude::{Arr1, ArrBase, ArrD, WrapNdarray};
// use datatype::{BoolType, IsNone, Number, TvNumber};
use ndarray::{Data, Dimension, Ix1, Zip};
use num::Zero;
use tea_macros::arr_agg_ext;
use tevec::prelude::*;

impl<T: Clone, S: Data<Elem = T>> ArrBase<S, Ix1> {
    /// sum of the array on a given axis, return valid_num n and the sum of the array
    pub fn nsum_1d(&self) -> (usize, T::Inner)
    where
        T: IsNone,
        T::Inner: Number,
    {
        if let Some(slc) = self.0.try_as_slice() {
            slc.titer().vfold_n(T::Inner::zero(), |acc, x| acc + x)
        } else {
            // fall back to normal calculation
            self.0.titer().vfold_n(T::Inner::zero(), |acc, x| acc + x)
        }
    }
}

#[arr_agg_ext]
impl<T: IsNone + Clone + Send + Sync, S: Data<Elem = T>, D: Dimension> BasicAggExt
    for ArrBase<S, D>
{
    /// Max value of the array on a given axis
    pub fn max(&self) -> T
    where
        T::Inner: Number,
    {
        let arr1 = self.as_dim1().0;
        let res = if let Some(slc) = arr1.try_as_slice() {
            slc.titer().vmax()
        } else {
            arr1.titer().vmax()
        };
        T::from_opt(res)
    }

    /// Min value of the array on a given axis
    pub fn min(&self) -> T
    where
        T::Inner: Number,
    {
        let arr1 = self.as_dim1().0;
        let res = if let Some(slc) = arr1.try_as_slice() {
            slc.titer().vmin()
        } else {
            arr1.titer().vmin()
        };
        T::from_opt(res)
    }

    /// sum of the array on a given axis
    #[inline]
    pub fn sum(&self) -> T
    where
        T::Inner: Number,
    {
        T::from_inner(self.as_dim1().nsum_1d().1)
    }

    /// mean of the array on a given axis
    #[inline]
    pub fn mean(&self, min_periods: usize) -> f64
    where
        T::Inner: Number,
    {
        let (n, sum) = self.as_dim1().nsum_1d();
        if n >= min_periods {
            sum.f64() / n.f64()
        } else {
            f64::NAN
        }
    }

    /// count a value of an array on a given axis
    #[inline]
    pub fn count_v(&self, value: T) -> i32
    where
        T::Inner: PartialEq,
    {
        self.as_dim1().0.titer().vcount_value(value) as i32
    }

    /// count a value of an array on a given axis
    #[inline]
    pub fn count_none(&self) -> i32 {
        self.as_dim1().0.titer().count_none() as i32
    }

    /// count a value of an array on a given axis
    #[inline]
    pub fn count_valid(&self) -> i32 {
        self.as_dim1().0.titer().count_valid() as i32
    }

    #[inline]
    pub fn any(&self) -> bool
    where
        T: BoolType,
    {
        self.as_dim1().0.titer().any()
    }

    #[inline]
    pub fn all(&self) -> bool
    where
        T: BoolType,
    {
        self.as_dim1().0.titer().all()
    }
}
