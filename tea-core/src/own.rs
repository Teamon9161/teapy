use std::mem::MaybeUninit;

use error::TpResult;

use super::prelude::{ArrBase, WrapNdarray};
use ndarray::{arr0 as nd_arr0, Array, Dimension, Ix0, Ix1, Ix2, IxDyn, OwnedRepr, ShapeBuilder};

pub type Arr<T, D> = ArrBase<OwnedRepr<T>, D>;
pub type ArrD<T> = Arr<T, IxDyn>;
pub type Arr1<T> = Arr<T, Ix1>;
pub type Arr2<T> = Arr<T, Ix2>;

impl<T, D: Dimension> Arr<T, D> {
    /// Cast the dtype of the array without copy.
    ///
    /// # Safety
    ///
    /// The size of `T` and `T2` must be the same
    #[inline]
    pub unsafe fn into_dtype<T2>(self) -> Arr<T2, D> {
        use std::mem;
        if mem::size_of::<T>() == mem::size_of::<T2>() {
            let out = mem::transmute_copy(&self);
            mem::forget(self);
            out
        } else {
            panic!("the size of new type is different when calling into_dtype for Arr")
        }
    }

    #[inline(always)]
    pub fn from_elem<Sh>(shape: Sh, elem: T) -> Self
    where
        T: Clone,
        Sh: ShapeBuilder<Dim = D>,
    {
        Array::from_elem(shape, elem).wrap()
    }

    /// Create an array with default values, shape `shape`
    ///
    /// **Panics** if the product of non-zero axis lengths overflows `isize`.
    #[inline(always)]
    pub fn default<Sh>(shape: Sh) -> Self
    where
        T: Default,
        Sh: ShapeBuilder<Dim = D>,
    {
        Array::default(shape).wrap()
    }

    #[inline(always)]
    pub fn uninit<Sh>(shape: Sh) -> Arr<MaybeUninit<T>, D>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Array::uninit(shape).wrap()
    }

    #[inline(always)]
    pub fn into_raw_vec(self) -> Vec<T> {
        self.0.into_raw_vec()
    }

    // pub unsafe fn assume_init(&mut self: Arr<MaybeUninit<T>, D>) -> Arr<T, D>
    // {
    //     self.0.assume_init()
    // }
}

impl<T, D: Dimension> Arr<MaybeUninit<T>, D> {
    /// **Promise** that the array's elements are all fully initialized, and convert
    /// the array from element type `MaybeUninit<A>` to `A`.
    ///
    /// For example, it can convert an `Arr<MaybeUninit<f64>, D>` to `Arr<f64, D>`.
    ///
    /// ## Safety
    ///
    /// Safe to use if all the array's elements have been initialized.
    ///
    /// Note that for owned and shared ownership arrays, the promise must include all of the
    /// array's storage; it is for example possible to slice these in place, but that must
    /// only be done after all elements have been initialized.
    #[inline(always)]
    pub unsafe fn assume_init(self) -> Arr<T, D> {
        self.0.assume_init().wrap()
    }
}

impl<T> Arr<T, Ix0> {
    #[inline]
    pub fn into_scalar(self) -> T {
        self.0.into_scalar()
    }
}

impl<T> From<T> for Arr<T, Ix0> {
    #[inline(always)]
    fn from(t: T) -> Self {
        nd_arr0(t).wrap()
    }
}

#[inline(always)]
pub fn arr0<T>(t: T) -> Arr<T, Ix0> {
    nd_arr0(t).wrap()
}

impl<T> ArrD<T> {
    #[inline(always)]
    pub fn into_scalar(self) -> TpResult<T> {
        Ok(self.to_dim0()?.into_scalar())
    }
}
