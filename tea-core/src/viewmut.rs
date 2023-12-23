use error::TpResult;

use super::prelude::{ArrBase, WrapNdarray};
use ndarray::{Dimension, Ix0, Ix1, Ix2, IxDyn, RawArrayViewMut, StrideShape, ViewRepr};

pub type ArrViewMut<'a, T, D> = ArrBase<ViewRepr<&'a mut T>, D>;
pub type ArrViewMut1<'a, T> = ArrViewMut<'a, T, Ix1>;
pub type ArrViewMut2<'a, T> = ArrViewMut<'a, T, Ix2>;
pub type ArrViewMutD<'a, T> = ArrViewMut<'a, T, IxDyn>;

impl<'a, T, D: Dimension> ArrViewMut<'a, T, D> {
    /// Create a 1d array view mut from slice directly.
    #[inline]
    pub fn from_slice<Sh>(shape: Sh, slc: &mut [T]) -> Self
    where
        Sh: Into<StrideShape<D>>,
    {
        unsafe {
            RawArrayViewMut::from_shape_ptr(shape, slc.as_mut_ptr())
                .deref_into_view_mut()
                .wrap()
        }
    }

    /// Cast the dtype of the array without copy.
    ///
    /// # Safety
    ///
    /// The size of `T` and `T2` must be the same
    #[inline]
    pub unsafe fn into_dtype<T2>(self) -> ArrViewMut<'a, T2, D> {
        use std::mem;
        if mem::size_of::<T>() == mem::size_of::<T2>() {
            let out = mem::transmute_copy(&self);
            mem::forget(self);
            out
        } else {
            panic!("the size of new type is different when into_dtype")
        }
    }

    /// # Safety
    ///
    /// See the safety requirements of `ArrayViewMut::from_shape_ptr`
    #[inline]
    pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *mut T) -> Self
    where
        Sh: Into<StrideShape<D>>,
    {
        assert!(!ptr.is_null(), "ptr is null when create ArrayViewMut");
        unsafe {
            RawArrayViewMut::from_shape_ptr(shape, ptr)
                .deref_into_view_mut()
                .wrap()
        }
    }
}

impl<'a, T> ArrViewMut<'a, T, Ix0> {
    #[inline(always)]
    pub fn into_scalar(self) -> &'a mut T {
        self.0.into_scalar()
    }
}

impl<'a, T> ArrViewMutD<'a, T> {
    #[inline(always)]
    pub fn into_scalar(self) -> TpResult<&'a mut T> {
        Ok(self.to_dim0()?.into_scalar())
    }
}
