use crate::TpResult;

use super::{ArrBase, WrapNdarray};
use ndarray::{Dimension, Ix0, Ix1, IxDyn, RawArrayView, StrideShape, ViewRepr};

pub type ArrView<'a, T, D> = ArrBase<ViewRepr<&'a T>, D>;
pub type ArrView1<'a, T> = ArrView<'a, T, Ix1>;
pub type ArrViewD<'a, T> = ArrView<'a, T, IxDyn>;

impl<'a, T, D: Dimension> ArrView<'a, T, D> {
    /// Cast the dtype of the array without copy.
    ///
    /// # Safety
    ///
    /// The size of `T` and `T2` must be the same
    pub unsafe fn into_dtype<T2>(self) -> ArrView<'a, T2, D> {
        use std::mem;
        if mem::size_of::<T>() == mem::size_of::<T2>() {
            let out = mem::transmute_copy(&self);
            mem::forget(self);
            out
        } else {
            panic!("the size of new type is different when into_dtype")
        }
    }

    /// Create an array view from slice directly.
    pub fn from_slice<Sh>(shape: Sh, slc: &[T]) -> Self
    where
        Sh: Into<StrideShape<D>>,
    {
        unsafe {
            RawArrayView::from_shape_ptr(shape, slc.as_ptr())
                .deref_into_view()
                .wrap()
        }
    }

    /// Create an array view from vec directly.
    pub fn from_ref_vec<Sh>(shape: Sh, vec: &Vec<T>) -> Self
    where
        Sh: Into<StrideShape<D>>,
    {
        unsafe {
            RawArrayView::from_shape_ptr(shape, vec.as_ptr())
                .deref_into_view()
                .wrap()
        }
    }

    /// # Safety
    ///
    /// See the safety requirements of `ArrayView::from_shape_ptr`
    pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *const T) -> Self
    where
        Sh: Into<StrideShape<D>>,
    {
        assert!(!ptr.is_null(), "ptr is null when create ArrayView");
        unsafe {
            RawArrayView::from_shape_ptr(shape, ptr)
                .deref_into_view()
                .wrap()
        }
    }
}

impl<'a, T> ArrView<'a, T, Ix0> {
    #[inline]
    pub fn into_scalar(self) -> &'a T {
        self.0.into_scalar()
    }
}

impl<'a, T> ArrViewD<'a, T> {
    pub fn into_scalar(self) -> TpResult<&'a T> {
        Ok(self.to_dim0()?.into_scalar())
    }
}
