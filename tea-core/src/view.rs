use std::mem::MaybeUninit;

use crate::{
    prelude::{ArrD, TpResult},
    utils::vec_uninit,
};

use super::prelude::{ArrBase, WrapNdarray};
use ndarray::{
    s, Array, Dimension, Ix0, Ix1, Ix2, IxDyn, NewAxis, RawArrayView, ShapeBuilder, StrideShape,
    ViewRepr,
};

pub type ArrView<'a, T, D> = ArrBase<ViewRepr<&'a T>, D>;
pub type ArrView1<'a, T> = ArrView<'a, T, Ix1>;
pub type ArrView2<'a, T> = ArrView<'a, T, Ix2>;
pub type ArrViewD<'a, T> = ArrView<'a, T, IxDyn>;

impl<'a, T, D: Dimension> ArrView<'a, T, D> {
    /// Cast the dtype of the array without copy.
    ///
    /// # Safety
    ///
    /// The size of `T` and `T2` must be the same
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline(always)]
    pub fn into_scalar(self) -> &'a T {
        self.0.into_scalar()
    }
}

impl<'a, T> ArrViewD<'a, T> {
    #[inline(always)]
    pub fn into_scalar(self) -> TpResult<&'a T> {
        Ok(self.to_dim0()?.into_scalar())
    }

    #[inline]
    pub fn no_dim0(self) -> ArrViewD<'a, T> {
        if self.ndim() == 0 {
            self.0.slice_move(s!(NewAxis)).wrap().to_dimd()
        } else {
            self
        }
    }

    #[inline]
    pub fn to_owned_f(&self) -> ArrD<T>
    where
        T: Clone,
    {
        if self.t().is_standard_layout() {
            self.to_owned()
        } else {
            let mut arr_f =
                Array::from_shape_vec(self.shape().f(), vec_uninit(self.len())).unwrap();
            arr_f.zip_mut_with(self, |out, v| *out = MaybeUninit::new(v.clone()));
            unsafe { arr_f.assume_init().wrap() }
        }
    }
}
