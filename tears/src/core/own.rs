use super::{ArrBase, WrapNdarray};
use ndarray::{Array, Dimension, Ix1, Ix2, IxDyn, OwnedRepr, ShapeBuilder};

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
    pub unsafe fn into_dtype<T2>(self) -> Arr<T2, D> {
        use std::mem;
        if mem::size_of::<T>() == mem::size_of::<T2>() {
            // mem::transmute(self)
            let out = mem::transmute_copy(&self);
            mem::forget(self);
            out
        } else {
            panic!("the size of new type is different when calling into_dtype for Arr")
        }

        // let shape = self.raw_dim();
        // let vec = self.0.into_raw_vec();
        // let (ptr, len, cap) = vec.into_raw_parts();
        // let vec = Vec::<T2>::from_raw_parts(ptr as *mut T2, len, cap);
        // Array::<T2, D>::from_shape_vec_unchecked(shape, vec).wrap()
    }

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
    pub fn default<Sh>(shape: Sh) -> Self
    where
        T: Default,
        Sh: ShapeBuilder<Dim = D>,
    {
        Array::default(shape).wrap()
    }
}
