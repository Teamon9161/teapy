mod arbarray;
mod arrok;
mod own;
mod traits;
mod view;
mod viewmut;

pub use arbarray::ArbArray;
pub use arrok::ArrOk;
pub use own::{Arr, Arr1, Arr2, ArrD};
pub use traits::WrapNdarray;
pub use view::{ArrView, ArrView1, ArrViewD};
pub use viewmut::{ArrViewMut, ArrViewMut1, ArrViewMutD};

use crate::{export::*, DateTime, GetDataType, Iter, IterMut, TimeUnit};

use ndarray::{
    Array1, ArrayBase, Axis, Data, DataMut, DataOwned, Dimension, Ix0, Ix1, Ix2, IxDyn, RawData,
    RemoveAxis, ShapeBuilder, ShapeError, Zip,
};

use num::{traits::AsPrimitive, Zero};
use pyo3::{Python, ToPyObject};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::fmt::Debug;
use std::sync::Arc;

// #[cfg(feature = "option_dtype")]
// use crate::datatype::ArrToOpt;

#[cfg(feature = "npy")]
use ndarray_npy::{write_npy, WritableElement, WriteNpyError};

pub struct ArrBase<S, D>(pub ArrayBase<S, D>)
where
    S: RawData;

pub trait Dim1: Dimension + RemoveAxis {}
impl Dim1 for Ix1 {}

pub type ArrBase1<S> = ArrBase<S, Ix1>;

impl<T, S, D> ArrBase<S, D>
where
    S: RawData<Elem = T>,
    D: Dimension,
{
    #[inline]
    pub fn new(a: ArrayBase<S, D>) -> Self {
        Self(a)
    }

    #[inline]
    pub fn ensure_axis(&self, axis: i32) -> usize {
        if axis < 0 {
            (self.ndim() as i32 + axis) as usize
        } else {
            axis as usize
        }
    }

    #[inline]
    pub fn norm_axis(&self, axis: i32) -> Axis {
        Axis(self.ensure_axis(axis))
    }

    #[inline]
    pub fn ensure_index(&self, index: i32, length: usize) -> usize {
        if index < 0 {
            (length as i32 + index) as usize
        } else {
            index as usize
        }
    }

    #[cfg(feature = "npy")]
    pub fn write_npy<P>(self, path: P) -> Result<(), WriteNpyError>
    where
        P: AsRef<std::path::Path>,
        T: WritableElement,
        S: Data,
    {
        write_npy(path, &self.0)
    }
    /// Create a one-dimensional array from a vector (no copying needed).
    #[inline]
    pub fn from_vec(v: Vec<T>) -> Arr1<T>
    where
        S: DataOwned,
        D: Dim1,
    {
        Array1::from_vec(v).wrap()
    }

    #[inline]
    pub fn from_par_iter<I: IntoParallelIterator<Item = T>>(iterable: I) -> Arr1<T>
    where
        T: Send,
        D: Dim1,
        S: DataOwned<Elem = T>,
    {
        Arr1::from_vec(iterable.into_par_iter().collect())
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I: IntoIterator<Item = T>>(iterable: I) -> Arr1<T>
    where
        D: Dim1,
        S: DataOwned<Elem = T>,
    {
        Array1::from_iter(iterable).wrap()
    }

    /// Create a 1d array from slice, need clone.
    pub fn clone_from_slice(slc: &[T]) -> Arr1<T>
    where
        T: Clone,
        D: Dim1,
    {
        Array1::from_vec(slc.to_vec()).wrap()
    }

    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        S: DataOwned<Elem = T>,
        T: Clone + Zero,
        Sh: ShapeBuilder<Dim = D>,
    {
        ArrayBase::zeros(shape).wrap()
    }

    /// Change the array to dim0.
    ///
    /// Note that the original array must be dim0.
    #[inline]
    pub fn to_dim0(self) -> Result<ArrBase<S, Ix0>, ShapeError> {
        self.to_dim::<Ix0>()
    }

    /// Change the array to dim1.
    ///
    /// Note that the original array must be dim1.
    #[inline]
    pub fn to_dim1(self) -> Result<ArrBase<S, Ix1>, ShapeError> {
        self.to_dim::<Ix1>()
    }

    /// Change the array to dim2.
    ///
    /// Note that the original array must be dim2.
    #[inline]
    pub fn to_dim2(self) -> Result<ArrBase<S, Ix2>, ShapeError> {
        self.to_dim::<Ix2>()
    }

    /// Change the array to dimD.
    #[inline]
    pub fn to_dimd(self) -> Result<ArrBase<S, IxDyn>, ShapeError> {
        self.to_dim::<IxDyn>()
    }

    /// Change the array to another dim.
    #[inline]
    pub fn to_dim<D2: Dimension>(self) -> Result<ArrBase<S, D2>, ShapeError> {
        let res = self.0.into_dimensionality::<D2>();
        res.map(|arr| ArrBase(arr))
    }
}

impl<T, S, D> ArrBase<S, D>
where
    S: Data<Elem = T>,
    D: Dimension,
{
    /// Clone the elements in the array to `out` array.
    #[inline]
    pub fn clone_to<S2>(&self, out: &mut ArrBase<S2, D>)
    where
        T: Clone,
        S2: DataMut<Elem = T>,
    {
        out.zip_mut_with(self, |vo, v| *vo = v.clone());
    }

    /// Return a read-only view of the array
    pub fn view(&self) -> ArrView<'_, T, D> {
        ArrBase(self.0.view())
    }

    /// Return a read-write view of the array
    pub fn view_mut(&mut self) -> ArrViewMut<'_, T, D>
    where
        S: DataMut,
    {
        ArrBase(self.0.view_mut())
    }

    /// Return an uniquely owned copy of the array.
    pub fn to_owned(&self) -> Arr<T, D>
    where
        T: Clone,
    {
        self.0.to_owned().wrap()
    }

    pub fn to_arc(self) -> Arc<Self>
    where
        S: Data,
    {
        Arc::new(self)
    }

    /// Call `f` by reference on each element and create a new array
    /// with the new values.
    ///
    /// Elements are visited in arbitrary order.
    ///
    /// Return an array with the same shape as `self`.
    pub fn map<'a, T2, F>(&'a self, f: F) -> Arr<T2, D>
    where
        F: FnMut(&'a T) -> T2,
        T: 'a,
    {
        self.0.map(f).wrap()
    }

    /// Call `f` by **v**alue on each element and create a new array
    /// with the new values.
    ///
    /// Elements are visited in arbitrary order.
    ///
    /// Return an array with the same shape as `self`.
    pub fn mapv<T2, F>(&self, mut f: F) -> Arr<T2, D>
    where
        F: FnMut(T) -> T2,
        T: Copy,
    {
        self.map(move |x| f(*x))
    }

    pub fn cast<T2>(self) -> Arr<T2, D>
    where
        T: GetDataType + AsPrimitive<T2>,
        T2: GetDataType + Copy + 'static,
    {
        self.mapv(|v| v.as_())
    }

    pub fn apply_along_axis<S2, T2, F>(&self, out: &mut ArrBase<S2, D>, axis: Axis, par: bool, f: F)
    where
        T: Send + Sync,
        T2: Send + Sync,
        S2: DataMut<Elem = T2>,
        F: Fn(ArrView1<T>, ArrViewMut1<T2>) + Send + Sync,
    {
        let arr_zip = Zip::from(self.lanes(axis)).and(out.lanes_mut(axis));
        let ndim = self.ndim();
        if !par || (ndim == 1) {
            // 非并行
            arr_zip.for_each(|a, b| f(a.wrap(), b.wrap()));
        } else {
            // 并行
            arr_zip.par_for_each(|a, b| f(a.wrap(), b.wrap()));
        }
    }

    pub fn apply_along_axis_with<S2, T2, S3, T3, F>(
        &self,
        other: ArrBase<S2, D>,
        out: &mut ArrBase<S3, D>,
        axis: Axis,
        par: bool,
        f: F,
    ) where
        T: Send + Sync,
        T2: Send + Sync,
        T3: Send + Sync,
        S2: Data<Elem = T2>,
        S3: DataMut<Elem = T3>,
        F: Fn(ArrView1<T>, ArrView1<T2>, ArrViewMut1<T3>) + Send + Sync,
    {
        let arr_zip = Zip::from(self.lanes(axis))
            .and(other.lanes(axis))
            .and(out.lanes_mut(axis));
        let ndim = self.ndim();
        if !par || (ndim == 1) {
            // 非并行
            arr_zip.for_each(|a, b, c| f(a.wrap(), b.wrap(), c.wrap()));
        } else {
            // 并行
            arr_zip.par_for_each(|a, b, c| f(a.wrap(), b.wrap(), c.wrap()));
        }
    }

    /// Try to cast to bool
    pub fn to_bool(&self) -> Arr<bool, D>
    where
        T: Debug + Clone + AsPrimitive<i32>,
    {
        self.mapv(|v| {
            if v.as_() == 0 {
                false
            } else if v.as_() == 1 {
                true
            } else {
                panic!("can not cast {v:?} to bool")
            }
        })
    }

    /// Try to cast to pyobject
    pub fn to_object(&self, py: Python) -> Arr<PyValue, D>
    where
        T: Debug + Clone + ToPyObject,
    {
        self.map(|v| PyValue(v.clone().to_object(py)))
    }

    /// Try to cast to datetime
    pub fn to_datetime(&self, unit: TimeUnit) -> Arr<DateTime, D>
    where
        T: AsPrimitive<i64> + GetDataType,
    {
        match T::dtype() {
            DataType::DateTime => unsafe { self.to_owned().into_dtype::<DateTime>() },
            _ => self.map(|v| match unit {
                TimeUnit::Nanosecond => DateTime::from_timestamp_ns(v.as_()).unwrap_or_default(),
                TimeUnit::Microsecond => DateTime::from_timestamp_us(v.as_()).unwrap_or_default(),
                TimeUnit::Millisecond => DateTime::from_timestamp_ms(v.as_()).unwrap_or_default(),
                TimeUnit::Second => DateTime::from_timestamp_opt(v.as_(), 0),
                _ => unimplemented!("not support datetime unit"),
            }), // _ => todo!(),
        }
    }

    /// Try to cast to string
    pub fn to_string(&self) -> Arr<String, D>
    where
        T: ToString,
    {
        self.map(|v| v.to_string())
    }

    pub fn iter(&self) -> Iter<'_, T, D>
    where
        D: Dim1,
        S: Data,
    {
        Iter::new(self)
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, T, D>
    where
        D: Dim1,
        S: DataMut,
    {
        IterMut::new(self)
    }
}

impl<S: Data<Elem = PyValue>, D: Dimension> ArrBase<S, D> {
    /// Try to cast to string
    pub fn object_to_string(self, py: Python) -> Arr<String, D> {
        self.map(|v| v.0.extract::<String>(py).unwrap())
    }

    // /// Try to cast to str
    // pub fn object_to_str(self, py: Python) -> Arr<&str, D>
    // {
    //     self.map(|v| v.0.extract::<&str>(py).unwrap())
    // }
}