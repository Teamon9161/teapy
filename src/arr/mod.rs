mod agg;
mod arr_func;
mod corr;
mod export;
mod iterators;
#[cfg(feature = "lazy")]
mod join;
mod macros;
mod option_datetype;

#[cfg(feature = "window_func")]
mod window;

pub mod datatype;
#[cfg(feature = "lazy")]
pub mod groupby;
#[cfg(feature = "lazy")]
#[macro_use]
pub mod lazy;
mod impls;
pub mod time;
pub mod util_trait;
pub mod utils;

pub use agg::QuantileMethod;
pub use arr_func::{FillMethod, WinsorizeMethod};
pub use corr::CorrMethod;
pub(crate) use datatype::match_datatype_arm;
pub use datatype::{BoolType, DataType, GetDataType, GetNone, Number};
#[cfg(feature = "lazy")]
pub use groupby::{flatten, groupby, groupby_par};
pub use iterators::{Iter, IterMut};
#[cfg(feature = "lazy")]
pub use join::{join_left, JoinType};
pub use util_trait::{CollectTrusted, CollectTrustedToVec, TrustedLen};
pub use utils::{kh_sum, DefaultNew, EmptyNew};

#[cfg(feature = "lazy")]
pub use lazy::{DropNaMethod, Expr, ExprElement, ExprOut, ExprOutView, Exprs, OlsResult, RefType};

pub use time::{DateTime, TimeDelta, TimeUnit};

use ndarray::{
    Array, Array1, ArrayBase, ArrayD, Axis, Data, DataMut, DataOwned, Dimension, Ix0, Ix1, Ix2,
    IxDyn, OwnedRepr, RawArrayView, RawArrayViewMut, RawData, RawDataClone, RemoveAxis,
    ShapeBuilder, ShapeError, StrideShape, ViewRepr, Zip,
};

use num::{traits::AsPrimitive, Zero};
use pyo3::{Python, ToPyObject};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::Debug;
use std::sync::Arc;

#[cfg(feature = "npy")]
use ndarray_npy::{write_npy, WritableElement, WriteNpyError};

pub struct ArrBase<S, D>(pub ArrayBase<S, D>)
where
    S: RawData;

pub trait Dim1: Dimension + RemoveAxis {}
impl Dim1 for Ix1 {}

pub type Arr<T, D> = ArrBase<OwnedRepr<T>, D>;
pub type ArrD<T> = Arr<T, IxDyn>;
pub type ArrView<'a, T, D> = ArrBase<ViewRepr<&'a T>, D>;
pub type ArrViewMut<'a, T, D> = ArrBase<ViewRepr<&'a mut T>, D>;
pub type ArrBase1<S> = ArrBase<S, Ix1>;
pub type Arr1<T> = Arr<T, Ix1>;
pub type Arr2<T> = Arr<T, Ix2>;
pub type ArrViewMut1<'a, T> = ArrViewMut<'a, T, Ix1>;
pub type ArrView1<'a, T> = ArrView<'a, T, Ix1>;
pub type ArrViewMutD<'a, T> = ArrViewMut<'a, T, IxDyn>;
pub type ArrViewD<'a, T> = ArrView<'a, T, IxDyn>;

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
    pub fn to_datetime(&self, _unit: TimeUnit) -> Arr<DateTime, D>
    where
        T: AsPrimitive<i64> + GetDataType,
    {
        match T::dtype() {
            DataType::DateTime => unsafe { self.to_owned().into_dtype::<DateTime>() },
            _ => todo!(),
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

impl<'a, T, D: Dimension> ArrViewMut<'a, T, D> {
    /// Create a 1d array view mut from slice directly.
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

pub trait WrapNdarray<S: RawData, D: Dimension> {
    fn wrap(self) -> ArrBase<S, D>;
}

impl<S: RawData, D: Dimension> WrapNdarray<S, D> for ArrayBase<S, D> {
    fn wrap(self) -> ArrBase<S, D> {
        ArrBase::new(self)
    }
}

use std::convert::From;
impl<S: RawData, D: Dimension> From<ArrayBase<S, D>> for ArrBase<S, D> {
    fn from(arr: ArrayBase<S, D>) -> Self {
        ArrBase::new(arr)
    }
}

impl<S: RawDataClone, D: Clone + Dimension> Clone for ArrBase<S, D> {
    fn clone(&self) -> Self {
        self.0.clone().wrap()
    }
}

#[derive(Debug)]
pub enum ArbArray<'a, T> {
    View(ArrViewD<'a, T>),
    ViewMut(ArrViewMutD<'a, T>),
    Owned(ArrD<T>),
}

impl<'a, T> Serialize for ArbArray<'a, T>
where
    T: Serialize + Clone,
{
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: Serializer,
    {
        match self {
            ArbArray::View(arr_view) => arr_view.to_owned().serialize(serializer),
            ArbArray::ViewMut(arr_view) => arr_view.to_owned().serialize(serializer),
            ArbArray::Owned(arr) => arr.serialize(serializer),
        }
    }
}

impl<'a, 'de, T> Deserialize<'de> for ArbArray<'a, T>
where
    T: Deserialize<'de> + Clone,
{
    fn deserialize<D>(deserializer: D) -> Result<ArbArray<'a, T>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(ArbArray::<T>::Owned(
            ArrayD::<T>::deserialize(deserializer)?.wrap(),
        ))
    }
}

impl<'a, T: Default> Default for ArbArray<'a, T> {
    fn default() -> Self {
        ArbArray::Owned(Default::default())
    }
}

// impl<'a, T: Clone> Clone for ArbArray<'a, T> {
//     fn clone(&self) -> Self {
//         match self{
//             ArbArray::View(arr) => {
//                 let ptr = arr.as_ptr();
//                 let shape = arr.raw_dim();
//                 unsafe{ArrViewD::<'a, T>::from_shape_ptr(shape, ptr).into()}
//             },
//             ArbArray::ViewMut(arr) => arr.to_owned().into(),
//             ArbArray::Owned(arr) => arr.clone().into(),
//         }
//     }
// }

macro_rules! match_arbarray {
    ($arb_array: expr, $arr: ident, $body: tt) => {
        match_arbarray!($arb_array, $arr, $body, (View, ViewMut, Owned))
    };

    ($arb_array: expr, $arr: ident, $body: tt, ($($arm: ident),*)) => {
        match $arb_array {
            $(ArbArray::$arm($arr) => $body,)*
            _ => panic!("Invalid match of ArbArray")
        }
    };
}
pub(crate) use match_arbarray;

impl<'a, T> From<ArrViewD<'a, T>> for ArbArray<'a, T> {
    fn from(arr: ArrViewD<'a, T>) -> Self {
        ArbArray::View(arr)
    }
}

impl<'a, T> From<ArrViewMutD<'a, T>> for ArbArray<'a, T> {
    fn from(arr: ArrViewMutD<'a, T>) -> Self {
        ArbArray::ViewMut(arr)
    }
}

impl<T> From<ArrD<T>> for ArbArray<'_, T> {
    fn from(arr: ArrD<T>) -> Self {
        ArbArray::Owned(arr)
    }
}

use crate::from_py::PyValue;

#[derive(Debug)]
pub enum ArrOk<'a> {
    Bool(ArbArray<'a, bool>),
    Usize(ArbArray<'a, usize>),
    OpUsize(ArbArray<'a, Option<usize>>),
    F32(ArbArray<'a, f32>),
    F64(ArbArray<'a, f64>),
    I32(ArbArray<'a, i32>),
    I64(ArbArray<'a, i64>),
    String(ArbArray<'a, String>),
    Str(ArbArray<'a, &'a str>),
    Object(ArbArray<'a, PyValue>),
    DateTime(ArbArray<'a, DateTime>),
    TimeDelta(ArbArray<'a, TimeDelta>),
}

macro_rules! match_arr {
    ($dyn_arr: expr, $arr: ident, $body: tt) => {
        match $dyn_arr {
            ArrOk::Bool($arr) => $body,
            ArrOk::F32($arr) => $body,
            ArrOk::F64($arr) => $body,
            ArrOk::I32($arr) => $body,
            ArrOk::I64($arr) => $body,
            ArrOk::Usize($arr) => $body,
            ArrOk::OpUsize($arr) => $body,
            ArrOk::String($arr) => $body,
            ArrOk::Str($arr) => $body,
            ArrOk::Object($arr) => $body,
            ArrOk::DateTime($arr) => $body,
            ArrOk::TimeDelta($arr) => $body,
        }
    };
}
pub(crate) use match_arr;

impl<'a, T> ArbArray<'a, T> {
    #[allow(unreachable_patterns)]
    pub fn raw_dim(&self) -> IxDyn {
        match_arbarray!(self, a, { a.raw_dim() })
    }

    #[allow(unreachable_patterns)]
    pub fn ndim(&self) -> usize {
        match_arbarray!(self, a, { a.ndim() })
    }

    #[inline]
    pub fn is_owned(&self) -> bool {
        matches!(self, ArbArray::Owned(_))
    }

    #[allow(unreachable_patterns)]
    pub fn get_type(&self) -> &'static str {
        match self {
            ArbArray::Owned(_) => "Owned Array",
            ArbArray::ViewMut(_) => "ViewMut Array",
            ArbArray::View(_) => "View Array",
        }
    }

    #[allow(unreachable_patterns)]
    pub fn strides(&self) -> &[isize] {
        match_arbarray!(self, a, { a.strides() })
    }

    #[allow(unreachable_patterns)]
    pub fn as_ptr(&self) -> *const T {
        match_arbarray!(self, a, { a.as_ptr() })
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        match_arbarray!(self, a, { a.as_mut_ptr() }, (ViewMut, Owned))
    }

    /// Cast the dtype of the array without copy.
    ///
    /// # Safety
    ///
    /// The size of `T` and `T2` must be the same
    pub unsafe fn into_dtype<T2>(self) -> ArbArray<'a, T2> {
        std::mem::transmute(self)
    }

    pub fn to_owned(self) -> ArrD<T>
    where
        T: Clone,
    {
        match self {
            ArbArray::View(arr) => arr.to_owned(),
            ArbArray::ViewMut(arr) => arr.to_owned(),
            ArbArray::Owned(arr) => arr,
        }
    }

    pub fn into_owned_inner(self) -> Result<ArrD<T>, &'static str> {
        if let ArbArray::Owned(arr) = self {
            Ok(arr)
        } else {
            Err("ArbArray is not owned")
        }
    }

    /// Convert to f-continuous if there is no performance loss
    pub fn try_to_owned_f(self) -> ArrD<T>
    where
        T: Clone,
    {
        use std::mem::MaybeUninit;
        use utils::vec_uninit;
        match self {
            ArbArray::View(arr) => {
                if arr.t().is_standard_layout() {
                    arr.to_owned()
                } else {
                    let mut arr_f =
                        Array::from_shape_vec(arr.shape().f(), vec_uninit(arr.len())).unwrap();
                    arr_f.zip_mut_with(&arr, |out, v| *out = MaybeUninit::new(v.clone()));
                    unsafe { arr_f.assume_init().wrap() }
                }
            }
            ArbArray::ViewMut(arr) => {
                if arr.t().is_standard_layout() {
                    arr.to_owned()
                } else {
                    let mut arr_f =
                        Array::from_shape_vec(arr.shape().f(), vec_uninit(arr.len())).unwrap();
                    arr_f.zip_mut_with(&arr, |out, v| *out = MaybeUninit::new(v.clone()));
                    unsafe { arr_f.assume_init().wrap() }
                }
            }
            ArbArray::Owned(arr) => arr,
        }
    }

    /// create an array view of ArrOk.
    ///
    /// # Safety
    ///
    /// The data of the array view must exist.
    pub fn view(&self) -> ArrViewD<'_, T> {
        match self {
            ArbArray::View(arr) => arr.view(),
            ArbArray::ViewMut(arr) => arr.view(),
            ArbArray::Owned(arr) => arr.view(),
        }
    }
}

impl<'a> ArrOk<'a> {
    pub fn raw_dim(&self) -> IxDyn {
        match_arr!(self, a, { a.raw_dim() })
    }

    pub fn ndim(&self) -> usize {
        match_arr!(self, a, { a.ndim() })
    }

    pub fn get_type(&self) -> &'static str {
        match_arr!(self, a, { a.get_type() })
    }

    #[allow(unreachable_patterns)]
    pub fn as_ptr<T: GetDataType>(&self) -> *const T {
        // we have known the datatype of the enum ,so only one arm will be executed
        match_datatype_arm!(
            self,
            a,
            ArrOk,
            T,
            (Bool, F32, F64, I32, I64, Usize, Object, String, Str, DateTime, TimeDelta, OpUsize),
            { a.as_ptr() as *const T }
        )
    }

    #[allow(unreachable_patterns)]
    pub fn as_mut_ptr<T: GetDataType>(&mut self) -> *mut T {
        // we have known the datatype of the enum ,so only one arm will be executed
        match_datatype_arm!(
            self,
            a,
            ArrOk,
            T,
            (Bool, F32, F64, I32, I64, Usize, Object, String, Str, DateTime, TimeDelta, OpUsize),
            { a.as_mut_ptr() as *mut T }
        )
    }

    /// cast ArrOk to ArbArray.
    ///
    /// # Safety
    ///
    /// T must be the correct dtype.
    #[allow(unreachable_patterns)]
    pub unsafe fn downcast<T>(self) -> ArbArray<'a, T> {
        match_arr!(self, arr, {
            match_arbarray!(arr, a, { a.into_dtype::<T>().into() })
        })
    }

    /// create an array view of ArrOk.
    ///
    /// # Safety
    ///
    /// T must be the correct dtype and the data of the
    /// array view must exist.
    pub unsafe fn view<T>(&self) -> ArrViewD<'_, T> {
        match_arr!(self, arr, { arr.view().into_dtype::<T>() })
    }
}
