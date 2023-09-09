use crate::{Cast, DataType, GetDataType, TpResult};

use super::{ArrD, ArrViewD, ArrViewMutD, WrapNdarray};
use ndarray::{s, Array, ArrayD, Axis, IxDyn, NewAxis, ShapeBuilder, SliceArg};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::marker::PhantomPinned;
use std::ops::Deref;
use std::pin::Pin;

pub enum ArbArray<'a, T> {
    View(ArrViewD<'a, T>),
    ViewMut(ArrViewMutD<'a, T>),
    Owned(ArrD<T>),
    ViewOnBase(Pin<Box<ViewOnBase<'a, T>>>),
}

pub struct ViewOnBase<'a, T> {
    pub base: ArbArray<'a, T>,
    pub view: Option<ArrViewD<'a, T>>,
    _pin: PhantomPinned,
}

impl<'a, T> Deref for ViewOnBase<'a, T> {
    type Target = ArrViewD<'a, T>;

    fn deref(&self) -> &Self::Target {
        self.view.as_ref().unwrap()
    }
}

impl<'a, T: Clone> ToOwned for ArbArray<'a, T> {
    type Owned = ArbArray<'a, T>;

    fn to_owned(&self) -> Self {
        self.view().to_owned().into()
    }
}

// impl<'a, T> From<ArbArray<'a, T>> for ViewOnBase<'a, T> {
//     fn from(base: ArbArray<'a, T>) -> Self {
//         Self {
//             base: base,
//             view: None,
//             _pin: PhantomPinned,
//         }
//     }
// }

// impl<'a, T> From<ArrViewD<'a, T>> for ViewOnBase<'a, T> {
//     fn from(base: ArrViewD<'a, T>) -> Self {
//         Self {
//             base: base.into(),
//             view: None,
//             _pin: PhantomPinned,
//         }
//     }
// }

// impl<'a, T> From<ArrD<T>> for ViewOnBase<'a, T> {
//     fn from(base: ArrD<T>) -> Self {
//         Self {
//             base: base,
//             view: None,
//             // _pin: PhantomPinned,
//         }
//     }
// }

// impl<'a, T> Pin<ViewOnBase<'a, T>> {
//     pub fn set_view(&mut self, view: ArrViewD<'a, T>) {
//         // 这里其实安全的，因为修改一个字段不会转移整个结构体的所有权
//         unsafe {
//             let mut_ref: Pin<&mut Self> = Pin::as_mut(&mut self);
//             Pin::get_unchecked_mut(mut_ref).view = view;
//         }
//     }
// }

impl<'a, T> ViewOnBase<'a, T> {
    pub fn view(&self) -> &ArrViewD<'a, T> {
        self.view.as_ref().unwrap()
    }

    pub fn new(arr: ArbArray<'a, T>, view: ArrViewD<'a, T>) -> Pin<Box<Self>> {
        let out = Self {
            base: arr,
            view: Some(view),
            _pin: PhantomPinned,
        };
        Box::pin(out)
    }

    // /// Cast the dtype of the array without copy.
    // ///
    // /// # Safety
    // ///
    // /// The size of `T` and `T2` must be the same
    // pub unsafe fn into_dtype<T2>(self) -> ViewOnBase<'a, T2> {
    //     use std::mem;
    //     if mem::size_of::<T>() == mem::size_of::<T2>() {
    //         let out = mem::transmute_copy(&self);
    //         mem::forget(self);
    //         out
    //     } else {
    //         panic!("the size of new type is different when into_dtype")
    //     }
    // }
}

impl<'a, T: std::fmt::Debug> std::fmt::Debug for ArbArray<'a, T> {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            ArbArray::View(a) => write!(f, "ArrayView({a:#?})"),
            ArbArray::ViewMut(a) => write!(f, "ArrayViewMut({a:#?})"),
            ArbArray::Owned(a) => write!(f, "ArrayOwned({a:#?})"),
            ArbArray::ViewOnBase(a) => write!(f, "ViewonBase({:#?})", a.view()),
        }
    }
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
            ArbArray::ViewOnBase(vb) => vb.view().to_owned().serialize(serializer),
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
        match_arbarray!($arb_array, $arr, $body, (View, ViewMut, Owned, ViewOnBase))
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

impl<'a, T> From<Pin<Box<ViewOnBase<'a, T>>>> for ArbArray<'a, T> {
    fn from(vb: Pin<Box<ViewOnBase<'a, T>>>) -> Self {
        ArbArray::ViewOnBase(vb)
    }
}

impl<'a, T> ArbArray<'a, T> {
    // #[allow(unreachable_patterns)]
    pub fn raw_dim(&self) -> IxDyn {
        self.view().raw_dim()
        // match_arbarray!(self, a, { a.raw_dim() })
    }

    pub fn dtype(&self) -> DataType
    where
        T: GetDataType,
    {
        T::dtype()
    }

    #[allow(unreachable_patterns)]
    pub fn ndim(&self) -> usize {
        match_arbarray!(self, a, { a.ndim() })
    }

    #[allow(unreachable_patterns)]
    pub fn shape(&self) -> &[usize] {
        match_arbarray!(self, a, { a.shape() })
    }

    #[allow(unreachable_patterns, clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match_arbarray!(self, a, { a.len() })
    }

    #[allow(unreachable_patterns)]
    pub fn len_of(&self, axis: Axis) -> usize {
        match_arbarray!(self, a, { a.len_of(axis) })
    }

    #[allow(unreachable_patterns)]
    pub fn norm_axis(&self, axis: i32) -> Axis {
        match_arbarray!(self, a, { a.norm_axis(axis) })
    }

    pub fn deref(&self) -> ArbArray<'_, T> {
        match &self {
            ArbArray::View(view) => view.view().into(),
            ArbArray::ViewMut(view) => view.view().into(),
            ArbArray::Owned(arr) => arr.view().into(),
            ArbArray::ViewOnBase(vb) => vb.view().view().into(),
        }
    }

    #[inline]
    pub fn is_owned(&self) -> bool {
        matches!(self, ArbArray::Owned(_))
    }

    #[inline]
    pub fn is_float(&self) -> bool
    where
        T: GetDataType,
    {
        self.dtype().is_float()
    }

    #[inline]
    pub fn is_int(&self) -> bool
    where
        T: GetDataType,
    {
        self.dtype().is_int()
    }

    /// Cast to another type
    #[inline]
    pub fn cast<T2>(self) -> ArbArray<'a, T2>
    where
        T: GetDataType + Cast<T2> + Clone,
        T2: GetDataType + Clone + 'a,
    {
        if T::dtype() == T2::dtype() {
            // safety: T and T2 are the same type
            unsafe { std::mem::transmute(self) }
        } else {
            self.view().cast::<T2>().into()
        }
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn get_type(&self) -> &'static str {
        match self {
            ArbArray::Owned(_) => "Owned Array",
            ArbArray::ViewMut(_) => "ViewMut Array",
            ArbArray::View(_) => "View Array",
            ArbArray::ViewOnBase(_) => "View Array",
        }
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn strides(&self) -> &[isize] {
        match_arbarray!(self, a, { a.strides() })
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        match_arbarray!(self, a, { a.as_ptr() })
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        match_arbarray!(self, a, { a.as_mut_ptr() }, (ViewMut, Owned))
    }

    #[inline]
    pub fn no_dim0(self) -> Self
    where
        T: Clone,
    {
        if self.ndim() == 0 {
            ArbArray::Owned(self.into_owned().0.slice_move(s!(NewAxis)).wrap().to_dimd())
        } else {
            self
        }
    }

    #[inline]
    pub fn slice<'b, I: SliceArg<IxDyn>>(&'b self, info: I) -> ArbArray<'b, T>
    where
        'a: 'b,
    {
        // safety: the view has lifetime 'b, this is safe as self exists
        let view: ArrViewD<'b, T> =
            unsafe { std::mem::transmute(self.view().slice(info).to_dimd()) };
        view.into()
    }

    /// Cast the dtype of the array without copy.
    ///
    /// # Safety
    ///
    /// The size of `T` and `T2` must be the same
    #[inline]
    pub unsafe fn into_dtype<T2>(self) -> ArbArray<'a, T2> {
        std::mem::transmute(self)
    }

    #[inline]
    pub fn into_owned(self) -> ArrD<T>
    where
        T: Clone,
    {
        match self {
            ArbArray::View(arr) => arr.to_owned(),
            ArbArray::ViewMut(arr) => arr.to_owned(),
            ArbArray::Owned(arr) => arr,
            ArbArray::ViewOnBase(vb) => vb.view().to_owned(),
        }
    }

    pub fn into_owned_inner(self) -> TpResult<ArrD<T>> {
        if let ArbArray::Owned(arr) = self {
            Ok(arr)
        } else {
            Err("ArbArray is not owned".into())
        }
    }

    /// Convert to f-continuous if there is no performance loss
    pub fn try_to_owned_f(self) -> ArrD<T>
    where
        T: Clone,
    {
        use crate::utils::vec_uninit;
        use std::mem::MaybeUninit;
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
            ArbArray::ViewOnBase(vb) => {
                let arr = vb.view();
                ArbArray::View(arr.view()).try_to_owned_f()
            }
        }
    }

    /// create an array view of ArrOk.
    #[inline]
    pub fn view(&self) -> ArrViewD<'_, T> {
        match self {
            ArbArray::View(arr) => arr.view(),
            ArbArray::ViewMut(arr) => arr.view(),
            ArbArray::Owned(arr) => arr.view(),
            ArbArray::ViewOnBase(vb) => vb.view().view(),
        }
    }

    pub fn viewmut(&mut self) -> ArrViewMutD<'_, T>
    where
        T: Clone,
    {
        match self {
            ArbArray::ViewMut(arr) => arr.view_mut(),
            ArbArray::Owned(arr) => arr.view_mut(),
            ArbArray::View(arr_view) => {
                let arr = arr_view.to_owned();
                *self = ArbArray::Owned(arr);
                self.viewmut()
            }
            ArbArray::ViewOnBase(vb) => {
                let arr = vb.view().to_owned();
                *self = ArbArray::Owned(arr);
                self.viewmut()
            }
        }
    }

    pub fn concat<'b>(self, other: Vec<Self>, axis: Axis) -> ArbArray<'b, T>
    where
        T: Clone,
    {
        let arrs = std::iter::once(self.view().0)
            .chain(other.iter().map(|o| o.view().0))
            .collect::<Vec<_>>();
        ArbArray::Owned(ndarray::concatenate(axis, &arrs).unwrap().wrap())
    }

    pub fn stack<'b>(&self, other: Vec<Self>, axis: Axis) -> ArbArray<'b, T>
    where
        T: Clone,
    {
        let arrs = std::iter::once(self.view().0)
            .chain(other.iter().map(|o| o.view().0))
            .collect::<Vec<_>>();
        ArbArray::Owned(ndarray::stack(axis, &arrs).unwrap().wrap())
    }
}
