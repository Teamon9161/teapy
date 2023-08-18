use super::{ArrD, ArrViewD, ArrViewMutD, WrapNdarray};
use ndarray::{Array, ArrayD, IxDyn, ShapeBuilder};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

// #[derive(Debug)]
pub enum ArbArray<'a, T> {
    View(ArrViewD<'a, T>),
    ViewMut(ArrViewMutD<'a, T>),
    Owned(ArrD<T>),
}

impl<'a, T: std::fmt::Debug> std::fmt::Debug for ArbArray<'a, T> {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // match_arbarray!(self, a, { a.fmt(f) })
        match &self {
            ArbArray::View(a) => write!(f, "ArrayView({a:#?})"),
            ArbArray::ViewMut(a) => write!(f, "ArrayViewMut({a:#?})"),
            ArbArray::Owned(a) => write!(f, "ArrayOwned({a:#?})"),
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
