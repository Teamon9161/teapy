use std::fmt::Debug;

use super::arbarray::ArbArray;
use super::view::ArrViewD;
use crate::{
    match_datatype_arm, Arr1, Cast, CollectTrustedToVec, DataType, DateTime, GetDataType, OptUsize,
    PyValue, TimeDelta,
};
use ndarray::{Axis, IxDyn, SliceArg};

#[cfg(feature = "option_dtype")]
use crate::datatype::{OptF32, OptF64, OptI32, OptI64};

pub enum ArrOk<'a> {
    Bool(ArbArray<'a, bool>),
    Usize(ArbArray<'a, usize>),
    OptUsize(ArbArray<'a, OptUsize>),
    F32(ArbArray<'a, f32>),
    F64(ArbArray<'a, f64>),
    I32(ArbArray<'a, i32>),
    I64(ArbArray<'a, i64>),
    String(ArbArray<'a, String>),
    Str(ArbArray<'a, &'a str>),
    Object(ArbArray<'a, PyValue>),
    DateTime(ArbArray<'a, DateTime>),
    TimeDelta(ArbArray<'a, TimeDelta>),
    VecUsize(ArbArray<'a, Vec<usize>>),
    #[cfg(feature = "option_dtype")]
    OptF64(ArbArray<'a, OptF64>),
    #[cfg(feature = "option_dtype")]
    OptF32(ArbArray<'a, OptF32>),
    #[cfg(feature = "option_dtype")]
    OptI32(ArbArray<'a, OptI32>),
    #[cfg(feature = "option_dtype")]
    OptI64(ArbArray<'a, OptI64>),
}

#[macro_export]
macro_rules! match_all {
    // select the match arm
    ($enum: ident, $exprs: expr, $e: ident, $body: tt, $($(#[$meta: meta])? $arm: ident),*) => {
        match $exprs {
            $($(#[$meta])? $enum::$arm($e) => $body,)*
            _ => unimplemented!("Not supported dtype in match_exprs")
        }
    };

    ($enum: ident, $exprs: expr, $e: ident, $body: tt) => {
        {
            #[cfg(feature="option_dtype")]
            macro_rules! inner_macro {
                () => {
                    match_all!($enum, $exprs, $e, $body, F32, F64, I32, I64, Bool, Usize, Str, String, Object, DateTime, TimeDelta, VecUsize, OptF32, OptF64, OptI32, OptI64, OptUsize)
                };
            }

            #[cfg(not(feature="option_dtype"))]
            macro_rules! inner_macro {
                () => {
                    match_all!($enum, $exprs, $e, $body, F32, F64, I32, I64, Bool, Usize, Str, String, Object, DateTime, TimeDelta, OptUsize, VecUsize)
                };
            }
            inner_macro!()
        }

    };

    ($enum: ident, ($exprs1: expr, $e1: ident, $($arm1: ident),*), ($exprs2: expr, $e2: ident, $($arm2: ident),*), $body: tt) => {
        match_all!($enum, $exprs1, $e1, {match_all!($enum, $exprs2, $e2, $body, $($arm2),*)}, $($arm1),*)
    };
}

#[macro_export]
macro_rules! match_arrok {
    // (numeric $($tt: tt)*) => {match_all!(ArrOk, $($tt)*, F32, F64, I32, I64, Usize)};
    (numeric $($tt: tt)*) => {
        match_all!(ArrOk, $($tt)*,
            F32, F64, I32, I64, Usize, OptUsize,
            #[cfg(feature = "option_dtype")] OptF32,
            #[cfg(feature = "option_dtype")] OptF64,
            #[cfg(feature = "option_dtype")] OptI32,
            #[cfg(feature = "option_dtype")] OptI64
        )
    };
    (numeric2 $($tt: tt)*) => {match_all!(ArrOk, $($tt)*, F32, F64, I32, I64, Usize, OptUsize, #[cfg(feature = "option_dtype")] OptF32, #[cfg(feature = "option_dtype")] OptF64, #[cfg(feature = "option_dtype")] OptI32, #[cfg(feature = "option_dtype")] OptI64)};
    (int $($tt: tt)*) => {match_all!(ArrOk, $($tt)*, I32, I64, Usize)};//, OptUsize, #[cfg(feature = "option_dtype")] OptI32, #[cfg(feature = "option_dtype")] OptI64)};
    (float $($tt: tt)*) => {match_all!(ArrOk, $($tt)*, F32, F64)};//, #[cfg(feature = "option_dtype")] OptF32, #[cfg(feature = "option_dtype")] OptF64)};
    (hash $($tt: tt)*) => {match_all!(ArrOk, $($tt)*, F32, F64, I32, I64, Usize, String, Str, DateTime, Bool)};
    (castable $($tt: tt)*) => {match_all!(
        ArrOk, $($tt)*,
        F32, F64, I32, I64, Usize, String, DateTime, Bool, OptUsize,
        #[cfg(feature = "option_dtype")] OptF32,
        #[cfg(feature = "option_dtype")] OptF64,
        #[cfg(feature = "option_dtype")] OptI32,
        #[cfg(feature = "option_dtype")] OptI64
        )};
    (pyelement $($tt: tt)*) => {match_all!(ArrOk, $($tt)*, F32, F64, I32, I64, Usize, Bool, Object)};
    ($($tt: tt)*) => {match_all!(ArrOk, $($tt)*)};
}

impl<'a> Debug for ArrOk<'a> {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match_arrok!(self, arbarray, { arbarray.fmt(f) })
    }
}

impl<'a> ArrOk<'a> {
    #[allow(unreachable_patterns)]
    #[inline]
    pub fn raw_dim(&self) -> IxDyn {
        match_arrok!(self, a, { a.raw_dim() })
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn ndim(&self) -> usize {
        match_arrok!(self, a, { a.ndim() })
    }

    #[allow(unreachable_patterns, clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        match_arrok!(self, a, { a.len() })
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn len_of(&self, axis: Axis) -> usize {
        match_arrok!(self, a, { a.len_of(axis) })
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn shape(&self) -> &[usize] {
        match_arrok!(self, a, { a.shape() })
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn norm_axis(&self, axis: i32) -> Axis {
        match_arrok!(self, a, { a.norm_axis(axis) })
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn get_type(&self) -> &'static str {
        match_arrok!(self, a, { a.get_type() })
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn dtype(&self) -> DataType {
        match_arrok!(self, a, { a.dtype() })
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn deref(&self) -> ArrOk<'_> {
        match_arrok!(self, a, { a.deref().into() })
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn slice<I: SliceArg<IxDyn>>(&self, info: I) -> ArrOk<'_> {
        match_arrok!(self, a, { a.slice(info).into() })
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn is_owned(&self) -> bool {
        match_arrok!(self, a, { a.is_owned() })
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn into_owned<'b>(self) -> ArrOk<'b> {
        match_arrok!(self, a, {
            let a: ArrOk<'a> = a.into_owned().into();
            // this is safe because we only need it for &str type
            // and the lifetime of str should be longer than both
            // 'a and 'b
            // remove this transmute once we add a different lifetime
            // for &str datatype
            unsafe { std::mem::transmute(a) }
        })
    }

    #[inline]
    pub fn is_float(&self) -> bool {
        self.dtype().is_float()
    }

    #[inline]
    pub fn is_int(&self) -> bool {
        self.dtype().is_int()
    }

    #[inline]
    pub fn cast_float(self) -> Self {
        match self {
            ArrOk::I32(a) => a.cast::<f32>().into(),
            ArrOk::I64(a) => a.cast::<f64>().into(),
            ArrOk::Usize(a) => a.cast::<f64>().into(),
            ArrOk::OptUsize(a) => a.cast::<f64>().into(),
            #[cfg(feature = "option_dtype")]
            ArrOk::OptI32(a) => a.cast::<OptF32>().into(),
            #[cfg(feature = "option_dtype")]
            ArrOk::OptI64(a) => a.cast::<OptF64>().into(),
            _ => self.cast_f64().into(),
        }
    }

    #[inline]
    pub fn cast_int(self) -> Self {
        match self {
            ArrOk::F32(a) => a.cast::<i32>().into(),
            ArrOk::F64(a) => a.cast::<i64>().into(),
            ArrOk::Usize(a) => a.into(),
            ArrOk::OptUsize(a) => a.into(),
            #[cfg(feature = "option_dtype")]
            ArrOk::OptF32(a) => a.cast::<OptI32>().into(),
            #[cfg(feature = "option_dtype")]
            ArrOk::OptF64(a) => a.cast::<OptI64>().into(),
            _ => self.cast_i64().into(),
        }
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn as_ptr<T: GetDataType>(&self) -> *const T {
        // we have known the datatype of the enum ,so only one arm will be executed
        match_datatype_arm!(
            all
            self,
            a,
            ArrOk,
            T,
            { a.as_ptr() as *const T }
        )
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn as_mut_ptr<T: GetDataType>(&mut self) -> *mut T {
        // we have known the datatype of the enum ,so only one arm will be executed
        match_datatype_arm!(
            all
            self,
            a,
            ArrOk,
            T,
            // (Bool, F32, F64, I32, I64, Usize, Object, String, Str, DateTime, TimeDelta, OpUsize, OpF64),
            { a.as_mut_ptr() as *mut T }
        )
    }

    // /// reinterpret ArrOk to ArbArray<'a, T> directly.
    // ///
    // /// # Safety
    // ///
    // /// T must be the correct dtype.
    // #[allow(unreachable_patterns)]
    // pub unsafe fn downcast<T>(self) -> ArbArray<'a, T> {
    //     match_arrok!(self, arr, {
    //         match_arbarray!(arr, a, { a.into_dtype::<T>().into() }, (View, ViewMut, Owned))
    //     })
    // }

    #[inline]
    pub fn cast_str(self) -> ArbArray<'a, &'a str> {
        match_arrok!(self, a, { a }, Str)
    }

    #[inline]
    pub fn cast_object(self) -> ArbArray<'a, PyValue> {
        match_arrok!(self, a, { a }, Object)
    }

    #[inline]
    pub fn cast_vecusize(self) -> ArbArray<'a, Vec<usize>> {
        match_arrok!(self, a, { a }, VecUsize)
    }

    #[inline]
    pub fn as_float(&self) -> ArrOk<'_> {
        if self.dtype().is_float() {
            self.deref()
        } else {
            self.deref().cast_float()
        }
    }

    #[inline]
    pub fn as_int(&self) -> ArrOk<'_> {
        if self.dtype().is_int() {
            self.deref()
        } else {
            self.deref().cast_int()
        }
    }

    // /// create an array view of ArrOk.
    // ///
    // /// # Safety
    // ///
    // /// T must be the correct dtype and the data of the
    // /// array view must exist.
    // #[allow(unreachable_patterns)]
    // pub unsafe fn view<T>(&self) -> ArrViewD<'_, T> {
    //     match_arrok!(self, arr, { arr.view().into_dtype::<T>() })
    // }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn view(&self) -> ArrOk<'_> {
        match self {
            ArrOk::Bool(arr) => arr.view().into(),
            ArrOk::F32(arr) => arr.view().into(),
            ArrOk::F64(arr) => arr.view().into(),
            ArrOk::I32(arr) => arr.view().into(),
            ArrOk::I64(arr) => arr.view().into(),
            ArrOk::Usize(arr) => arr.view().into(),
            ArrOk::OptUsize(arr) => arr.view().into(),
            ArrOk::String(arr) => arr.view().into(),
            ArrOk::Str(arr) => unsafe {
                std::mem::transmute::<_, ArrViewD<'_, &'_ str>>(arr.view()).into()
            },
            ArrOk::Object(arr) => arr.view().into(),
            ArrOk::DateTime(arr) => arr.view().into(),
            ArrOk::TimeDelta(arr) => arr.view().into(),
            ArrOk::VecUsize(arr) => arr.view().into(),
            #[cfg(feature = "option_dtype")]
            ArrOk::OptF64(arr) => arr.view().into(),
            #[cfg(feature = "option_dtype")]
            ArrOk::OptF32(arr) => arr.view().into(),
            #[cfg(feature = "option_dtype")]
            ArrOk::OptI32(arr) => arr.view().into(),
            #[cfg(feature = "option_dtype")]
            ArrOk::OptI64(arr) => arr.view().into(),
            _ => unimplemented!("view is not implemented for this dtype"),
        }
        // match_arr!(&self, arr, { arr.view().into() })
    }
}

macro_rules! impl_same_dtype_concat_1d {
    ($($(#[$meta: meta])? $arm: ident),*) => {
        impl<'a> ArrOk<'a> {
            #[inline]
            pub fn same_dtype_concat_1d(arr_vec: Vec<Self>) -> ArrOk<'a> {
                if arr_vec.is_empty() {
                    Default::default()
                } else {
                    let o1 = unsafe{arr_vec.get_unchecked(0)};
                    let ndim = o1.ndim();
                    use ArrOk::*;
                    match o1 {
                        $($(#[$meta])? $arm(_) => {
                            if ndim == 0 {
                                let out = arr_vec.into_iter().map(|a| {
                                    // let a = std::mem::transmute::<_, ArbArray<'a, bool>>(a).into_owned();
                                    let a = if let $arm(a) = a {a.into_owned()} else {unreachable!()};
                                    a.into_scalar().unwrap()
                                }).collect_trusted();
                                Arr1::from_vec(out).to_dimd().into()
                            } else {
                                let out = arr_vec.into_iter().map(|a| {
                                    let a = if let $arm(a) = a {a.into_owned()} else {unreachable!()};
                                    a.to_dim1().unwrap().0.into_raw_vec().into_iter()
                                }).flatten().collect();
                                Arr1::from_vec(out).to_dimd().into()
                            }
                        }),*
                        // _ => unimplemented!()
                    }

                }
            }
        }

    };
}

impl_same_dtype_concat_1d!(
    #[cfg(feature = "option_dtype")]
    OptF32,
    #[cfg(feature = "option_dtype")]
    OptF64,
    #[cfg(feature = "option_dtype")]
    OptI32,
    #[cfg(feature = "option_dtype")]
    OptI64,
    Bool,
    F32,
    F64,
    I32,
    I64,
    Usize,
    OptUsize,
    String,
    Str,
    Object,
    DateTime,
    TimeDelta,
    VecUsize
);

impl<'a> ToOwned for ArrOk<'a> {
    type Owned = ArrOk<'a>;
    #[allow(unreachable_patterns)]
    #[inline]
    fn to_owned(&self) -> Self::Owned {
        match_arrok!(self, a, { a.to_owned().into() })
    }
}

impl<'a> Cast<ArbArray<'a, &'a str>> for ArrOk<'a> {
    #[inline]
    fn cast(self) -> ArbArray<'a, &'a str> {
        match self {
            ArrOk::Str(e) => e,
            _ => unimplemented!("Cast to str is unimplemented"),
        }
    }
}

impl<'a> Cast<ArbArray<'a, PyValue>> for ArrOk<'a> {
    #[inline]
    fn cast(self) -> ArbArray<'a, PyValue> {
        match self {
            ArrOk::Object(e) => e,
            _ => unimplemented!("Cast to pyobject is unimplemented"),
        }
    }
}

impl<'a> Cast<ArbArray<'a, Vec<usize>>> for ArrOk<'a> {
    #[inline]
    fn cast(self) -> ArbArray<'a, Vec<usize>> {
        match self {
            ArrOk::VecUsize(e) => e,
            _ => unimplemented!("Cast to Vec<usize> is unimplemented"),
        }
    }
}

macro_rules! impl_arrok_cast {
    ($($(#[$meta: meta])? $T: ty: $cast_func: ident),*) => {
        $(
            $(#[$meta])?
            impl<'a> Cast<ArbArray<'a, $T>> for ArrOk<'a>
            {
                #[inline]
                fn cast(self) -> ArbArray<'a, $T> {
                    match self {
                        ArrOk::F32(e) => e.cast::<$T>(),
                        ArrOk::F64(e) => e.cast::<$T>(),
                        ArrOk::I32(e) => e.cast::<$T>(),
                        ArrOk::I64(e) => e.cast::<$T>(),
                        ArrOk::Bool(e) => e.cast::<i32>().cast::<$T>(),
                        ArrOk::Usize(e) => e.cast::<$T>(),
                        ArrOk::Str(e) => e.cast::<$T>(),
                        ArrOk::String(e) => e.cast::<$T>(),
                        // ArrOk::Object(e) => e.cast::<$T>(),
                        ArrOk::DateTime(e) => e.cast::<i64>().cast::<$T>(),
                        ArrOk::TimeDelta(e) => e.cast::<i64>().cast::<$T>(),
                        #[cfg(feature = "option_dtype")]
                        ArrOk::OptF64(e) => e.cast::<$T>(),
                        #[cfg(feature = "option_dtype")]
                        ArrOk::OptF32(e) => e.cast::<$T>(),
                        #[cfg(feature = "option_dtype")]
                        ArrOk::OptI32(e) => e.cast::<$T>(),
                        #[cfg(feature = "option_dtype")]
                        ArrOk::OptI64(e) => e.cast::<$T>(),
                        ArrOk::OptUsize(e) => e.cast::<$T>(),
                        _ => unimplemented!("Cast to this dtype is unimplemented"),
                    }
                }
            }

            // $(#[$meta])?
            // impl<'a> Cast<ArrViewD<'a, $T>> for ArrOk<'a> {
            //     fn cast(self) -> ArrViewD<'a, $T> {
            //         let arb: ArbArray<'a, $T> = self.cast();
            //         unsafe {std::mem::transmute(arb.view())}
            //     }
            // }

            $(#[$meta])?
            impl<'a> ArrOk<'a>
            {
                #[inline]
                pub fn $cast_func(self) -> ArbArray<'a, $T> {
                    self.cast()
                }
            }

            $(#[$meta])?
            impl<'a, U: GetDataType + Cast<$T> + Clone> ArbArray<'a, U>
            {
                #[inline]
                pub fn $cast_func(self) -> ArbArray<'a, $T> {
                    self.cast::<$T>()
                }
            }
        )*
    };
}
impl_arrok_cast!(
    i32: cast_i32,
    i64: cast_i64,
    f32: cast_f32,
    f64: cast_f64,
    usize: cast_usize,
    bool: cast_bool,
    String: cast_string,
    DateTime: cast_datetime_default,
    OptUsize: cast_optusize,
    TimeDelta: cast_timedelta,
    #[cfg(feature = "option_dtype")]
    OptF32: cast_optf32,
    #[cfg(feature = "option_dtype")]
    OptF64: cast_optf64,
    #[cfg(feature = "option_dtype")]
    OptI32: cast_opti32,
    #[cfg(feature = "option_dtype")]
    OptI64: cast_opti64
);
