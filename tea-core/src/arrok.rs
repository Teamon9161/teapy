use std::fmt::Debug;
// use serde::Serialize;

use super::arbarray::ArbArray;
use super::view::ArrViewD;
#[cfg(any(feature = "concat", feature = "arw"))]
use crate::{own::Arr1, utils::CollectTrustedToVec};
use datatype::{match_datatype_arm, Cast, DataType, GetDataType, OptUsize, PyValue};
#[cfg(feature = "time")]
use datatype::{DateTime, TimeDelta};
use ndarray::{Axis, IxDyn, SliceArg};

#[cfg(feature = "arw")]
use crate::ArrView1;
#[cfg(feature = "option_dtype")]
use datatype::{OptBool, OptF32, OptF64, OptI32, OptI64};

#[derive(Clone)]
pub enum ArrOk<'a> {
    Bool(ArbArray<'a, bool>),
    U8(ArbArray<'a, u8>),
    U64(ArbArray<'a, u64>),
    Usize(ArbArray<'a, usize>),
    OptUsize(ArbArray<'a, OptUsize>),
    F32(ArbArray<'a, f32>),
    F64(ArbArray<'a, f64>),
    I32(ArbArray<'a, i32>),
    I64(ArbArray<'a, i64>),
    String(ArbArray<'a, String>),
    Str(ArbArray<'a, &'a str>),
    Object(ArbArray<'a, PyValue>),
    VecUsize(ArbArray<'a, Vec<usize>>),
    #[cfg(feature = "time")]
    DateTime(ArbArray<'a, DateTime>),
    #[cfg(feature = "time")]
    TimeDelta(ArbArray<'a, TimeDelta>),
    #[cfg(feature = "option_dtype")]
    OptBool(ArbArray<'a, OptBool>),
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
            _ => unimplemented!("Not supported dtype")
        }
    };

    ($enum: ident, $exprs: expr, $e: ident, $body: tt) => {
        {
            match_all!(
                $enum, $exprs, $e, $body,
                F32, F64, I32, I64, U8, U64, Bool, Usize, Str, String, Object, OptUsize, VecUsize,
                #[cfg(feature="time")] DateTime,
                #[cfg(feature="time")] TimeDelta,
                #[cfg(feature="option_dtype")] OptF32,
                #[cfg(feature="option_dtype")] OptF64,
                #[cfg(feature="option_dtype")] OptI32,
                #[cfg(feature="option_dtype")] OptI64,
                #[cfg(feature="option_dtype")] OptBool
            )
        }
    };

    ($enum: ident, ($exprs1: expr, $e1: ident, $($arm1: ident),*), ($exprs2: expr, $e2: ident, $($arm2: ident),*), $body: tt) => {
        match_all!($enum, $exprs1, $e1, {match_all!($enum, $exprs2, $e2, $body, $($arm2),*)}, $($arm1),*)
    };
}

#[macro_export]
macro_rules! match_arrok {
    (numeric $($tt: tt)*) => {
        match_all!(ArrOk, $($tt)*,
            F32, F64, I32, I64, U64, Usize, OptUsize,
            #[cfg(feature = "option_dtype")] OptF32,
            #[cfg(feature = "option_dtype")] OptF64,
            #[cfg(feature = "option_dtype")] OptI32,
            #[cfg(feature = "option_dtype")] OptI64
        )
    };
    (int $($tt: tt)*) => {match_all!(ArrOk, $($tt)*, I32, I64, Usize)};
    (float $($tt: tt)*) => {match_all!(ArrOk, $($tt)*, F32, F64)};
    (bool $($tt: tt)*) => {match_all!(ArrOk, $($tt)*, Bool)};
    (hash $($tt: tt)*) => {match_all!(ArrOk, $($tt)*, I32, I64, U64, Usize, String, Str, #[cfg(feature="time")] DateTime, Bool, U8, U64)};
    (tphash $($tt: tt)*) => {match_all!(ArrOk, $($tt)*, F32, F64, I32, I64, U64, Usize, String, Str, #[cfg(feature="time")] DateTime, Bool, U8, U64)};
    (castable $($tt: tt)*) => {match_all!(
        ArrOk, $($tt)*,
        F32, F64, I32, I64, U64, Usize, String,
        Bool, OptUsize,
        #[cfg(feature="time")] DateTime,
        #[cfg(feature="time")] TimeDelta,
        #[cfg(feature = "option_dtype")] OptF32,
        #[cfg(feature = "option_dtype")] OptF64,
        #[cfg(feature = "option_dtype")] OptI32,
        #[cfg(feature = "option_dtype")] OptI64
    )};
    (nostr $($tt: tt)*) => {match_all!(
        ArrOk, $($tt)*,
        F32, F64, I32, I64, U64, Usize, String, U8, Bool, OptUsize, VecUsize, Object,
        #[cfg(feature="time")] DateTime,
        #[cfg(feature="time")] TimeDelta,
        #[cfg(feature = "option_dtype")] OptBool,
        #[cfg(feature = "option_dtype")] OptF32,
        #[cfg(feature = "option_dtype")] OptF64,
        #[cfg(feature = "option_dtype")] OptI32,
        #[cfg(feature = "option_dtype")] OptI64
    )};
    (pyelement $($tt: tt)*) => {match_all!(ArrOk, $($tt)*, F32, F64, I32, I64, U64, Usize, Bool, Object)};
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
    pub fn prepare(&mut self) {
        match_arrok!(self, a, { a.prepare() })
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
            unsafe { std::mem::transmute::<ArrOk<'a>, ArrOk<'b>>(a) }
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
            ArrOk::U64(a) => a.into(),
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

    #[allow(unreachable_patterns, clippy::missing_transmute_annotations)]
    #[inline]
    pub fn view(&self) -> ArrOk<'_> {
        match self {
            ArrOk::Bool(arr) => arr.view().into(),
            ArrOk::F32(arr) => arr.view().into(),
            ArrOk::F64(arr) => arr.view().into(),
            ArrOk::I32(arr) => arr.view().into(),
            ArrOk::I64(arr) => arr.view().into(),
            ArrOk::U64(arr) => arr.view().into(),
            ArrOk::Usize(arr) => arr.view().into(),
            ArrOk::OptUsize(arr) => arr.view().into(),
            ArrOk::String(arr) => arr.view().into(),
            ArrOk::Str(arr) => unsafe {
                std::mem::transmute::<_, ArrViewD<'_, &'_ str>>(arr.view()).into()
            },
            ArrOk::Object(arr) => arr.view().into(),
            ArrOk::VecUsize(arr) => arr.view().into(),
            #[cfg(feature = "time")]
            ArrOk::DateTime(arr) => arr.view().into(),
            #[cfg(feature = "time")]
            ArrOk::TimeDelta(arr) => arr.view().into(),
            #[cfg(feature = "option_dtype")]
            ArrOk::OptF64(arr) => arr.view().into(),
            #[cfg(feature = "option_dtype")]
            ArrOk::OptF32(arr) => arr.view().into(),
            #[cfg(feature = "option_dtype")]
            ArrOk::OptI32(arr) => arr.view().into(),
            #[cfg(feature = "option_dtype")]
            ArrOk::OptI64(arr) => arr.view().into(),
            #[cfg(feature = "option_dtype")]
            ArrOk::OptBool(arr) => arr.view().into(),
            _ => unimplemented!("view is not implemented for this dtype"),
        }
        // match_arr!(&self, arr, { arr.view().into() })
    }
}

#[cfg(feature = "concat")]
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

#[cfg(feature = "concat")]
impl_same_dtype_concat_1d!(
    Bool,
    U8,
    U64,
    F32,
    F64,
    I32,
    I64,
    Usize,
    String,
    Str,
    Object,
    OptUsize,
    VecUsize,
    #[cfg(feature = "time")]
    DateTime,
    #[cfg(feature = "time")]
    TimeDelta,
    #[cfg(feature = "option_dtype")]
    OptF32,
    #[cfg(feature = "option_dtype")]
    OptF64,
    #[cfg(feature = "option_dtype")]
    OptI32,
    #[cfg(feature = "option_dtype")]
    OptI64,
    #[cfg(feature = "option_dtype")]
    OptBool
);

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

#[cfg(feature = "arw")]
macro_rules! impl_from_arrow {
    ($([$arrow_dt: ident, $arrow_array: ident, $real: ty]),*) => {
        impl<'a> ArrOk<'a> {
            // pub fn from_arrow_vec(arr_vec: Vec<Box<dyn arrow::array::Array>>) -> ArrOk<'a> {
            //     use arrow::datatypes::DataType as ArrowDT;
            //     if arr_vec.is_empty() {
            //         return Default::default()
            //     }
            //     let arr = &arr_vec[0];
            //     match arr.data_type() {
            //         $(ArrowDT::$arrow_dt => {
            //             ArbArray::<'a, $real>::ArrowChunk(arr_vec).into()
            //         }),*
            //         ArrowDT::Int32 = > ArbArray::<'a, i32>::ArrowChunk(arr_vec).into(),
            //         ArrowDT::Int64 = > ArbArray::<'a, i64>::ArrowChunk(arr_vec).into(),
            //         _ => unimplemented!()
            //     }
            // }

            pub fn from_arrow(arr: Box<dyn arrow::array::Array>) -> ArrOk<'a> {
                use arrow::datatypes::DataType as ArrowDT;
                use crate::{prelude::ViewOnBase, datatype::{GetNone, Number}};
                use arrow::array::PrimitiveArray;
                match arr.data_type() {
                    $(ArrowDT::$arrow_dt => {
                        let a = arr.as_any().downcast_ref::<PrimitiveArray<$real>>().unwrap();
                        let data: &[$real] = a.values();
                        let nulls = a.validity();
                        if nulls.is_some() {
                            let data = a.into_iter().map(|v| v.cloned().unwrap_or_else(|| <$real>::none())).collect_trusted();
                            Arr1::from_vec(data).to_dimd().into()
                        } else {
                            let view: ArrViewD<'a, $real> = unsafe {
                                std::mem::transmute(ArrView1::from_slice(data.len(), data).to_dimd())
                            };
                            let out = ViewOnBase::new_from_arrow(arr, view);
                            out.into()
                        }

                    }),*
                    ArrowDT::Int32 => {
                        let a = arr.as_any().downcast_ref::<PrimitiveArray<i32>>().unwrap();
                        let data: &[i32] = a.values();
                        let nulls = a.validity();
                        if nulls.is_some() {
                            let data = a.into_iter().map(|v| v.map(|v| v.f64()).unwrap_or(f64::NAN)).collect_trusted();
                            Arr1::from_vec(data).to_dimd().into()
                        } else {
                            let view: ArrViewD<'a, i32> = unsafe {
                                std::mem::transmute(ArrView1::from_slice(data.len(), data).to_dimd())
                            };
                            let out = ViewOnBase::new_from_arrow(arr, view);
                            out.into()
                        }

                    },
                    ArrowDT::Int64 => {
                        let a = arr.as_any().downcast_ref::<PrimitiveArray<i64>>().unwrap();
                        let data: &[i64] = a.values();
                        let nulls = a.validity();
                        if nulls.is_some() {
                            let data = a.into_iter().map(|v| v.map(|v| v.f64()).unwrap_or(f64::NAN)).collect_trusted();
                            Arr1::from_vec(data).to_dimd().into()
                        } else {
                            let view: ArrViewD<'a, i64> = unsafe {
                                std::mem::transmute(ArrView1::from_slice(data.len(), data).to_dimd())
                            };
                            let out = ViewOnBase::new_from_arrow(arr, view);
                            out.into()
                        }

                    },
                    ArrowDT::Boolean => {
                        let a = arr.as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
                        let (data, start, len) = a.values().as_slice();
                        if start != 0 {
                            unreachable!("start index of the boolean is not 0")
                        }
                        let nulls = a.validity();
                        if nulls.is_some() {
                            unreachable!("can not read feather with null bool");
                        } else {
                            let view: ArrViewD<'a, u8> = unsafe {
                                std::mem::transmute(ArrView1::from_slice(len, data).to_dimd())
                            };
                            let out = ViewOnBase::new_from_arrow(arr, view);
                            out.into()
                        }
                    },
                    ArrowDT::Utf8 => {
                        // todo: remove unneeded clone here
                        let a = arr.as_any().downcast_ref::<arrow::array::Utf8Array<i32>>().unwrap();
                        let data = a.into_iter().map(|s| s.map(|s| s.to_string()).unwrap_or_default()).collect_trusted();
                        Arr1::from_vec(data).to_dimd().into()
                    },
                    ArrowDT::LargeUtf8 => {
                        let a = arr.as_any().downcast_ref::<arrow::array::Utf8Array<i64>>().unwrap();
                        let data = a.into_iter().map(|s| s.map(|s| s.to_string()).unwrap_or_default()).collect_trusted();
                        Arr1::from_vec(data).to_dimd().into()
                    }
                    #[cfg(feature="time")]
                    ArrowDT::Timestamp(arw_unit, arw_tz) => {
                        use arrow::datatypes::TimeUnit;
                        assert!(arw_tz.is_none(), "Timezone is not supported yet");
                        let a = arr.as_any().downcast_ref::<PrimitiveArray<i64>>().unwrap();
                        let data = match arw_unit {
                            TimeUnit::Second => {
                                a.into_iter().map(|v| {
                                    if let Some(v) = v {
                                        DateTime::from_timestamp_opt(*v, 0)
                                    } else {
                                        None.into()
                                    }
                                }).collect_trusted()
                            },
                            TimeUnit::Millisecond => {
                                a.into_iter().map(|v| {
                                    if let Some(v) = v {
                                        DateTime::from_timestamp_ms(*v).unwrap_or_default()
                                    } else {
                                        DateTime(None)
                                    }
                                }).collect_trusted()
                            },
                            TimeUnit::Microsecond => {
                                a.into_iter().map(|v| {
                                    if let Some(v) = v {
                                        DateTime::from_timestamp_us(*v).unwrap_or_default()
                                    } else {
                                        DateTime(None)
                                    }
                                }).collect_trusted()
                            },
                            TimeUnit::Nanosecond => {
                                a.into_iter().map(|v| {
                                    if let Some(v) = v {
                                        DateTime::from_timestamp_ns(*v).unwrap_or_default()
                                    } else {
                                        DateTime(None)
                                    }
                                }).collect_trusted()
                            },
                        };
                        Arr1::from_vec(data).to_dimd().into()
                    }
                    _ => unimplemented!("Arrow datatype {:?} is not supported yet", arr.data_type())
                }
            }
        }
    };
}

#[cfg(feature = "arw")]
impl_from_arrow!(
    [Float32, Float32Array, f32],
    [Float64, Float64Array, f64] // [Int32, Int32Array, i32], [Int64, Int64Array, i64]
);

macro_rules! impl_arrok_cast {
    ($($(#[$meta: meta])? $T: ty: $cast_func: ident),*) => {
        $(
            $(#[$meta])?
            impl<'a> Cast<ArbArray<'a, $T>> for ArrOk<'a>
            {
                #[inline]
                fn cast(self) -> ArbArray<'a, $T> {
                    match_arrok!(self, a, { a.cast::<$T>() },
                        U8, F32, F64, I32, I64, U64, Usize, Bool, OptUsize, String, Str, Object,
                        #[cfg(feature="time")] DateTime,
                        #[cfg(feature="time")] TimeDelta,
                        #[cfg(feature = "option_dtype")] OptF32,
                        #[cfg(feature = "option_dtype")] OptF64,
                        #[cfg(feature = "option_dtype")] OptI32,
                        #[cfg(feature = "option_dtype")] OptI64,
                        #[cfg(feature = "option_dtype")] OptBool
                    )
                }
            }

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
    u8: cast_u8,
    i32: cast_i32,
    i64: cast_i64,
    f32: cast_f32,
    f64: cast_f64,
    u64: cast_u64,
    usize: cast_usize,
    bool: cast_bool,
    String: cast_string,
    #[cfg(feature="time")]
    DateTime: cast_datetime_default,
    OptUsize: cast_optusize,
    #[cfg(feature="time")]
    TimeDelta: cast_timedelta,
    #[cfg(feature = "option_dtype")]
    OptF32: cast_optf32,
    #[cfg(feature = "option_dtype")]
    OptF64: cast_optf64,
    #[cfg(feature = "option_dtype")]
    OptI32: cast_opti32,
    #[cfg(feature = "option_dtype")]
    OptI64: cast_opti64,
    #[cfg(feature = "option_dtype")]
    OptBool: cast_optbool

);
