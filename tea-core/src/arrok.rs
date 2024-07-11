use super::arbarray::ArbArray;
use super::py_dtype::Object;
#[cfg(feature = "arw")]
use super::view::ArrViewD;
#[cfg(any(feature = "concat", feature = "arw"))]
use crate::{own::Arr1, utils::CollectTrustedToVec};
use derive_more::From;
use ndarray::{Axis, IxDyn, SliceArg};
use std::fmt::Debug;
use tevec::prelude::*;

#[cfg(feature = "arw")]
use crate::ArrView1;

#[derive(Clone, From)]
pub enum ArrOk<'a> {
    Bool(ArbArray<'a, bool>),
    U8(ArbArray<'a, u8>),
    U64(ArbArray<'a, u64>),
    Usize(ArbArray<'a, usize>),
    OptUsize(ArbArray<'a, Option<usize>>),
    F32(ArbArray<'a, f32>),
    F64(ArbArray<'a, f64>),
    I32(ArbArray<'a, i32>),
    I64(ArbArray<'a, i64>),
    OptBool(ArbArray<'a, Option<bool>>),
    OptF32(ArbArray<'a, Option<f32>>),
    OptF64(ArbArray<'a, Option<f64>>),
    OptI32(ArbArray<'a, Option<i32>>),
    OptI64(ArbArray<'a, Option<i64>>),
    String(ArbArray<'a, String>),
    // Str(ArbArray<'a, &'a str>),
    Object(ArbArray<'a, Object>),
    VecUsize(ArbArray<'a, Vec<usize>>),
    #[cfg(feature = "time")]
    DateTimeMs(ArbArray<'a, DateTime<unit::Millisecond>>),
    #[cfg(feature = "time")]
    DateTimeUs(ArbArray<'a, DateTime<unit::Microsecond>>),
    #[cfg(feature = "time")]
    DateTimeNs(ArbArray<'a, DateTime<unit::Nanosecond>>),

    #[cfg(feature = "time")]
    TimeDelta(ArbArray<'a, TimeDelta>),
}

#[macro_export]
macro_rules! match_arrok {
    ($($tt: tt)*) => {
        $crate::match_enum!(ArrOk, $($tt)*)
    }
}

impl<'a> Debug for ArrOk<'a> {
    #[allow(unreachable_patterns)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match_arrok!(self; Dynamic(a) => { Ok(a.fmt(f)) },).unwrap()
    }
}

impl<'a> ArrOk<'a> {
    #[allow(unreachable_patterns)]
    #[inline]
    pub fn raw_dim(&self) -> IxDyn {
        match_arrok!(self; Dynamic(a) => { Ok(a.raw_dim()) },).unwrap()
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn ndim(&self) -> usize {
        match_arrok!(self; Dynamic(a) => { Ok(a.ndim()) },).unwrap()
    }

    #[allow(unreachable_patterns, clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        match_arrok!(self; Dynamic(a) => { Ok(a.len()) },).unwrap()
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn len_of(&self, axis: Axis) -> usize {
        match_arrok!(self; Dynamic(a) => { Ok(a.len_of(axis)) },).unwrap()
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn shape(&self) -> &[usize] {
        match_arrok!(self; Dynamic(a) => { Ok(a.shape()) },).unwrap()
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn norm_axis(&self, axis: i32) -> Axis {
        match_arrok!(self; Dynamic(a) => { Ok(a.norm_axis(axis)) },).unwrap()
    }

    #[allow(unreachable_patterns)]
    pub fn prepare(&mut self) {
        match_arrok!(self; Dynamic(a) => {
            a.prepare();
            Ok(())
        },)
        .unwrap()
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn get_type(&self) -> &'static str {
        match_arrok!(self; Dynamic(a) => { Ok(a.get_type()) },).unwrap()
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn dtype(&self) -> DataType {
        match_arrok!(self; Dynamic(a) => { Ok(a.dtype()) },).unwrap()
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn deref(&self) -> ArrOk<'_> {
        match_arrok!(self; Dynamic(a) => { Ok(a.deref().into()) },).unwrap()
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn slice<I: SliceArg<IxDyn>>(&self, info: I) -> ArrOk<'_> {
        match_arrok!(self; Dynamic(a) => { Ok(a.slice(info).into()) },).unwrap()
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn is_owned(&self) -> bool {
        match_arrok!(self; Dynamic(a) => { Ok(a.is_owned()) },).unwrap()
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn into_owned<'b>(self) -> ArrOk<'b> {
        match_arrok!(self; Dynamic(a) => {
            let a: ArrOk<'a> = a.into_owned().into();
            // this is safe because we only need it for &str type
            // and the lifetime of str should be longer than both
            // 'a and 'b
            // remove this transmute once we add a different lifetime
            // for &str datatype
            Ok(unsafe { std::mem::transmute::<ArrOk<'a>, ArrOk<'b>>(a) })
            // Ok(a)
        },)
        .unwrap()
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
            // ArrOk::OptUsize(a) => a.cast::<f64>().into(),
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
            _ => self.cast_i64().into(),
        }
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn as_ptr<T: GetDataType>(&self) -> *const T {
        // we have known the datatype of the enum ,so only one arm will be executed
        match_arrok!(self; Dynamic(a) => { Ok(a.as_ptr() as *const T) },).unwrap()
    }

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn as_mut_ptr<T: GetDataType>(&mut self) -> *mut T {
        // we have known the datatype of the enum ,so only one arm will be executed
        match_arrok!(self; Dynamic(a) => { Ok(a.as_mut_ptr() as *mut T) },).unwrap()
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

    // #[inline]
    // pub fn cast_str(self) -> ArbArray<'a, &'a str> {
    //     match_arrok!(self; Str(a) => { Ok(a) },).unwrap()
    // }

    // #[inline]
    // pub fn cast_object(self) -> ArbArray<'a, PyValue> {
    //     match_arrok!(self, a, { a }, Object)
    // }

    #[inline]
    pub fn cast_vecusize(self) -> ArbArray<'a, Vec<usize>> {
        match_arrok!(self; VecUsize(a) => { Ok(a) },).unwrap()
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

    #[allow(unreachable_patterns)]
    #[inline]
    pub fn view(&self) -> ArrOk<'_> {
        match_arrok!(self; Dynamic(a) => { Ok(a.view().into()) },).unwrap()
    }

    #[cfg(feature = "time")]
    pub fn cast_datetime(self, unit: Option<TimeUnit>) -> Self {
        if let Some(unit) = unit {
            match_arrok!(
                &self;
                DateTimeMs(a) => {
                    if unit == TimeUnit::Millisecond {
                        return self;
                    } else {
                        Ok(a.view().to_datetime(Some(unit)))
                    }
                },
                DateTimeUs(a) => {
                    if unit == TimeUnit::Microsecond {
                        return self;
                    } else {
                        Ok(a.view().to_datetime(Some(unit)))
                    }
                },
                DateTimeNs(a) => {
                    if unit == TimeUnit::Nanosecond {
                        return self;
                    } else {
                        Ok(a.view().to_datetime(Some(unit)))
                    }
                },
                Numeric(a) => { Ok(a.view().to_datetime(Some(unit))) },
            )
            .unwrap()
        } else {
            let dtype = self.dtype();
            if dtype.is_time() {
                self
            } else {
                match_arrok!(self; Numeric(a) => {Ok(a.cast_datetime_ns().into())},).unwrap()
            }
        }
    }
}

#[cfg(feature = "concat")]
macro_rules! impl_same_dtype_concat_1d {
    ($($(#[$meta: meta])? $arm: ident),* $(,)*) => {
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
                                Arr1::from_vec(out).into_dyn().into()
                            } else {
                                let out = arr_vec.into_iter().map(|a| {
                                    let a = if let $arm(a) = a {a.into_owned()} else {unreachable!()};
                                    a.to_dim1().unwrap().0.into_raw_vec().into_iter()
                                }).flatten().collect();
                                Arr1::from_vec(out).into_dyn().into()
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
    // Str,
    Object,
    OptBool,
    OptI32,
    OptI64,
    OptF32,
    OptF64,
    OptUsize,
    VecUsize,
    #[cfg(feature = "time")]
    DateTimeMs,
    #[cfg(feature = "time")]
    DateTimeUs,
    #[cfg(feature = "time")]
    DateTimeNs,
    #[cfg(feature = "time")]
    TimeDelta,
);

// impl<'a> Cast<ArbArray<'a, &'a str>> for ArrOk<'a> {
//     #[inline]
//     fn cast(self) -> ArbArray<'a, &'a str> {
//         match self {
//             ArrOk::Str(e) => e,
//             _ => unimplemented!("Cast to str is unimplemented"),
//         }
//     }
// }

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
                use crate::prelude::ViewOnBase;
                use arrow::array::PrimitiveArray;
                match arr.data_type() {
                    $(ArrowDT::$arrow_dt => {
                        let a = arr.as_any().downcast_ref::<PrimitiveArray<$real>>().unwrap();
                        let data: &[$real] = a.values();
                        let nulls = a.validity();
                        if nulls.is_some() {
                            let data = a.into_iter().map(|v| v.cloned().unwrap_or_else(|| <$real>::none())).collect_trusted();
                            Arr1::from_vec(data).into_dyn().into()
                        } else {
                            let view: ArrViewD<'a, $real> = unsafe {
                                std::mem::transmute(ArrView1::from_slice(data.len(), data).into_dyn())
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
                            Arr1::from_vec(data).into_dyn().into()
                        } else {
                            let view: ArrViewD<'a, i32> = unsafe {
                                std::mem::transmute(ArrView1::from_slice(data.len(), data).into_dyn())
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
                            Arr1::from_vec(data).into_dyn().into()
                        } else {
                            let view: ArrViewD<'a, i64> = unsafe {
                                std::mem::transmute(ArrView1::from_slice(data.len(), data).into_dyn())
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
                                std::mem::transmute(ArrView1::from_slice(len, data).into_dyn())
                            };
                            let out = ViewOnBase::new_from_arrow(arr, view);
                            out.into()
                        }
                    },
                    ArrowDT::Utf8 => {
                        // todo: remove unneeded clone here
                        let a = arr.as_any().downcast_ref::<arrow::array::Utf8Array<i32>>().unwrap();
                        let data = a.into_iter().map(|s| s.map(|s| s.to_string()).unwrap_or_default()).collect_trusted();
                        Arr1::from_vec(data).into_dyn().into()
                    },
                    ArrowDT::LargeUtf8 => {
                        let a = arr.as_any().downcast_ref::<arrow::array::Utf8Array<i64>>().unwrap();
                        let data = a.into_iter().map(|s| s.map(|s| s.to_string()).unwrap_or_default()).collect_trusted();
                        Arr1::from_vec(data).into_dyn().into()
                    }
                    #[cfg(feature="time")]
                    ArrowDT::Timestamp(arw_unit, arw_tz) => {
                        use arrow::datatypes::TimeUnit;
                        assert!(arw_tz.is_none(), "Timezone is not supported yet");
                        let a = arr.as_any().downcast_ref::<PrimitiveArray<i64>>().unwrap();
                        match arw_unit {
                            TimeUnit::Second => {
                                unimplemented!("Timeunit second is not supported yet")
                                // let data = a.into_iter().map(|v| {
                                //     DateTime::<unit::Second>::from_opt_i64(v.cloned())
                                // }).collect_trusted();
                                // Arr1::from_vec(data).into_dyn().into()
                            },
                            TimeUnit::Millisecond => {
                                let data = a.into_iter().map(|v| {
                                    DateTime::<unit::Millisecond>::from_opt_i64(v.cloned())
                                }).collect_trusted();
                                Arr1::from_vec(data).into_dyn().into()
                            },
                            TimeUnit::Microsecond => {
                                let data = a.into_iter().map(|v| {
                                    DateTime::<unit::Microsecond>::from_opt_i64(v.cloned())
                                }).collect_trusted();
                                Arr1::from_vec(data).into_dyn().into()
                            },
                            TimeUnit::Nanosecond => {
                                let data = a.into_iter().map(|v| {
                                    DateTime::<unit::Nanosecond>::from_opt_i64(v.cloned())
                                }).collect_trusted();
                                Arr1::from_vec(data).into_dyn().into()
                            },
                        }
                        // Arr1::from_vec(data).into_dyn().into()
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
    ($($(#[$meta: meta])? $T: ty: $cast_func: ident),* $(,)? ) => {
        $(
            $(#[$meta])?
            impl<'a> Cast<ArbArray<'a, $T>> for ArrOk<'a>
            {
                #[inline]
                fn cast(self) -> ArbArray<'a, $T> {
                    match_arrok!(self; Cast(a) => { Ok(a.cast::<$T>()) },
                        // U8, F32, F64, I32, I64, U64, Usize, OptUsize, Bool, String, Str, Object,
                        // #[cfg(feature="time")] DateTime,
                        // #[cfg(feature="time")] TimeDelta
                    ).unwrap()
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
    Option<bool>: cast_opt_bool,
    Option<f32>: cast_opt_f32,
    Option<f64>: cast_opt_f64,
    Option<i32>: cast_opt_i32,
    Option<i64>: cast_opt_i64,
    #[cfg(feature="time")]
    DateTime<unit::Millisecond>: cast_datetime_ms,
    #[cfg(feature="time")]
    DateTime<unit::Microsecond>: cast_datetime_us,
    #[cfg(feature="time")]
    DateTime<unit::Nanosecond>: cast_datetime_ns,
    Option<usize>: cast_optusize,
    Object: cast_object,
    #[cfg(feature="time")]
    TimeDelta: cast_timedelta
);
