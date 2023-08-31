use std::fmt::Debug;

use super::arbarray::{match_arbarray, ArbArray};
use super::view::ArrViewD;
// use crate::core::arbarray;
use crate::{
    match_datatype_arm, Cast, DataType, DateTime, GetDataType, OptUsize, PyValue, TimeDelta,
};
use ndarray::IxDyn;

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
    // #[cfg(feature = "option_dtype")]
    // OptUsize(ArbArray<'a, OptUsize>),
}

impl<'a> Debug for ArrOk<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match_arr!(self, arbarray, { arbarray.fmt(f) })
    }
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
            ArrOk::OptUsize($arr) => $body,
            ArrOk::String($arr) => $body,
            ArrOk::Str($arr) => $body,
            ArrOk::Object($arr) => $body,
            ArrOk::DateTime($arr) => $body,
            ArrOk::TimeDelta($arr) => $body,
            ArrOk::VecUsize($arr) => $body,
            #[cfg(feature = "option_dtype")]
            ArrOk::OptF64($arr) => $body,
            #[cfg(feature = "option_dtype")]
            ArrOk::OptF32($arr) => $body,
            #[cfg(feature = "option_dtype")]
            ArrOk::OptI32($arr) => $body,
            #[cfg(feature = "option_dtype")]
            ArrOk::OptI64($arr) => $body,
            // #[cfg(feature = "option_dtype")]
            // ArrOk::OptUsize($arr) => $body,
        }
    };
}

pub(crate) use match_arr;

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
            all
            self,
            a,
            ArrOk,
            T,
            { a.as_ptr() as *const T }
        )
    }

    #[allow(unreachable_patterns)]
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
    (numeric $($tt: tt)*) => {match_all!(ArrOk, $($tt)*, F32, F64, I32, I64, Usize)};
    (hash $($tt: tt)*) => {match_all!(ArrOk, $($tt)*, F32, F64, I32, I64, Usize, String, Str, DateTime)};
    ($($tt: tt)*) => {match_all!(ArrOk, $($tt)*)};
}

macro_rules! impl_arrok_cast {
    ($($(#[$meta: meta])? $T: ty: $cast_func: ident),*) => {
        $(
            $(#[$meta])?
            impl<'a> Cast<ArbArray<'a, $T>> for ArrOk<'a>
            {
                fn cast(self) -> ArbArray<'a, $T> {
                    match self {
                        ArrOk::F32(e) => e.cast::<$T>(),
                        ArrOk::F64(e) => e.cast::<$T>(),
                        ArrOk::I32(e) => e.cast::<$T>(),
                        ArrOk::I64(e) => e.cast::<$T>(),
                        ArrOk::Bool(e) => e.cast::<i32>().cast::<$T>(),
                        ArrOk::Usize(e) => e.cast::<$T>(),
                        // ArrOk::Str(e) => e.cast::<$T>(),
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
            // impl<'a> Cast<ArrViewD<'a, $T>> for ArrOk<'a>
            // {
            //     fn cast(self) -> ArrViewD<'a, $T> {
            //         let arb: ArbArray<'a, $T> = self.cast();
            //         arb.view()
            //     }
            // }

            $(#[$meta])?
            impl<'a> ArrOk<'a>
            {
                pub fn $cast_func(self) -> ArbArray<'a, $T> {
                    self.cast()
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
