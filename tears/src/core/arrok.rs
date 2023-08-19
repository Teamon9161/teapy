use std::fmt::Debug;

use super::arbarray::{match_arbarray, ArbArray};
use super::view::ArrViewD;
// use crate::core::arbarray;
use crate::{match_datatype_arm, DataType, DateTime, GetDataType, OptUsize, PyValue, TimeDelta};
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
