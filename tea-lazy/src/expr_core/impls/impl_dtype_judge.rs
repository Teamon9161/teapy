// use super::export::*;
use crate::Expr;
use core::prelude::*;

macro_rules! impl_dtype_judge {
    ($($(#[$meta: meta])? $func: ident -> $arm: ident),* $(,)?) => {
        impl<'a> Expr<'a> {
            $(
                $(#[$meta])?
                pub fn $func(&self) -> Option<bool> {
                    let a = self.view_arr(None).ok()?;
                    Some(matches!(a, ArrOk::$arm(_)))
                }
            )*
        }
    };
}

impl_dtype_judge!(
    is_f32 -> F32,
    is_f64 -> F64,
    is_i32 -> I32,
    is_i64 -> I64,
    is_u64 -> U64,
    is_usize -> Usize,
    is_string -> String,
    is_str -> Str,
    is_bool -> Bool,
    #[cfg(feature="time")]
    is_datetime -> DateTime,
    #[cfg(feature="time")]
    is_timedelta -> TimeDelta,
    is_optusize -> OptUsize,
    is_object -> Object,
    is_vecusize -> VecUsize,
    #[cfg(feature = "option_dtype")] is_optf32 -> OptF32,
    #[cfg(feature = "option_dtype")] is_optf64 -> OptF64,
    #[cfg(feature = "option_dtype")] is_opti32 -> OptI32,
    #[cfg(feature = "option_dtype")] is_opti64 -> OptI64,
);

impl<'a> Expr<'a> {
    #[inline]
    pub fn is_float(&self) -> Option<bool> {
        let a = self.view_arr(None).ok()?;
        Some(a.dtype().is_float())
    }

    #[inline]
    pub fn is_int(&self) -> Option<bool> {
        let a = self.view_arr(None).ok()?;
        Some(a.dtype().is_int())
    }
}
