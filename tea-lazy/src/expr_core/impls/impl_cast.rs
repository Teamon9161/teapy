use crate::Expr;
use core::prelude::*;
use core::{match_all, match_arrok};
use pyo3::Python;

macro_rules! impl_cast {
    ($($(#[$meta: meta])? $func: ident: $dtype: ident),* $(,)?) => {
        impl<'a> Expr<'a> {
            $(
                $(#[$meta])?
                pub fn $func(&mut self) -> &mut Self {
                    self.chain_f_ctx(|(arr, ctx)| {
                        let dtype = arr.view_arr(ctx.as_ref())?.dtype();
                        if dtype == DataType::$dtype {
                            return Ok((arr, ctx));
                        } else {
                            Ok((arr.into_arr(ctx.clone())?.$func().into(), ctx))
                        }
                    });
                    self
                }
            )*
        }
    };
}

impl_cast!(
    cast_f32: F32,
    cast_f64: F64,
    cast_i32: I32,
    cast_i64: I64,
    cast_u64: U64,
    cast_usize: Usize,
    cast_string: String,
    cast_bool: Bool,
    #[cfg(feature="time")] cast_datetime_default: DateTime,
    #[cfg(feature="time")] cast_timedelta: TimeDelta,
    cast_optusize: OptUsize,
    cast_vecusize: VecUsize,
    #[cfg(feature = "option_dtype")] cast_optf32: OptF32,
    #[cfg(feature = "option_dtype")] cast_optf64: OptF64,
    #[cfg(feature = "option_dtype")] cast_opti32: OptI32,
    #[cfg(feature = "option_dtype")] cast_opti64: OptI64,
    #[cfg(feature = "option_dtype")] cast_optbool: OptBool,


);

impl<'a> Expr<'a> {
    pub fn cast_float(&mut self) -> &mut Self {
        self.chain_f_ctx(|(arr, ctx)| {
            let dtype = arr.view_arr(ctx.as_ref())?.dtype();
            if dtype.is_float() {
                Ok((arr, ctx))
            } else {
                Ok((arr.into_arr(ctx.clone())?.cast_float().into(), ctx))
            }
        });
        self
    }

    pub fn cast_int(&mut self) -> &mut Self {
        self.chain_f_ctx(|(arr, ctx)| {
            let dtype = arr.view_arr(ctx.as_ref())?.dtype();
            if dtype.is_int() {
                Ok((arr, ctx))
            } else {
                Ok((arr.into_arr(ctx.clone())?.cast_int().into(), ctx))
            }
        });
        self
    }

    #[cfg(feature = "time")]
    pub fn cast_datetime(&mut self, unit: Option<TimeUnit>) -> &mut Self {
        if let Some(unit) = unit {
            self.chain_f_ctx(move |(arr, ctx)| {
                let arr = arr.view_arr(ctx.as_ref())?;
                let out = match_arrok!(
                    arr,
                    a,
                    { a.view().to_datetime(unit)? },
                    F32,
                    F64,
                    I32,
                    I64,
                    Usize,
                    DateTime
                );
                Ok((out.into(), ctx))
            });
            self
        } else {
            self.cast_datetime_default()
        }
    }

    #[allow(unreachable_patterns)]
    pub fn cast_object_eager(&mut self, py: Python) -> TpResult<&mut Self> {
        self.eval_inplace(None)?;
        let arr = self.view_arr(None)?;
        let out = match_arrok!(arr, a, { a.view().to_object(py) });
        self.lock().set_base(out.into());
        Ok(self)
    }
}
