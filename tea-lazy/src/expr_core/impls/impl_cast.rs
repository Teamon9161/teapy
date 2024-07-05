use crate::Expr;
use tea_core::prelude::*;

macro_rules! impl_cast {
    ($($(#[$meta: meta])? $func: ident: $dtype: ident $(($inner: path))?),* $(,)?) => {
        impl<'a> Expr<'a> {
            $(
                $(#[$meta])?
                pub fn $func(&mut self) -> &mut Self {
                    self.chain_f_ctx(|(arr, ctx)| {
                        let dtype = arr.view_arr(ctx.as_ref())?.dtype();
                        if dtype == DataType::$dtype $(($inner))? {
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
    cast_object: Object,
    #[cfg(feature="time")] cast_datetime_ms: DateTime(TimeUnit::Millisecond),
    #[cfg(feature="time")] cast_datetime_us: DateTime(TimeUnit::Microsecond),
    #[cfg(feature="time")] cast_datetime_ns: DateTime(TimeUnit::Nanosecond),
    #[cfg(feature="time")] cast_timedelta: TimeDelta,
    cast_optusize: OptUsize,
    cast_vecusize: VecUsize,
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
        self.chain_f_ctx(move |(arr, ctx)| {
            let arr = arr.into_arr(ctx.clone())?;
            Ok((arr.cast_datetime(unit).into(), ctx))
        });
        self
    }
}
