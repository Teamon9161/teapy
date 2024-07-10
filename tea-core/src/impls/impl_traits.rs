use crate::prelude::*;
use ndarray::{arr0, ArrayBase, Data, DataOwned, Dimension, RawData};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;

impl<T, S, D> Default for ArrBase<S, D>
where
    S: DataOwned<Elem = T>,
    D: Dimension,
    T: Default,
{
    // NOTE: We can implement Default for non-zero dimensional array views by
    // using an empty slice, however we need a trait for nonzero Dimension.
    #[inline(always)]
    fn default() -> Self {
        ArrayBase::default(D::default()).wrap()
    }
}

impl<S, T, D> fmt::Debug for ArrBase<S, D>
where
    S: Data<Elem = T>,
    D: Dimension,
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let string = format!("{:?}", self.0);
        let data = string.split(", shape=").next().unwrap_or("");
        f.write_str(data)
        // self.0.fmt(f)
    }
}

impl<S: RawData, D> Deref for ArrBase<S, D> {
    type Target = ArrayBase<S, D>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: RawData, D> DerefMut for ArrBase<S, D> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> From<T> for ArrD<T> {
    #[inline(always)]
    fn from(v: T) -> Self {
        arr0(v).wrap().into_dyn()
    }
}

trait SingleElement {}
impl SingleElement for bool {}
impl SingleElement for f32 {}
impl SingleElement for f64 {}
impl SingleElement for i32 {}
impl SingleElement for i64 {}
impl SingleElement for u8 {}
impl SingleElement for u64 {}
impl SingleElement for usize {}
impl SingleElement for String {}
#[cfg(feature = "time")]
impl<U: TimeUnitTrait> SingleElement for DateTime<U> {}
#[cfg(feature = "time")]
impl SingleElement for TimeDelta {}

impl<T: SingleElement> SingleElement for Option<T> {}

impl<T: SingleElement> From<T> for ArbArray<'_, T> {
    #[inline(always)]
    fn from(v: T) -> Self {
        let arr = arr0(v).wrap().into_dyn();
        arr.into()
    }
}

impl Default for ArrOk<'_> {
    #[inline(always)]
    fn default() -> Self {
        let out: ArrD<i32> = Default::default();
        out.into()
    }
}

impl<'a, T> From<ArrViewD<'a, T>> for ArrOk<'a>
where
    Self: From<ArbArray<'a, T>>,
{
    #[inline]
    fn from(ty: ArrViewD<'a, T>) -> Self {
        ArbArray::<'a, T>::from(ty).into()
    }
}

impl<'a, T> From<ArrViewMutD<'a, T>> for ArrOk<'a>
where
    Self: From<ArbArray<'a, T>>,
{
    #[inline]
    fn from(ty: ArrViewMutD<'a, T>) -> Self {
        ArbArray::<'a, T>::from(ty).into()
    }
}

impl<'a, T: 'a> From<ArrD<T>> for ArrOk<'a>
where
    Self: From<ArbArray<'a, T>>,
{
    #[inline]
    fn from(ty: ArrD<T>) -> Self {
        ArbArray::<'a, T>::from(ty).into()
    }
}

impl<'a, T> From<Pin<Box<ViewOnBase<'a, T>>>> for ArrOk<'a>
where
    Self: From<ArbArray<'a, T>>,
{
    #[inline]
    fn from(ty: Pin<Box<ViewOnBase<'a, T>>>) -> Self {
        ArbArray::<'a, T>::from(ty).into()
    }
}

macro_rules! impl_from {
    ($($(#[$meta:meta])? ($arm: ident, $dtype: ident $(($inner: path))?, $ty: ty, $func_name: ident)),* $(,)?) => {
        impl<'a> ArrOk<'a> {
            $(
                $(#[$meta])?
                pub fn $func_name(self) -> TResult<ArbArray<'a, $ty>> {
                    if let ArrOk::$arm(v) = self {
                        Ok(v)
                    } else {
                        tbail!("ArrOk is not of type {:?}", <$ty>::dtype())
                    }
                }
            )*
        }
    };
}

impl_from!(
    (Bool, Bool, bool, bool),
    (F32, F32, f32, f32),
    (F64, F64, f64, f64),
    (I32, I32, i32, i32),
    (I64, I64, i64, i64),
    (U8, U8, u8, u8),
    (U64, U64, u64, u64),
    (Usize, Usize, usize, usize),
    (String, String, String, string),
    (OptBool, OptBool, Option<bool>, opt_bool),
    (OptF32, OptF32, Option<f32>, opt_f32),
    (OptF64, OptF64, Option<f64>, opt_f64),
    (OptI32, OptI32, Option<i32>, opt_i32),
    (OptI64, OptI64, Option<i64>, opt_i64),
    (OptUsize, OptUsize, Option<usize>, opt_usize),
    (VecUsize, VecUsize, Vec<usize>, vec_usize),
    // #[cfg(feature = "py")]
    (Object, Object, Object, object),
    #[cfg(feature = "time")]
    (DateTimeMs, DateTime(TimeUnit::Millisecond), DateTime<unit::Millisecond>, datetime_ms),
    #[cfg(feature = "time")]
    (DateTimeUs, DateTime(TimeUnit::Microsecond), DateTime<unit::Microsecond>, datetime_us),
    #[cfg(feature = "time")]
    (DateTimeNs, DateTime(TimeUnit::Nanosecond), DateTime<unit::Nanosecond>, datetime_ns),
    #[cfg(feature = "time")]
    (TimeDelta, TimeDelta, TimeDelta, timedelta)
);
