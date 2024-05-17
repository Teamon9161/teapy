#[cfg(feature = "time")]
use core::datatype::{DateTime, TimeDelta};
use core::datatype::{GetDataType, Object};
#[cfg(feature = "option_dtype")]
use core::datatype::{OptBool, OptF32, OptF64, OptI32, OptI64};
use std::fmt::Debug;

pub trait ExprElement: GetDataType + Default + Sync + Send + Debug {}
impl ExprElement for u8 {}
impl ExprElement for u64 {}
impl ExprElement for f32 {}
impl ExprElement for f64 {}
impl ExprElement for i32 {}
impl ExprElement for i64 {}
impl ExprElement for bool {}
impl ExprElement for usize {} // only for index currently
impl ExprElement for String {}
impl<'a> ExprElement for &'a str {}
impl ExprElement for Option<usize> {}
impl ExprElement for Vec<usize> {}
impl ExprElement for Object {}
#[cfg(feature = "time")]
impl ExprElement for DateTime {}
#[cfg(feature = "time")]
impl ExprElement for TimeDelta {}
