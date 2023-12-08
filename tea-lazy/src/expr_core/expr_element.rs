#[cfg(feature = "time")]
use core::datatype::{DateTime, TimeDelta};
use core::datatype::{GetDataType, OptUsize, PyValue};
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
#[cfg(feature = "time")]
impl ExprElement for DateTime {}
#[cfg(feature = "time")]
impl ExprElement for TimeDelta {}

impl ExprElement for OptUsize {}
impl ExprElement for Vec<usize> {}

#[cfg(feature = "option_dtype")]
impl ExprElement for OptF64 {}
#[cfg(feature = "option_dtype")]
impl ExprElement for OptF32 {}
#[cfg(feature = "option_dtype")]
impl ExprElement for OptI32 {}
#[cfg(feature = "option_dtype")]
impl ExprElement for OptI64 {}
#[cfg(feature = "option_dtype")]
impl ExprElement for OptBool {}

impl ExprElement for PyValue {}
