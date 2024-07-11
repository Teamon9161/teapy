use std::fmt::Debug;
use tea_core::prelude::*;

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
impl ExprElement for Option<bool> {}
impl ExprElement for Option<f32> {}
impl ExprElement for Option<f64> {}
impl ExprElement for Option<i32> {}
impl ExprElement for Option<i64> {}
// impl<'a> ExprElement for &'a str {}
impl ExprElement for Option<usize> {}
impl ExprElement for Vec<usize> {}
impl ExprElement for Object {}
#[cfg(feature = "time")]
impl<U: TimeUnitTrait> ExprElement for DateTime<U> where DateTime<U>: GetDataType {}
impl ExprElement for TimeDelta {}
