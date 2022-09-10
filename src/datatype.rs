use crate::algos::kh_sum;
use num::{cast::AsPrimitive, FromPrimitive, Num, NumCast, ToPrimitive};
use numpy::Element;
use std::cmp::PartialOrd;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

pub enum DataType {
    F32T,
    F64T,
    I32T,
    I64T,
    UintT,
}

pub trait GetDataType: Send + Sync {
    fn dtype() -> DataType
    where
        Self: Sized;
}

macro_rules! impl_datatype {
    ($tyname:ident, $physical:ty) => {
        pub struct $tyname {}

        impl GetDataType for $physical {
            fn dtype() -> DataType {
                DataType::$tyname
            }
        }
    };
}

impl_datatype!(F32T, f32);
impl_datatype!(F64T, f64);
impl_datatype!(I32T, i32);
impl_datatype!(I64T, i64);
impl_datatype!(UintT, usize);

pub trait Number:
    Copy
    + Clone
    + Sized
    + FromPrimitive
    + Num
    + NumCast
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + PartialOrd
    + GetDataType
    + Element
    + ToPrimitive
    + AsPrimitive<f64>
    + AsPrimitive<f32>
    + AsPrimitive<usize>
    + AsPrimitive<i32>
    + AsPrimitive<i64>
{
    type Dtype;
    fn min_() -> Self;
    fn max_() -> Self;
    fn f32(self) -> f32;
    fn f64(self) -> f64;
    fn i32(self) -> i32;
    fn fromas<U: Number>(v: U) -> Self;
    fn to<T: Number>(self) -> T
    where
        Self: AsPrimitive<T>;
    fn isnan(self) -> bool;
    fn notnan(self) -> bool;
}

macro_rules! impl_number {
    ($dtype:ty, $datatype:ident) => {
        impl Number for $dtype {
            type Dtype = $datatype;
            #[inline(always)]
            fn min_() -> $dtype {
                <$dtype>::MIN
            }
            #[inline(always)]
            fn max_() -> $dtype {
                <$dtype>::MAX
            }
            #[inline(always)]
            fn f64(self) -> f64 {
                AsPrimitive::<f64>::as_(self)
            }
            #[inline(always)]
            fn f32(self) -> f32 {
                AsPrimitive::<f32>::as_(self)
            }
            #[inline(always)]
            fn i32(self) -> i32 {
                AsPrimitive::<i32>::as_(self)
            }
            #[inline(always)]
            fn fromas<U: Number>(v: U) -> Self {
                v.to::<Self>()
            }
            #[inline(always)]
            fn to<T: Number>(self) -> T
            where
                Self: AsPrimitive<T>,
            {
                AsPrimitive::<T>::as_(self)
            }
            #[inline(always)]
            fn isnan(self) -> bool {
                self != self
            }
            #[inline(always)]
            fn notnan(self) -> bool {
                self == self
            }
        }
    };
}

impl_number!(f32, F32T);
impl_number!(f64, F64T);
impl_number!(i32, I32T);
impl_number!(i64, I64T);
impl_number!(usize, UintT);

pub trait KhSum {
    fn kh_sum(&mut self, v: Self, c: &mut Self);
}

impl KhSum for f64 {
    #[inline(always)]
    fn kh_sum(&mut self, v: f64, c: &mut Self) {
        *self = kh_sum(*self, v, c);
    }
}

impl KhSum for f32 {
    #[inline(always)]
    fn kh_sum(&mut self, v: f32, c: &mut Self) {
        *self = kh_sum(*self, v, c);
    }
}
