use num::{Num, NumCast, traits::NumOps, FromPrimitive, ToPrimitive, cast::AsPrimitive, One, Zero};
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign};
use std::cmp::PartialOrd;
use numpy::ndarray::{ArrayView1, ArrayViewMut1};
use numpy::Element;

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

pub trait Number : Copy + Clone + FromPrimitive + NumOps + Num + NumCast + 
AddAssign + SubAssign + MulAssign + DivAssign + PartialOrd + GetDataType + Element + 
ToPrimitive + Zero + One + AsPrimitive<f64> + AsPrimitive<usize> + AsPrimitive<i32> {
    type Dtype;
    fn min_() -> Self;
    fn max_() -> Self;
    fn f64(self) -> f64;
    fn to<T: Number>(self) -> T where Self: AsPrimitive<T>;
    fn isnan(self) -> bool;
    fn notnan(self) -> bool;
}

macro_rules! impl_number {
    ($dtype:ty, $datatype:ident) => {
        impl Number for $dtype {
            type Dtype = $datatype;
            #[inline(always)]
            fn min_() -> $dtype {<$dtype>::MIN}
            #[inline(always)]
            fn max_() -> $dtype {<$dtype>::MAX}
            #[inline(always)]
            fn f64(self) -> f64 {AsPrimitive::<f64>::as_(self)}
            #[inline(always)]
            fn to<T: Number>(self) -> T 
            where Self: AsPrimitive<T> {
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

// function datatype
pub type ArrayFunc<T, U> = fn(ArrayView1<T>, ArrayViewMut1<U>);
pub type TsFunc<T> = fn(ArrayView1<T>, ArrayViewMut1<f64>, usize, usize, usize);
pub type TsFunc2<T> = fn(ArrayView1<T>, ArrayView1<T>, ArrayViewMut1<f64>, usize, usize, usize);
