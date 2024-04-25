mod cast;
mod option_datatype;
mod pyvalue;

pub use cast::Cast;
pub use option_datatype::OptUsize;
pub use pyvalue::PyValue;

use num::{traits::MulAdd, Num};
use std::cmp::{Ordering, PartialOrd};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use tea_utils::kh_sum;

#[cfg(feature = "time")]
pub use tea_time::{DateTime, TimeDelta, TimeUnit};

#[cfg(feature = "option_dtype")]
pub use option_datatype::{OptBool, OptF32, OptF64, OptI32, OptI64};

#[derive(PartialEq, Eq, Debug)]
pub enum DataType {
    Bool,
    F32,
    F64,
    I32,
    I64,
    U8,
    U64,
    Usize,
    Str,
    String,
    Object,
    OptUsize,
    VecUsize,
    #[cfg(feature = "time")]
    DateTime,
    #[cfg(feature = "time")]
    TimeDelta,
    #[cfg(feature = "option_dtype")]
    OptF64,
    #[cfg(feature = "option_dtype")]
    OptF32,
    #[cfg(feature = "option_dtype")]
    OptI32,
    #[cfg(feature = "option_dtype")]
    OptI64,
    #[cfg(feature = "option_dtype")]
    OptBool,
}

#[macro_export]
macro_rules! match_datatype_arm {

    (all $expr: expr, $v: ident, $other_enum: ident, $ty: ty, $body: tt) => {
        {
            // #[cfg(not(feature="option_dtype"))]
            macro_rules! inner_macro {
                () => {
                    match_datatype_arm!($expr, $v, $other_enum, $ty, (
                        Bool, F32, F64, I32, I64, U8, U64, Usize, Object, String, Str,
                        OptUsize, VecUsize,
                        #[cfg(feature="time")] DateTime,
                        #[cfg(feature="time")] TimeDelta,
                        #[cfg(feature="option_dtype")] OptF64,
                        #[cfg(feature="option_dtype")] OptF32,
                        #[cfg(feature="option_dtype")] OptI32,
                        #[cfg(feature="option_dtype")] OptI64,
                        #[cfg(feature="option_dtype")] OptBool
                    ), $body)
                };
            }
            inner_macro!()
        }
    };
    ($expr: expr, $v: ident, $other_enum: ident, $ty: ty, ($($(#[$meta: meta])? $arm: ident $(($arg: ident))?),*), $body: tt) => {
        match <$ty>::dtype() {
             $($(#[$meta])? DataType::$arm $(($arg))? => {if let $other_enum::$arm($v) = $expr $body else {panic!("datatype mismatch {:?} {:?}", <$ty>::dtype(), $expr)}},)*
            _ => unreachable!()
        }
    };
}
// pub(crate) use match_datatype_arm;

pub trait GetDataType: Send + Sync {
    // type Physical;
    fn dtype() -> DataType
    where
        Self: Sized;
}

macro_rules! impl_datatype {
    ($tyname:ident, $physical:ty) => {
        impl GetDataType for $physical {
            // type Physical = $physical;
            #[inline(always)]
            fn dtype() -> DataType {
                DataType::$tyname
            }
        }
    };
}

impl DataType {
    #[cfg(not(feature = "option_dtype"))]
    #[inline(always)]
    pub fn is_float(&self) -> bool {
        matches!(self, DataType::F32 | DataType::F64)
    }

    #[cfg(feature = "option_dtype")]
    #[inline(always)]
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DataType::F32 | DataType::F64 | DataType::OptF32 | DataType::OptF64
        )
    }

    #[cfg(not(feature = "option_dtype"))]
    #[inline(always)]
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            DataType::I32 | DataType::I64 | DataType::Usize | DataType::U64 | DataType::OptUsize
        )
    }

    #[cfg(feature = "option_dtype")]
    #[inline(always)]
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            DataType::I32
                | DataType::I64
                | DataType::Usize
                | DataType::U64
                | DataType::OptUsize
                | DataType::OptI32
                | DataType::OptI64
        )
    }

    pub fn float(self) -> Self {
        use DataType::*;
        match self {
            F32 => F32,
            I32 => F32,
            I64 => F64,
            Usize => F64,
            U64 => F64,
            OptUsize => F64,
            #[cfg(feature = "option_dtype")]
            OptI32 => OptF32,
            #[cfg(feature = "option_dtype")]
            OptI64 => OptF64,
            _ => F64,
        }
    }

    pub fn int(self) -> Self {
        use DataType::*;
        match self {
            I32 => I32,
            F32 => I32,
            F64 => I64,
            Usize => Usize,
            U64 => I64,
            OptUsize => OptUsize,
            #[cfg(feature = "option_dtype")]
            OptF32 => OptI32,
            #[cfg(feature = "option_dtype")]
            OptF64 => OptI64,
            _ => I64,
        }
    }
}

impl_datatype!(Bool, bool);
impl_datatype!(U8, u8);
impl_datatype!(F32, f32);
impl_datatype!(F64, f64);
impl_datatype!(I32, i32);
impl_datatype!(I64, i64);
impl_datatype!(U64, u64);
impl_datatype!(Usize, usize);
impl_datatype!(String, String);
impl_datatype!(OptUsize, OptUsize);
impl_datatype!(VecUsize, Vec<usize>);

#[cfg(feature = "time")]
impl_datatype!(DateTime, DateTime);
#[cfg(feature = "time")]
impl_datatype!(TimeDelta, TimeDelta);

#[cfg(feature = "option_dtype")]
impl_datatype!(OptF64, OptF64);
#[cfg(feature = "option_dtype")]
impl_datatype!(OptF32, OptF32);
#[cfg(feature = "option_dtype")]
impl_datatype!(OptI32, OptI32);
#[cfg(feature = "option_dtype")]
impl_datatype!(OptI64, OptI64);
#[cfg(feature = "option_dtype")]
impl_datatype!(OptBool, OptBool);

pub trait GetNone {
    fn none() -> Self;
    #[allow(clippy::wrong_self_convention)]
    fn is_none(&self) -> bool;
}

impl GetNone for f64 {
    #[inline(always)]
    fn none() -> Self {
        f64::NAN
    }
    #[inline(always)]
    fn is_none(&self) -> bool {
        self.is_nan()
    }
}

impl GetNone for f32 {
    #[inline(always)]
    fn none() -> Self {
        f32::NAN
    }
    #[inline(always)]
    fn is_none(&self) -> bool {
        self.is_nan()
    }
}

impl GetNone for String {
    #[inline(always)]
    fn none() -> Self {
        "None".to_owned()
    }
    #[inline(always)]
    fn is_none(&self) -> bool {
        self == "None"
    }
}

impl GetNone for &str {
    #[inline(always)]
    fn none() -> Self {
        "None"
    }
    #[inline(always)]
    fn is_none(&self) -> bool {
        *self == "None"
    }
}

macro_rules! impl_getnone {
    (int $($T: ty),*) => {
        $(
            impl GetNone for $T {
                #[inline(always)]
                fn none() -> Self {
                    unreachable!("int dtype can not be None")
                }
                #[inline(always)]
                fn is_none(&self) -> bool {
                    false
                }
            }
        )*
    };
}
impl_getnone!(int char, i8, i16, i32, i64, u8, u16, u32, u64, usize, isize);

impl GetNone for bool {
    #[inline(always)]
    fn none() -> Self {
        panic!("bool doesn't have None value")
    }
    #[inline(always)]
    fn is_none(&self) -> bool {
        false
    }
}

impl GetNone for Vec<usize> {
    #[inline(always)]
    fn none() -> Self {
        vec![]
    }
    #[inline(always)]
    fn is_none(&self) -> bool {
        self.is_empty()
    }
}

#[cfg(feature = "time")]
impl GetNone for DateTime {
    #[inline(always)]
    fn none() -> Self {
        Self(None)
    }
    #[inline(always)]
    fn is_none(&self) -> bool {
        self.0.is_none()
    }
}

#[cfg(feature = "time")]
impl GetNone for TimeDelta {
    #[inline(always)]
    fn none() -> Self {
        TimeDelta::nat()
    }
    #[inline(always)]
    fn is_none(&self) -> bool {
        self.is_nat()
    }
}

impl<'a> GetDataType for &'a str {
    // type Physical = &'a str;
    #[inline(always)]
    fn dtype() -> DataType {
        DataType::Str
    }
}

pub trait Number:
    Copy
    + Clone
    + GetNone
    + Sized
    + Default
    + Num
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + PartialOrd
    + MulAdd
    + GetDataType
    + Cast<f64>
    + Cast<f32>
    + Cast<usize>
    + Cast<i32>
    + Cast<i64>
    + 'static
{
    // type Dtype;
    /// return the min value of the data type
    fn min_() -> Self;

    /// return the max value of the data type
    fn max_() -> Self;

    #[inline(always)]
    fn kh_sum(&mut self, v: Self, c: &mut Self) {
        *self = kh_sum(*self, v, c);
    }

    /// return min(self, other)
    /// note that self should not be nan
    #[inline(always)]
    fn min_with(self, other: Self) -> Self {
        if other < self {
            // if other is nan, this condition should be false
            other
        } else {
            self
        }
    }

    /// return max(self, other)
    /// note that self should not be nan
    #[inline(always)]
    fn max_with(self, other: Self) -> Self {
        if other > self {
            // if other is nan, this condition should be false
            other
        } else {
            self
        }
    }

    #[inline(always)]
    fn f32(self) -> f32 {
        Cast::<f32>::cast(self)
    }

    #[inline(always)]
    fn f64(self) -> f64 {
        Cast::<f64>::cast(self)
    }

    #[inline(always)]
    fn i32(self) -> i32 {
        Cast::<i32>::cast(self)
    }

    #[inline(always)]
    fn i64(self) -> i64 {
        Cast::<i64>::cast(self)
    }

    #[inline(always)]
    fn usize(self) -> usize {
        Cast::<usize>::cast(self)
    }

    /// create a value of type T using a value of type U using `Cast`
    #[inline(always)]
    fn fromas<U>(v: U) -> Self
    where
        U: Number + Cast<Self>,
        Self: 'static,
    {
        v.to::<Self>()
    }

    /// cast self to another dtype using `Cast`
    #[inline(always)]
    fn to<T: Number>(self) -> T
    where
        Self: Cast<T>,
    {
        Cast::<T>::cast(self)
    }

    /// check whether self is nan
    #[inline(always)]
    #[allow(clippy::eq_op)]
    fn isnan(self) -> bool {
        self != self
    }
    /// check whether self is not nan
    #[inline(always)]
    #[allow(clippy::eq_op)]
    fn notnan(self) -> bool {
        self == self
    }

    /// return nan value
    fn nan() -> Self;

    /// if other is nan, then add other to self and n += 1
    /// else just return self
    #[inline]
    fn n_add(self, other: Self, n: &mut usize) -> Self {
        // note: only check if other is NaN
        // assume that self is not NaN
        if other.notnan() {
            *n += 1;
            self + other
        } else {
            self
        }
    }

    /// if other is nan, then product other to self and n += 1
    /// else just return self
    #[inline]
    fn n_prod(self, other: Self, n: &mut usize) -> Self {
        // note: only check if other is NaN
        // assume that self is not NaN
        if other.notnan() {
            *n += 1;
            self * other
        } else {
            self
        }
    }

    /// if other is nan, then add other to self
    /// else just return self
    #[inline(always)]
    fn nanadd(self, other: Self) -> Self {
        // note: only check if other is NaN
        // assume that self is not NaN
        if other.notnan() {
            self + other
        } else {
            self
        }
    }

    /// let NaN value be largest, only for sorting(from smallest to largest)
    #[inline(always)]
    fn nan_sort_cmp(&self, other: &Self) -> Ordering {
        if other.isnan() | (self < other) {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }

    /// let NaN value be smallest, only for sorting(from largest to smallest)
    #[inline]
    fn nan_sort_cmp_rev(&self, other: &Self) -> Ordering {
        if other.isnan() | (self > other) {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }

    /// let NaN value be largest, only for sorting(from smallest to largest)
    #[inline]
    fn nan_sort_cmp_stable(&self, other: &Self) -> Ordering {
        if other.isnan() {
            if self.isnan() {
                Ordering::Equal
            } else {
                Ordering::Less
            }
        } else if self.isnan() | (self > other) {
            Ordering::Greater
        } else if self == other {
            Ordering::Equal
        } else {
            Ordering::Less
        }
    }

    #[inline]
    fn nan_sort_cmp_rev_stable(&self, other: &Self) -> Ordering {
        if other.isnan() {
            if self.isnan() {
                Ordering::Equal
            } else {
                Ordering::Less
            }
        } else if self.isnan() | (self < other) {
            Ordering::Greater
        } else if self == other {
            Ordering::Equal
        } else {
            Ordering::Less
        }
    }
}

macro_rules! impl_number {
    // base impl for number
    (@ base_impl $dtype:ty, $datatype:ident) => {
        // type Dtype = $datatype;

        #[inline(always)]
        fn min_() -> $dtype {
            <$dtype>::MIN
        }

        #[inline(always)]
        fn max_() -> $dtype {
            <$dtype>::MAX
        }

    };
    // special impl for float
    (float $($dtype:ty, $datatype:ident); *) => {
        $(impl Number for $dtype {
            impl_number!(@ base_impl $dtype, $datatype);

            #[inline(always)]
            fn nan() -> Self {
                <$dtype>::NAN
            }

            // #[inline(always)]
            // fn kh_sum(&mut self, v: Self, c: &mut Self) {
            //     *self = kh_sum(*self, v, c);
            // }
        })*
    };
    // special impl for other type
    (other $($dtype:ty, $datatype:ident); *) => {
        $(impl Number for $dtype {
            impl_number!(@ base_impl $dtype, $datatype);

            #[inline]
            fn nan() -> Self {
                panic!("This type of number doesn't have NaN value")
            }

            // these types doen't have NaN
            #[inline(always)]
            fn isnan(self) -> bool {
                false
            }

            #[inline(always)]
            fn notnan(self) -> bool {
                true
            }
            #[inline(always)]
            fn kh_sum(&mut self, v: Self, _c: &mut Self) {
                *self += v;
            }
        })*
    };
}

impl_number!(
    float
    f32, F32;
    f64, F64
);

impl_number!(
    other
    i32, I32;
    i64, I64;
    u64, U64;
    usize, Usize
);

pub trait BoolType {
    fn bool_(self) -> bool;
}

impl BoolType for bool {
    #[inline(always)]
    fn bool_(self) -> bool {
        self
    }
}

#[cfg(feature = "option_dtype")]
impl BoolType for OptBool {
    #[inline(always)]
    fn bool_(self) -> bool {
        self.unwrap()
    }
}
