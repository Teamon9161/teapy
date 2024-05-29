mod object;
#[cfg(feature = "time")]
mod time;

pub use num::{One, Zero};
pub use object::Object;
pub use tevec::prelude::{BoolType, Cast, IsNone, Number as TvNumber};

#[cfg(feature = "time")]
pub use tevec::prelude::{DateTime, TimeDelta, TimeUnit};
#[cfg(feature = "time")]
pub use time::{DateTimeToPy, DateTimeToRs};

/// just for old code compatibility
pub trait Number: TvNumber {
    #[inline(always)]
    fn nan() -> Self {
        Self::none()
    }

    #[inline(always)]
    fn notnan(self) -> bool {
        self.not_none()
    }

    #[inline(always)]
    fn isnan(self) -> bool {
        self.is_none()
    }
}

impl<T: TvNumber> Number for T {}

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
                        #[cfg(feature="time")] TimeDelta
                    ), $body)
                };
            }
            inner_macro!()
        }
    };
    ($expr: expr, $v: ident, $other_enum: ident, $ty: ty, ($($(#[$meta: meta])? $arm: ident $(($arg: ident))?),* $(,)?), $body: tt) => {
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
    #[inline(always)]
    pub fn is_float(&self) -> bool {
        matches!(self, DataType::F32 | DataType::F64)
    }

    #[inline(always)]
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            DataType::I32 | DataType::I64 | DataType::Usize | DataType::U64 | DataType::OptUsize
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
// impl_datatype!(OptUsize, OptUsize);
impl_datatype!(OptUsize, Option<usize>);
impl_datatype!(VecUsize, Vec<usize>);

#[cfg(feature = "time")]
impl_datatype!(DateTime, DateTime);
#[cfg(feature = "time")]
impl_datatype!(TimeDelta, TimeDelta);

impl<'a> GetDataType for &'a str {
    // type Physical = &'a str;
    #[inline(always)]
    fn dtype() -> DataType {
        DataType::Str
    }
}

// pub trait Number:
//     Copy
//     + Clone
//     + IsNone
//     + Sized
//     + Default
//     + Num
//     + AddAssign
//     + SubAssign
//     + MulAssign
//     + DivAssign
//     + PartialOrd
//     + MulAdd
//     + GetDataType
//     + Cast<f64>
//     + Cast<f32>
//     + Cast<usize>
//     + Cast<i32>
//     + Cast<i64>
//     + 'static
// {
//     // type Dtype;
//     /// return the min value of the data type
//     fn min_() -> Self;

//     /// return the max value of the data type
//     fn max_() -> Self;

//     #[inline(always)]
//     fn kh_sum(&mut self, v: Self, c: &mut Self) {
//         *self = kh_sum(*self, v, c);
//     }

//     /// return min(self, other)
//     /// note that self should not be nan
//     #[inline(always)]
//     fn min_with(self, other: Self) -> Self {
//         if other < self {
//             // if other is nan, this condition should be false
//             other
//         } else {
//             self
//         }
//     }

//     /// return max(self, other)
//     /// note that self should not be nan
//     #[inline(always)]
//     fn max_with(self, other: Self) -> Self {
//         if other > self {
//             // if other is nan, this condition should be false
//             other
//         } else {
//             self
//         }
//     }

//     #[inline(always)]
//     fn f32(self) -> f32 {
//         Cast::<f32>::cast(self)
//     }

//     #[inline(always)]
//     fn f64(self) -> f64 {
//         Cast::<f64>::cast(self)
//     }

//     #[inline(always)]
//     fn i32(self) -> i32 {
//         Cast::<i32>::cast(self)
//     }

//     #[inline(always)]
//     fn i64(self) -> i64 {
//         Cast::<i64>::cast(self)
//     }

//     #[inline(always)]
//     fn usize(self) -> usize {
//         Cast::<usize>::cast(self)
//     }

//     /// create a value of type T using a value of type U using `Cast`
//     #[inline(always)]
//     fn fromas<U>(v: U) -> Self
//     where
//         U: Number + Cast<Self>,
//         Self: 'static,
//     {
//         v.to::<Self>()
//     }

//     /// cast self to another dtype using `Cast`
//     #[inline(always)]
//     fn to<T: Number>(self) -> T
//     where
//         Self: Cast<T>,
//     {
//         Cast::<T>::cast(self)
//     }

//     /// check whether self is nan
//     #[inline(always)]
//     #[allow(clippy::eq_op)]
//     fn isnan(self) -> bool {
//         self != self
//     }
//     /// check whether self is not nan
//     #[inline(always)]
//     #[allow(clippy::eq_op)]
//     fn notnan(self) -> bool {
//         self == self
//     }

//     /// return nan value
//     fn nan() -> Self;

//     /// if other is nan, then add other to self and n += 1
//     /// else just return self
//     #[inline]
//     fn n_add(self, other: Self, n: &mut usize) -> Self {
//         // note: only check if other is NaN
//         // assume that self is not NaN
//         if other.notnan() {
//             *n += 1;
//             self + other
//         } else {
//             self
//         }
//     }

//     /// if other is nan, then product other to self and n += 1
//     /// else just return self
//     #[inline]
//     fn n_prod(self, other: Self, n: &mut usize) -> Self {
//         // note: only check if other is NaN
//         // assume that self is not NaN
//         if other.notnan() {
//             *n += 1;
//             self * other
//         } else {
//             self
//         }
//     }

//     /// if other is nan, then add other to self
//     /// else just return self
//     #[inline(always)]
//     fn nanadd(self, other: Self) -> Self {
//         // note: only check if other is NaN
//         // assume that self is not NaN
//         if other.notnan() {
//             self + other
//         } else {
//             self
//         }
//     }

//     /// let NaN value be largest, only for sorting(from smallest to largest)
//     #[inline(always)]
//     fn nan_sort_cmp(&self, other: &Self) -> Ordering {
//         if other.isnan() | (self < other) {
//             Ordering::Less
//         } else {
//             Ordering::Greater
//         }
//     }

//     /// let NaN value be smallest, only for sorting(from largest to smallest)
//     #[inline]
//     fn nan_sort_cmp_rev(&self, other: &Self) -> Ordering {
//         if other.isnan() | (self > other) {
//             Ordering::Less
//         } else {
//             Ordering::Greater
//         }
//     }

//     /// let NaN value be largest, only for sorting(from smallest to largest)
//     #[inline]
//     fn nan_sort_cmp_stable(&self, other: &Self) -> Ordering {
//         if other.isnan() {
//             if self.isnan() {
//                 Ordering::Equal
//             } else {
//                 Ordering::Less
//             }
//         } else if self.isnan() | (self > other) {
//             Ordering::Greater
//         } else if self == other {
//             Ordering::Equal
//         } else {
//             Ordering::Less
//         }
//     }

//     #[inline]
//     fn nan_sort_cmp_rev_stable(&self, other: &Self) -> Ordering {
//         if other.isnan() {
//             if self.isnan() {
//                 Ordering::Equal
//             } else {
//                 Ordering::Less
//             }
//         } else if self.isnan() | (self < other) {
//             Ordering::Greater
//         } else if self == other {
//             Ordering::Equal
//         } else {
//             Ordering::Less
//         }
//     }
// }

// macro_rules! impl_number {
//     // base impl for number
//     (@ base_impl $dtype:ty, $datatype:ident) => {
//         // type Dtype = $datatype;

//         #[inline(always)]
//         fn min_() -> $dtype {
//             <$dtype>::MIN
//         }

//         #[inline(always)]
//         fn max_() -> $dtype {
//             <$dtype>::MAX
//         }

//     };
//     // special impl for float
//     (float $($dtype:ty, $datatype:ident); *) => {
//         $(impl Number for $dtype {
//             impl_number!(@ base_impl $dtype, $datatype);

//             #[inline(always)]
//             fn nan() -> Self {
//                 <$dtype>::NAN
//             }

//             // #[inline(always)]
//             // fn kh_sum(&mut self, v: Self, c: &mut Self) {
//             //     *self = kh_sum(*self, v, c);
//             // }
//         })*
//     };
//     // special impl for other type
//     (other $($dtype:ty, $datatype:ident); *) => {
//         $(impl Number for $dtype {
//             impl_number!(@ base_impl $dtype, $datatype);

//             #[inline]
//             fn nan() -> Self {
//                 panic!("This type of number doesn't have NaN value")
//             }

//             // these types doen't have NaN
//             #[inline(always)]
//             fn isnan(self) -> bool {
//                 false
//             }

//             #[inline(always)]
//             fn notnan(self) -> bool {
//                 true
//             }
//             #[inline(always)]
//             fn kh_sum(&mut self, v: Self, _c: &mut Self) {
//                 *self += v;
//             }
//         })*
//     };
// }

// impl_number!(
//     float
//     f32, F32;
//     f64, F64
// );

// impl_number!(
//     other
//     i32, I32;
//     i64, I64;
//     u64, U64;
//     usize, Usize
// );
