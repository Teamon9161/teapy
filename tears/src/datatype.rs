use super::utils::kh_sum;
use super::{DateTime, TimeDelta};
use num::{cast::AsPrimitive, traits::MulAdd, FromPrimitive, Num, NumCast, ToPrimitive};
use numpy::Element;
use std::cmp::{Ordering, PartialOrd};
use std::fmt::Display;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

#[derive(PartialEq, Eq, Debug)]
pub enum DataType {
    Bool,
    F32,
    F64,
    I32,
    I64,
    Usize,
    Str,
    String,
    Object,
    DateTime,
    TimeDelta,
    OpUsize,
    // F64Test,
}

macro_rules! match_datatype_arm {
    ($expr: expr, $v: ident, $other_enum: ident, $ty: ty, ($($arm: ident $(($arg: ident))?),*), $body: tt) => {
        match <$ty>::dtype() {
            $(DataType::$arm $(($arg))? => {if let $other_enum::$arm($v) = $expr $body else {panic!("datatype mismatch {:?} {:?}", <$ty>::dtype(), $expr)}},)*
            _ => unreachable!()
        }
    };
}
pub(crate) use match_datatype_arm;

pub trait GetDataType: Send + Sync {
    type Physical;
    fn dtype() -> DataType
    where
        Self: Sized;
}

macro_rules! impl_datatype {
    ($tyname:ident, $physical:ty) => {
        impl GetDataType for $physical {
            type Physical = $physical;
            fn dtype() -> DataType {
                DataType::$tyname
            }
        }
    };
}

impl_datatype!(Bool, bool);
impl_datatype!(F32, f32);
impl_datatype!(F64, f64);
impl_datatype!(I32, i32);
impl_datatype!(I64, i64);
impl_datatype!(Usize, usize);
impl_datatype!(String, String);
impl_datatype!(DateTime, DateTime);
impl_datatype!(TimeDelta, TimeDelta);

impl_datatype!(OpUsize, Option<usize>);

// impl_datatype!(F64Test, Option<f64>);
pub trait GetNone {
    fn none() -> Self;
}

impl GetNone for f64 {
    fn none() -> Self {
        f64::NAN
    }
}

impl GetNone for f32 {
    fn none() -> Self {
        f32::NAN
    }
}

impl GetNone for String {
    fn none() -> Self {
        "None".to_owned()
    }
}

impl<'a> GetDataType for &'a str {
    type Physical = &'a str;
    fn dtype() -> DataType {
        DataType::Str
    }
}

pub trait Number:
    Copy
    + Clone
    + Sized
    + Default
    + Display
    + FromPrimitive
    + Num
    + NumCast
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + PartialOrd
    + MulAdd
    + GetDataType
    + Element
    + ToPrimitive
    + AsPrimitive<f64>
    + AsPrimitive<f32>
    + AsPrimitive<usize>
    + AsPrimitive<i32>
    + AsPrimitive<i64>
{
    // type Dtype;
    /// return the min value of the data type
    fn min_() -> Self;

    /// return the max value of the data type
    fn max_() -> Self;

    fn kh_sum(&mut self, v: Self, c: &mut Self);

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
        AsPrimitive::<f32>::as_(self)
    }

    #[inline(always)]
    fn f64(self) -> f64 {
        AsPrimitive::<f64>::as_(self)
    }

    #[inline(always)]
    fn i32(self) -> i32 {
        AsPrimitive::<i32>::as_(self)
    }

    #[inline(always)]
    fn i64(self) -> i64 {
        AsPrimitive::<i64>::as_(self)
    }

    #[inline(always)]
    fn usize(self) -> usize {
        AsPrimitive::<usize>::as_(self)
    }

    /// create a value of type T using a value of type U using `AsPrimitive`
    #[inline(always)]
    fn fromas<U>(v: U) -> Self
    where
        U: Number + AsPrimitive<Self>,
    {
        v.to::<Self>()
    }

    /// cast self to another dtype using `AsPrimitive`
    #[inline(always)]
    fn to<T: Number>(self) -> T
    where
        Self: AsPrimitive<T>,
    {
        AsPrimitive::<T>::as_(self)
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
    #[inline(always)]
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
    #[inline(always)]
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

            #[inline(always)]
            fn kh_sum(&mut self, v: Self, c: &mut Self) {
                *self = kh_sum(*self, v, c);
            }
        })*
    };
    // special impl for other type
    (other $($dtype:ty, $datatype:ident); *) => {
        $(impl Number for $dtype {
            impl_number!(@ base_impl $dtype, $datatype);

            #[inline]
            fn nan() -> Self {
                panic!("This type of number doesn't habe NaN value")
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
    usize, Usize
);

pub trait BoolType {
    fn bool_(self) -> bool;
}
impl BoolType for bool {
    fn bool_(self) -> bool {
        self
    }
}
