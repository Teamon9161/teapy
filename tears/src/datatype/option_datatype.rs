use super::{kh_sum, GetNone, MulAdd, Number};
use crate::{Arr, ArrView, ArrViewMut};
use ndarray::Dimension;
use num::{traits::AsPrimitive, Num, One, Zero};
use pyo3::{Python, ToPyObject};
use std::cmp::{Ordering, PartialOrd};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, Sub, SubAssign};

pub trait ArrToOpt {
    type OutType;
    fn to_opt(&self) -> Self::OutType;
}

macro_rules! impl_as_primitive_for_opt {
    ($ty: ty, $real: ty, $as_ty: ty) => {
        impl AsPrimitive<$as_ty> for $ty {
            #[inline(always)]
            fn as_(self) -> $as_ty {
                self.0.unwrap_or(<$real>::nan()) as $as_ty
            }
        }
    };
}

macro_rules! impl_as_primitive_opt_for_opt {
    ($ty: ty, [$($as_ty: ty: $as_real: ty),*]) => {
        $(impl AsPrimitive<$as_ty> for $ty {
            #[inline(always)]
            fn as_(self) -> $as_ty {
                self.0.map(|v| v as $as_real).into()
            }
        })*
    };
}

macro_rules! define_option_dtype {
    (numeric $typ: ident, $real: ty) => {
        define_option_dtype!($typ, $real);
        define_option_dtype!(impl_numeric $typ, $real);
    };
    ($typ: ident, $real: ty) => {
        #[derive(Copy, Clone, Default)]
        #[repr(transparent)]
        pub struct $typ(pub Option<$real>);

        impl std::fmt::Debug for $typ {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.0 {
                    Some(v) => write!(f, "{}", v),
                    None => write!(f, "None"),
                }
            }
        }

        impl From<Option<$real>> for $typ {
            fn from(value: Option<$real>) -> Self {
                $typ(value)
            }
        }
        impl From<$real> for $typ {
            fn from(value: $real) -> Self {
                Some(value).into()
            }
        }

        impl GetNone for $typ {
            fn none() -> Self {
                Self(None)
            }
        }

        impl ToPyObject for $typ {
            fn to_object(&self, py: Python) -> pyo3::PyObject {
                self.0.to_object(py)
            }
        }

        impl<D: Dimension> ArrToOpt for Arr<$real, D>
        {
            type OutType = Arr<$typ, D>;
            fn to_opt(&self) -> Self::OutType
            {
                self.map(|v| $typ(Some(*v)))
            }
        }

        impl<'a, D: Dimension> ArrToOpt for ArrView<'a, $real, D>
        {
            type OutType = Arr<$typ, D>;
            fn to_opt(&self) -> Self::OutType
            {
                self.map(|v| $typ(Some(*v)))
            }
        }

        impl<'a, D: Dimension> ArrToOpt for ArrViewMut<'a, $real, D>
        {
            type OutType = Arr<$typ, D>;
            fn to_opt(&self) -> Self::OutType
            {
                self.map(|v| $typ(Some(*v)))
            }
        }
    };

    (impl_numeric $typ: ident, $real: ty) => {
        impl Add<$typ> for $typ {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self::Output {
                match (self.0, rhs.0) {
                    (Some(a), Some(b)) => Self(Some(a + b)),
                    _ => Self(None),
                }
            }
        }

        impl Sub<$typ> for $typ {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self::Output {
                match (self.0, rhs.0) {
                    (Some(a), Some(b)) => Self(Some(a - b)),
                    _ => Self(None),
                }
            }
        }

        impl Mul<$typ> for $typ {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self::Output {
                match (self.0, rhs.0) {
                    (Some(a), Some(b)) => Self(Some(a * b)),
                    _ => Self(None),
                }
            }
        }

        impl Div<$typ> for $typ {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: Self) -> Self::Output {
                match (self.0, rhs.0) {
                    (Some(a), Some(b)) => Self(Some(a / b)),
                    _ => Self(None),
                }
            }
        }

        impl Rem<$typ> for $typ {
            type Output = Self;

            #[inline(always)]
            fn rem(self, rhs: Self) -> Self::Output {
                match (self.0, rhs.0) {
                    (Some(a), Some(b)) => Self(Some(a % b)),
                    _ => Self(None),
                }
            }
        }

        impl MulAdd for $typ {
            type Output = Self;
            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                match (a.0, b.0) {
                    (Some(a), Some(b)) => a.mul_add(a, b).into(),
                    _ => Self(None),
                }
            }
        }

        impl PartialEq for $typ {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                match (self.0, other.0) {
                    (Some(a), Some(b)) => a.eq(&b),
                    (None, None) => true,
                    _ => false,
                }
            }
        }

        impl PartialOrd for $typ {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                match (self.0, other.0) {
                    (Some(a), Some(b)) => a.partial_cmp(&b),
                    _ => None,
                }
            }
        }

        impl AddAssign for $typ {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                match (self.0, rhs.0) {
                    (Some(a), Some(b)) => *self = Self(Some(a + b)),
                    _ => *self = Self(None),
                }
            }
        }

        impl SubAssign for $typ {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                match (self.0, rhs.0) {
                    (Some(a), Some(b)) => *self = Self(Some(a - b)),
                    _ => *self = Self(None),
                }
            }
        }

        impl MulAssign for $typ {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                match (self.0, rhs.0) {
                    (Some(a), Some(b)) => *self = Self(Some(a * b)),
                    _ => *self = Self(None),
                }
            }
        }

        impl DivAssign for $typ {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                match (self.0, rhs.0) {
                    (Some(a), Some(b)) => *self = Self(Some(a / b)),
                    _ => *self = Self(None),
                }
            }
        }

        impl Zero for $typ {
            #[inline(always)]
            fn zero() -> Self {
                Self(Some(<$real>::zero()))
            }

            #[inline(always)]
            fn is_zero(&self) -> bool {
                self.0.map(|v| v.is_zero()).unwrap_or(false)
            }
        }

        impl One for $typ {
            #[inline(always)]
            fn one() -> Self {
                Self(Some(<$real>::one()))
            }
        }

        impl Num for $typ {
            type FromStrRadixErr = <$real as Num>::FromStrRadixErr;
            #[inline]
            fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                <$real>::from_str_radix(s, radix).map(|v| v.into())
            }
        }

        impl AsPrimitive<$typ> for $real {
            #[inline(always)]
            fn as_(self) -> $typ {
                self.into()
            }
        }

        impl Number for $typ {
            #[inline(always)]
            fn min_() -> Self {
                <$real>::MIN.into()
            }

            #[inline(always)]
            fn max_() -> Self {
                <$real>::MAX.into()
            }

            #[inline]
            fn isnan(self) -> bool {
                if let Some(v) = self.0 {
                    v.isnan()
                } else {
                    true
                }
            }

            #[inline]
            fn notnan(self) -> bool {
                if let Some(v) = self.0 {
                    v.notnan()
                } else {
                    false
                }
            }

            #[inline(always)]
            fn nan() -> Self {
                None.into()
            }

            #[inline(always)]
            fn kh_sum(&mut self, v: Self, c: &mut Self) {
                *self = kh_sum(*self, v, c);
            }
        }
        impl_as_primitive_for_opt!($typ, $real, f64);
        impl_as_primitive_for_opt!($typ, $real, f32);
        impl_as_primitive_for_opt!($typ, $real, i64);
        impl_as_primitive_for_opt!($typ, $real, i32);
        impl_as_primitive_for_opt!($typ, $real, usize);
    };
}

define_option_dtype!(numeric OptF64, f64);
define_option_dtype!(numeric OptF32, f32);
define_option_dtype!(numeric OptI64, i64);
define_option_dtype!(numeric OptI32, i32);
define_option_dtype!(numeric OptUsize, usize);

// define_option_dtype!(OptDateTime, DateTime);

impl_as_primitive_opt_for_opt!(
    OptF32,
    [
        OptF64: f64,
        OptF32: f32,
        OptI32: i32,
        OptI64: i64,
        OptUsize: usize
    ]
);
impl_as_primitive_opt_for_opt!(
    OptF64,
    [
        OptF64: f64,
        OptF32: f32,
        OptI32: i32,
        OptI64: i64,
        OptUsize: usize
    ]
);
impl_as_primitive_opt_for_opt!(
    OptI32,
    [
        OptF64: f64,
        OptF32: f32,
        OptI32: i32,
        OptI64: i64,
        OptUsize: usize
    ]
);
impl_as_primitive_opt_for_opt!(
    OptI64,
    [
        OptF64: f64,
        OptF32: f32,
        OptI32: i32,
        OptI64: i64,
        OptUsize: usize
    ]
);
impl_as_primitive_opt_for_opt!(
    OptUsize,
    [
        OptF64: f64,
        OptF32: f32,
        OptI32: i32,
        OptI64: i64,
        OptUsize: usize
    ]
);
