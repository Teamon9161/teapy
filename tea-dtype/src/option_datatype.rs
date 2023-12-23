use super::{kh_sum, GetNone, MulAdd, Number};
use num::{traits::AsPrimitive, Num, One, Zero};
use pyo3::{FromPyObject, PyAny, PyResult, Python, ToPyObject};
#[cfg(feature = "srd")]
use serde::Serialize;
use std::cmp::{Ordering, PartialOrd};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, Sub, SubAssign};
use std::str::FromStr;

macro_rules! impl_as_primitive_for_opt {
    ($ty: ty, $real: ty, $as_ty: ty) => {
        impl AsPrimitive<$as_ty> for $ty {
            #[inline]
            fn as_(self) -> $as_ty {
                self.0.unwrap_or(<$real>::none()) as $as_ty
            }
        }
    };
}

#[cfg(feature = "option_dtype")]
macro_rules! impl_as_primitive_opt_for_opt {
    ($ty: ty, [$($as_ty: ty: $as_real: ty),*]) => {
        $(impl AsPrimitive<$as_ty> for $ty {
            #[inline]
            fn as_(self) -> $as_ty {
                self.0.map(|v| v as $as_real).into()
            }
        })*
    };
}

macro_rules! define_option_dtype {
    (numeric $typ: ident, $real: ty) => {
        define_option_dtype!($typ, $real);
        define_option_dtype(impl_numeric $typ, $real);
    };
    ($typ: ident, $real: ty) => {
        #[derive(Copy, Clone, Default)]
        #[cfg_attr(feature="srd", derive(Serialize))]
        #[repr(transparent)]
        pub struct $typ(pub Option<$real>);

        impl From<$typ> for Option<$real> {
            #[inline]
            fn from(value: $typ) -> Self {
                value.0
            }
        }

        impl std::fmt::Debug for $typ {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.0 {
                    Some(v) => write!(f, "{}", v),
                    None => write!(f, "None"),
                }
            }
        }

        impl std::fmt::Display for $typ {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.0 {
                    Some(v) => write!(f, "{}", v),
                    None => write!(f, "None"),
                }
            }
        }

        impl From<Option<$real>> for $typ {
            #[inline]
            fn from(value: Option<$real>) -> Self {
                $typ(value)
            }
        }
        impl From<$real> for $typ {
            #[inline]
            fn from(value: $real) -> Self {
                Some(value).into()
            }
        }


        impl FromStr for $typ {
            type Err = <$real as FromStr>::Err;
            #[inline]
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                if s == "None" || &s.to_lowercase() == "nan" {
                    return Ok(None.into());
                }
                <$real>::from_str(s).map(|v| v.into())
            }
        }


        impl GetNone for $typ {
            #[inline]
            fn none() -> Self {
                Self(None)
            }
            #[inline]
            fn is_none(&self) -> bool {
                self.0.is_none()
            }
        }

        impl ToPyObject for $typ {
            #[inline]
            fn to_object(&self, py: Python) -> pyo3::PyObject {
                self.0.to_object(py)
            }
        }


        impl<'source> FromPyObject<'source> for $typ {
            #[inline]
            fn extract(ob: &'source PyAny) -> PyResult<Self> {
                if ob.is_none() {
                    return Ok(None.into());
                } else {
                    let v = <$real>::extract(ob)?;
                    Ok(v.into())
                }
            }
        }

        impl $typ {
            #[inline(always)]
            pub fn unwrap(self) -> $real {
                self.0.unwrap()
            }

            #[inline(always)]
            pub fn map<U, F>(self, f: F) -> Option<U>
            where F: FnOnce($real) -> U
            {
                self.0.map(f)
            }

            #[inline(always)]
            pub fn unwrap_or_else<F>(self, f: F) -> $real
            where F: FnOnce() -> $real
            {
                self.0.unwrap_or_else(f)
            }

            #[inline(always)]
            pub fn into_real(self) -> $real {
                self.0.unwrap_or_else(|| <$real>::none())
            }
        }

    };

    (impl_numeric $typ: ident, $real: ty) => {
        impl Add<$typ> for $typ {
            type Output = Self;

            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                match (self.0, rhs.0) {
                    (Some(a), Some(b)) => Self(Some(a + b)),
                    _ => Self(None),
                }
            }
        }

        impl Sub<$typ> for $typ {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                match (self.0, rhs.0) {
                    (Some(a), Some(b)) => Self(Some(a - b)),
                    _ => Self(None),
                }
            }
        }

        impl Mul<$typ> for $typ {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                match (self.0, rhs.0) {
                    (Some(a), Some(b)) => Self(Some(a * b)),
                    _ => Self(None),
                }
            }
        }

        impl Div<$typ> for $typ {
            type Output = Self;

            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                match (self.0, rhs.0) {
                    (Some(a), Some(b)) => Self(Some(a / b)),
                    _ => Self(None),
                }
            }
        }

        impl Rem<$typ> for $typ {
            type Output = Self;

            #[inline]
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
            #[inline]
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
            #[inline]
            fn zero() -> Self {
                Self(Some(<$real>::zero()))
            }

            #[inline]
            fn is_zero(&self) -> bool {
                self.0.map(|v| v.is_zero()).unwrap_or(false)
            }
        }

        impl One for $typ {
            #[inline]
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

            #[inline]
            fn kh_sum(&mut self, v: Self, c: &mut Self) {
                *self = kh_sum(*self, v, c);
            }
        }
        impl_as_primitive_for_opt!($typ, $real, f64);
        impl_as_primitive_for_opt!($typ, $real, f32);
        impl_as_primitive_for_opt!($typ, $real, i64);
        impl_as_primitive_for_opt!($typ, $real, i32);
        impl_as_primitive_for_opt!($typ, $real, usize);
        // impl_as_primitive_for_opt!($typ, $real, bool);
    };
}

define_option_dtype!(OptUsize, usize);
define_option_dtype!(impl_numeric OptUsize, usize);

#[cfg(feature = "option_dtype")]
define_option_dtype!(OptF64, f64);
#[cfg(feature = "option_dtype")]
define_option_dtype!(OptF32, f32);
#[cfg(feature = "option_dtype")]
define_option_dtype!(OptI64, i64);
#[cfg(feature = "option_dtype")]
define_option_dtype!(OptI32, i32);
#[cfg(feature = "option_dtype")]
define_option_dtype!(OptBool, bool);

#[cfg(feature = "option_dtype")]
define_option_dtype!(impl_numeric OptF64, f64);
#[cfg(feature = "option_dtype")]
define_option_dtype!(impl_numeric OptF32, f32);
#[cfg(feature = "option_dtype")]
define_option_dtype!(impl_numeric OptI64, i64);
#[cfg(feature = "option_dtype")]
define_option_dtype!(impl_numeric OptI32, i32);

#[cfg(feature = "option_dtype")]
impl PartialEq for OptBool {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self.0, other.0) {
            (Some(a), Some(b)) => a.eq(&b),
            (None, None) => true,
            _ => false,
        }
    }
}

#[cfg(feature = "option_dtype")]
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
#[cfg(feature = "option_dtype")]
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
#[cfg(feature = "option_dtype")]
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
#[cfg(feature = "option_dtype")]
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
#[cfg(feature = "option_dtype")]
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
