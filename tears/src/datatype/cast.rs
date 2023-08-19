use super::{GetNone, OptUsize};
#[cfg(feature = "option_dtype")]
use super::{OptF32, OptF64, OptI32, OptI64};

pub trait Cast<T>: 'static
where
    T: 'static,
{
    fn cast(self) -> T;
}

macro_rules! impl_numeric_cast {
    (@ $T: ty => $(#[$cfg:meta])* impl $U: ty ) => {
        $(#[$cfg])*
        impl Cast<$U> for $T {
            #[inline] fn cast(self) -> $U { self as $U }
        }
    };
    (@to_option $T: ty => $(#[$cfg:meta])* impl $U: ty: $O: ty ) => {
        $(#[$cfg])*
        impl Cast<$O> for $T {
            #[inline] fn cast(self) -> $O {
                (self as $U).into()
            }
        }
    };
    (@ $T: ty => { $( $U: ty $(: $O: ty)? ),* } ) => {$(
        impl_numeric_cast!(@ $T => impl $U);
        $(impl_numeric_cast!(@to_option $T => impl $U: $O);)?
    )*};
    ($T: ty => { $( $U: ty $(: $O: ty)? ),* } ) => {
        #[cfg(feature="option_dtype")]
        impl_numeric_cast!(@ $T => { $( $U $(: $O)? ),* });
        #[cfg(not(feature = "option_dtype"))]
        impl_numeric_cast!(@ $T => { $( $U),* });
        impl_numeric_cast!(@ $T => { u8, u16, u32, u64, usize: OptUsize });
        #[cfg(not(feature = "option_dtype"))]
        impl_numeric_cast!(@ $T => { i8, i16, i32, i64, isize });
        #[cfg(feature="option_dtype")]
        impl_numeric_cast!(@ $T => { i8, i16, i32: OptI32, i64: OptI64, isize });
    };
}

impl_numeric_cast!(u8 => { char, f32: OptF32, f64: OptF64 });
impl_numeric_cast!(i8 => { f32: OptF32, f64: OptF64 });
impl_numeric_cast!(u16 => { f32: OptF32, f64: OptF64 });
impl_numeric_cast!(i16 => { f32: OptF32, f64: OptF64 });
impl_numeric_cast!(u32 => { f32: OptF32, f64: OptF64 });
impl_numeric_cast!(i32 => { f32: OptF32, f64: OptF64 });
impl_numeric_cast!(u64 => { f32: OptF32, f64: OptF64 });
impl_numeric_cast!(i64 => { f32: OptF32, f64: OptF64 });
impl_numeric_cast!(usize => { f32: OptF32, f64: OptF64 });
impl_numeric_cast!(isize => { f32: OptF32, f64: OptF64 });
impl_numeric_cast!(f32 => { f32: OptF32, f64: OptF64 });
impl_numeric_cast!(f64 => { f32: OptF32, f64: OptF64 });
impl_numeric_cast!(char => { char });
impl_numeric_cast!(bool => {});

macro_rules! impl_option_numeric_cast {
    (@ $T: ty: $Real: ty => $(#[$cfg:meta])* impl $U: ty ) => {
        $(#[$cfg])*
        impl Cast<$U> for $T {
            #[inline] fn cast(self) -> $U { self.unwrap_or_else(|| <$Real>::none()).cast() }
        }
    };
    (@to_option $T: ty: $Real: ty => $(#[$cfg:meta])* impl $U: ty: $O: ty ) => {
        $(#[$cfg])*
        impl Cast<$O> for $T {
            #[inline] fn cast(self) -> $O {
                self.map(|v| Cast::<$U>::cast(v)).into()
            }
        }
    };
    (@ $T: ty: $Real: ty => { $( $U: ty $(: $O: ty)? ),* } ) => {$(
        impl_option_numeric_cast!(@ $T: $Real => impl $U);
        $(impl_option_numeric_cast!(@to_option $T: $Real => impl $U: $O);)?
    )*};
    ($T: ty: $Real: ty => { $( $U: ty $(: $O: ty)? ),* } ) => {
        #[cfg(feature="option_dtype")]
        impl_option_numeric_cast!(@ $T: $Real => { $( $U $(: $O)? ),* });
        #[cfg(not(feature = "option_dtype"))]
        impl_option_numeric_cast!(@ $T: $Real => { $( $U),* });
        impl_option_numeric_cast!(@ $T: $Real => { u8, u16, u32, u64, usize: OptUsize });
        #[cfg(not(feature = "option_dtype"))]
        impl_option_numeric_cast!(@ $T: $Real => { i8, i16, i32, i64, isize });
        #[cfg(feature="option_dtype")]
        impl_option_numeric_cast!(@ $T: $Real => { i8, i16, i32: OptI32, i64: OptI64, isize });
    };
}

impl_option_numeric_cast!(OptUsize: usize => { f32: OptF32, f64: OptF64 });
#[cfg(feature = "option_dtype")]
impl_option_numeric_cast!(OptF32: f32 => { f32: OptF32, f64: OptF64 });
#[cfg(feature = "option_dtype")]
impl_option_numeric_cast!(OptF64: f64 => { f32: OptF32, f64: OptF64 });
#[cfg(feature = "option_dtype")]
impl_option_numeric_cast!(OptI32: i32 => { f32: OptF32, f64: OptF64 });
#[cfg(feature = "option_dtype")]
impl_option_numeric_cast!(OptI64: i64 => { f32: OptF32, f64: OptF64 });
