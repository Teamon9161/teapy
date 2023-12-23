use super::{GetNone, OptUsize};
#[cfg(feature = "option_dtype")]
use super::{OptBool, OptF32, OptF64, OptI32, OptI64};
#[cfg(feature = "time")]
use tea_time::{DateTime, TimeDelta};

pub trait Cast<T> {
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
    (@has_none_to_option $T: ty => $(#[$cfg:meta])* impl $U: ty: $O: ty ) => {
        $(#[$cfg])*
        impl Cast<$O> for $T
        where
            $T: GetNone,
        {
            #[inline] fn cast(self) -> $O {
                if self.is_none() {
                    return None.into();
                } else {
                    (self as $U).into()
                }
            }
        }
    };
    (@ $T: ty => { $( $U: ty $(: $O: ty)? ),* } ) => {$(
        impl_numeric_cast!(@ $T => impl $U);
        $(impl_numeric_cast!(@to_option $T => impl $U: $O);)?
    )*};

    (@has_none $T: ty => { $( $U: ty $(: $O: ty)? ),* } ) => {$(
        impl_numeric_cast!(@ $T => impl $U);
        $(impl_numeric_cast!(@has_none_to_option $T => impl $U: $O);)?
    )*};

    (@common_impl $T: ty => { $( $U: ty $(: $O: ty)? ),* } ) => {
        impl Cast<String> for $T {
            #[inline] fn cast(self) -> String { self.to_string() }
        }
        impl Cast<bool> for $T {
            #[inline] fn cast(self) -> bool {
                let value = Cast::<i32>::cast(self);
                if  value == 0_i32 {
                    false
                } else if value == 1 {
                    true
                } else {
                    panic!("can not cast {value:?} to bool")
                }
            }
        }
        #[cfg(feature="option_dtype")]
        impl Cast<OptBool> for $T {
            #[inline] fn cast(self) -> OptBool {
                if self.is_none() {
                    return None.into()
                } else {
                    Some(Cast::<bool>::cast(self)).into()
                }
            }
        }
        #[cfg(feature="time")]
        impl Cast<DateTime> for $T {
            #[inline] fn cast(self) -> DateTime { Cast::<i64>::cast(self).into() }
        }
        #[cfg(feature="time")]
        impl Cast<TimeDelta> for $T {
            #[inline] fn cast(self) -> TimeDelta { Cast::<i64>::cast(self).into() }
        }
    };

    (has_none $T: ty => { $( $U: ty $(: $O: ty)? ),* } ) => {
        impl_numeric_cast!(@common_impl $T => { $( $U $(: $O)? ),* });
        #[cfg(feature="option_dtype")]
        impl_numeric_cast!(@has_none $T => { $( $U $(: $O)? ),* });
        #[cfg(not(feature = "option_dtype"))]
        impl_numeric_cast!(@ $T => { $( $U),* });
        impl_numeric_cast!(@has_none $T => { u8, u16, u32, u64, usize: OptUsize });
        #[cfg(not(feature = "option_dtype"))]
        impl_numeric_cast!(@ $T => { i8, i16, i32, i64, isize });
        #[cfg(feature="option_dtype")]
        impl_numeric_cast!(@has_none $T => { i8, i16, i32: OptI32, i64: OptI64, isize });
    };

    ($T: ty => { $( $U: ty $(: $O: ty)? ),* } ) => {
        impl_numeric_cast!(@common_impl $T => { $( $U $(: $O)? ),* });

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
impl_numeric_cast!(has_none f32 => { f32: OptF32, f64: OptF64 });
impl_numeric_cast!(has_none f64 => { f32: OptF32, f64: OptF64 });
impl_numeric_cast!(char => { char });
impl_numeric_cast!(bool => {});

macro_rules! impl_bool_cast {
    ($($T: ty),*) => {
        $(
            impl Cast<$T> for bool {
                #[inline] fn cast(self) -> $T { Cast::<i32>::cast(self).cast() }
            }
            #[cfg(feature="option_dtype")]
            impl Cast<$T> for OptBool {
                #[inline] fn cast(self) -> $T {
                    if self.is_none() {
                        <$T>::none()
                    } else {
                        Cast::<$T>::cast(self.unwrap())
                    }
                 }
            }
        )*
    };
}

#[cfg(feature = "time")]
macro_rules! impl_time_cast {
    ($($T: ty),*) => {
        $(
            impl Cast<$T> for DateTime {
                #[inline] fn cast(self) -> $T { Cast::<i64>::cast(self).cast() }
            }


            impl Cast<$T> for TimeDelta {
                #[inline] fn cast(self) -> $T { Cast::<i64>::cast(self).cast() }
            }
        )*

    };
}

#[cfg(feature = "time")]
impl Cast<DateTime> for DateTime {
    #[inline]
    fn cast(self) -> DateTime {
        self
    }
}

#[cfg(feature = "time")]
impl Cast<TimeDelta> for TimeDelta {
    #[inline]
    fn cast(self) -> TimeDelta {
        self
    }
}

#[cfg(feature = "time")]
impl Cast<i64> for DateTime {
    #[inline(always)]
    fn cast(self) -> i64 {
        self.into_i64()
    }
}

#[cfg(feature = "time")]
impl Cast<i64> for TimeDelta {
    #[inline]
    fn cast(self) -> i64 {
        let months = self.months;
        if months != 0 {
            panic!("not support cast TimeDelta to i64 when months is not zero")
        } else {
            self.inner.num_microseconds().unwrap_or(i64::MIN)
        }
    }
}

impl_bool_cast!(f32, f64);
#[cfg(feature = "time")]
impl_time_cast!(f32, f64, i32, u8, u32, u64, usize, isize, bool, OptUsize);

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
        impl Cast<String> for $T {
            #[inline] fn cast(self) -> String { self.to_string() }
        }
        #[cfg(feature="time")]
        impl Cast<DateTime> for $T {
            #[inline] fn cast(self) -> DateTime {
                if Into::<Option<$Real>>::into(self).is_none() {
                    DateTime(None)
                } else {
                    Cast::<i64>::cast(self).into()
                }
            }
        }
        #[cfg(feature="time")]
        impl Cast<TimeDelta> for $T {
            #[inline] fn cast(self) -> TimeDelta {
                if Into::<Option<$Real>>::into(self).is_none() {
                    TimeDelta::nat()
                } else {
                    Cast::<i64>::cast(self).into()
                }
            }
        }
        impl Cast<bool> for $T {
            #[inline] fn cast(self) -> bool {
                // if Into::<Option<$Real>>::into(self).is_none() {
                if self.is_none() {
                    panic!("can not cast None to bool")
                }
                let value = Cast::<i32>::cast(self);
                if  value == 0_i32 {
                    false
                } else if value == 1 {
                    true
                } else {
                    panic!("can not cast {value:?} to bool")
                }
            }
        }
        #[cfg(feature = "option_dtype")]
        impl Cast<OptBool> for $T {
            #[inline] fn cast(self) -> OptBool {
                if self.is_none() {
                    return None.into()
                }
                let value = Cast::<i32>::cast(self);
                if  value == 0_i32 {
                    Some(false).into()
                } else if value == 1 {
                    Some(true).into()
                } else {
                    panic!("can not cast {value:?} to bool")
                }
            }
        }
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
#[cfg(feature = "option_dtype")]
impl_bool_cast!(OptF32, OptF64);
#[cfg(feature = "option_dtype")]
impl_option_numeric_cast!(OptBool: bool => {});

#[cfg(all(feature = "option_dtype", feature = "time"))]
impl_time_cast!(OptF32, OptF64, OptI32, OptI64, OptBool);

macro_rules! impl_cast_from_string {
    ($($T: ty),*) => {
        $(
            impl Cast<$T> for String {
                #[inline] fn cast(self) -> $T { self.parse().expect("Parse string error") }
            }

            impl Cast<$T> for &str {
                #[inline] fn cast(self) -> $T { self.parse().expect("Parse string error") }
            }
        )*
    };
}

impl_cast_from_string!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64, char, bool);

macro_rules! impl_option_cast_from_string {
    ($($T: ty),*) => {
        $(
            impl Cast<$T> for String {
                #[inline] fn cast(self) -> $T { self.parse().expect("Parse string error") }
            }

            impl Cast<$T> for &str {
                #[inline] fn cast(self) -> $T { self.parse().expect("Parse string error") }
            }
        )*
    };
}

impl Cast<String> for String {
    #[inline]
    fn cast(self) -> String {
        self
    }
}

impl Cast<String> for &str {
    #[inline]
    fn cast(self) -> String {
        self.to_string()
    }
}

impl<'a> Cast<&'a str> for &'a str {
    #[inline]
    fn cast(self) -> &'a str {
        self
    }
}

#[cfg(feature = "time")]
impl Cast<String> for DateTime {
    fn cast(self) -> String {
        self.to_string()
    }
}

#[cfg(feature = "time")]
const TIME_RULE_VEC: [&str; 9] = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d",
    "%Y%m%d",
    "%Y%m%d %H%M%S",
    "%d/%m/%Y",
    "%d/%m/%Y H%M%S",
    "%Y%m%d%H%M%S",
    "%d/%m/%YH%M%S",
];

#[cfg(feature = "time")]
impl Cast<DateTime> for String {
    #[inline]
    fn cast(self) -> DateTime {
        for rule in TIME_RULE_VEC {
            if let Ok(dt) = DateTime::parse(&self, rule) {
                return dt;
            }
        }
        panic!("can not parse datetime from string: {self}")
    }
}

#[cfg(feature = "time")]
impl Cast<DateTime> for &str {
    #[inline]
    fn cast(self) -> DateTime {
        for rule in TIME_RULE_VEC {
            if let Ok(dt) = DateTime::parse(self, rule) {
                return dt;
            }
        }
        panic!("can not parse datetime from string: {self}")
    }
}

#[cfg(feature = "time")]
impl Cast<String> for TimeDelta {
    #[inline]
    fn cast(self) -> String {
        format!("{:?}", self)
    }
}

#[cfg(feature = "time")]
impl Cast<TimeDelta> for DateTime {
    #[inline(always)]
    fn cast(self) -> TimeDelta {
        unreachable!()
    }
}

#[cfg(feature = "time")]
impl Cast<DateTime> for TimeDelta {
    #[inline(always)]
    fn cast(self) -> DateTime {
        unreachable!()
    }
}

#[cfg(feature = "time")]
impl Cast<TimeDelta> for &str {
    #[inline(always)]
    fn cast(self) -> TimeDelta {
        TimeDelta::parse(self)
    }
}

#[cfg(feature = "time")]
impl Cast<TimeDelta> for String {
    #[inline(always)]
    fn cast(self) -> TimeDelta {
        TimeDelta::parse(&self)
    }
}

impl_option_cast_from_string!(OptUsize);
#[cfg(feature = "option_dtype")]
impl_option_cast_from_string!(OptF32, OptF64, OptI32, OptI64, OptBool);
