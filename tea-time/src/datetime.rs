// use core::panic;
use ndarray::ScalarOperand;
use numpy::{
    datetime::{Datetime as NPDatetime, Unit as NPUnit},
    npyffi::NPY_DATETIMEUNIT,
};
use pyo3::prelude::*;
// use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    hash::Hash,
    ops::{Add, Div, Mul, Neg, Sub},
};

use chrono::{Datelike, DurationRound, Months, NaiveDateTime, NaiveTime, Timelike};

use crate::{convert::*, TimeDelta};

#[derive(Clone, Copy, Default, Hash, Eq, PartialEq, PartialOrd)]
pub struct DateTime(pub Option<NaiveDateTime>);

impl DateTime {
    #[inline]
    pub fn into_i64(self) -> i64 {
        // self.map_or(i64::MIN, |dt| dt.timestamp_micros())
        self.0
            .map_or(i64::MIN, |dt| dt.timestamp_nanos_opt().unwrap_or(i64::MIN))
    }

    #[inline]
    pub fn from_timestamp_opt(secs: i64, nsecs: u32) -> Self {
        Self(NaiveDateTime::from_timestamp_opt(secs, nsecs))
    }

    #[inline]
    pub fn from_timestamp_ms(ms: i64) -> Option<Self> {
        let mut secs = ms / MILLIS_PER_SEC;
        if ms < 0 {
            secs = secs.checked_sub(1)?;
        }

        let nsecs = (ms % MILLIS_PER_SEC).abs();
        let nsecs = if nsecs == 0 && ms < 0 {
            secs += 1;
            0
        } else {
            let mut nsecs = u32::try_from(nsecs).ok()? * NANOS_PER_MILLI as u32;
            if secs < 0 {
                nsecs = (NANOS_PER_SEC as u32).checked_sub(nsecs)?;
            }
            nsecs
        };
        Some(Self::from_timestamp_opt(secs, nsecs))
    }

    #[inline]
    pub fn from_timestamp_us(us: i64) -> Option<Self> {
        let mut secs = us / MICROS_PER_SEC;
        if us < 0 {
            secs = secs.checked_sub(1)?;
        }

        let nsecs = (us % MICROS_PER_SEC).abs();
        let nsecs = if nsecs == 0 && us < 0 {
            secs += 1;
            0
        } else {
            let mut nsecs = u32::try_from(nsecs).ok()? * NANOS_PER_MICRO as u32;
            if secs < 0 {
                nsecs = (NANOS_PER_SEC as u32).checked_sub(nsecs)?;
            }
            nsecs
        };
        Some(Self::from_timestamp_opt(secs, nsecs))
    }

    #[inline]
    pub fn from_timestamp_ns(ns: i64) -> Option<Self> {
        let mut secs = ns / NANOS_PER_SEC;
        if ns < 0 {
            secs = secs.checked_sub(1)?;
        }

        let nsecs = (ns % NANOS_PER_SEC).abs();
        let nsecs = if nsecs == 0 && ns < 0 {
            secs += 1;
            0
        } else {
            let mut nsecs = u32::try_from(nsecs).ok()?;
            if secs < 0 {
                nsecs = (NANOS_PER_SEC as u32).checked_sub(nsecs)?;
            }
            nsecs
        };
        Some(Self::from_timestamp_opt(secs, nsecs))
    }

    pub fn parse(s: &str, fmt: &str) -> Result<Self, String> {
        Ok(Self(Some(
            NaiveDateTime::parse_from_str(s, fmt).map_err(|e| format!("{e}"))?,
        )))
    }

    pub fn strftime(&self, fmt: Option<&str>) -> String {
        if let Some(fmt) = fmt {
            self.0
                .map_or("NaT".to_string(), |dt| dt.format(fmt).to_string())
        } else {
            self.0.map_or("NaT".to_string(), |dt| dt.to_string())
        }
    }

    pub fn duration_trunc(self, duration: TimeDelta) -> Self {
        if self.is_none() {
            return self;
        }
        let mut dt = self.0.unwrap();
        let dm = duration.months;
        if dm != 0 {
            let (flag, dt_year) = dt.year_ce();
            if dm < 0 {
                unimplemented!("not support year before ce or negative month")
            }
            let dt_month = if flag {
                (dt_year * 12 + dt.month()) as i32
            } else {
                dt_year as i32 * (-12) + dt.month() as i32
            };
            let delta_down = dt_month % dm;
            dt = match delta_down.cmp(&0) {
                Ordering::Equal => dt,
                Ordering::Greater => dt - Months::new(delta_down as u32),
                Ordering::Less => dt - Months::new((dm - delta_down.abs()) as u32),
            };
            if let Some(nd) = duration.inner.num_nanoseconds() {
                if nd == 0 {
                    return dt.into();
                }
            }
        }
        dt.duration_trunc(duration.inner)
            .expect("Rounding Error")
            .into()
    }
}

#[pyclass]
pub struct PyDateTime(DateTime);

impl ToPyObject for DateTime {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        PyDateTime(*self).into_py(py)
    }
}

impl DateTime {
    pub fn into_np_datetime<T: NPUnit>(self) -> NPDatetime<T> {
        use NPY_DATETIMEUNIT::*;
        if let Some(dt) = self.0 {
            match T::UNIT {
                NPY_FR_ms => dt.timestamp_millis().into(),
                NPY_FR_us => dt.timestamp_micros().into(),
                NPY_FR_ns => dt.timestamp_nanos_opt().unwrap_or(i64::MIN).into(),
                _ => unreachable!(),
            }
        } else {
            i64::MIN.into()
        }
    }

    #[inline]
    pub fn time(&self) -> Option<NaiveTime> {
        self.0.map(|dt| dt.time())
    }

    #[inline]
    pub fn day(&self) -> Option<usize> {
        self.0.map(|dt| dt.day() as usize)
    }

    #[inline]
    pub fn month(&self) -> Option<usize> {
        self.0.map(|dt| dt.month() as usize)
    }

    #[inline]
    pub fn hour(&self) -> Option<usize> {
        self.0.map(|dt| dt.hour() as usize)
    }

    #[inline]
    pub fn minute(&self) -> Option<usize> {
        self.0.map(|dt| dt.minute() as usize)
    }

    #[inline]
    pub fn second(&self) -> Option<usize> {
        self.0.map(|dt| dt.second() as usize)
    }
}

impl Neg for TimeDelta {
    type Output = TimeDelta;

    #[inline]
    fn neg(self) -> TimeDelta {
        if self.is_not_nat() {
            Self {
                months: -self.months,
                inner: -self.inner,
            }
        } else {
            self
        }
    }
}

impl Add for TimeDelta {
    type Output = TimeDelta;

    fn add(self, rhs: TimeDelta) -> TimeDelta {
        if self.is_not_nat() & rhs.is_not_nat() {
            Self {
                months: self.months + rhs.months,
                inner: self.inner + rhs.inner,
            }
        } else {
            TimeDelta::nat()
        }
    }
}

impl Sub for TimeDelta {
    type Output = TimeDelta;

    fn sub(self, rhs: TimeDelta) -> TimeDelta {
        if self.is_not_nat() & rhs.is_not_nat() {
            Self {
                months: self.months - rhs.months,
                inner: self.inner - rhs.inner,
            }
        } else {
            TimeDelta::nat()
        }
    }
}

impl Mul<i32> for TimeDelta {
    type Output = TimeDelta;

    fn mul(self, rhs: i32) -> Self {
        if self.is_not_nat() {
            Self {
                months: self.months * rhs,
                inner: self.inner * rhs,
            }
        } else {
            TimeDelta::nat()
        }
    }
}

impl Div<TimeDelta> for TimeDelta {
    type Output = i32;

    fn div(self, rhs: TimeDelta) -> Self::Output {
        if self.is_not_nat() & rhs.is_not_nat() {
            // may not as expected
            let inner_div =
                self.inner.num_nanoseconds().unwrap() / rhs.inner.num_nanoseconds().unwrap();
            if self.months == 0 || rhs.months == 0 {
                return inner_div as i32;
            }
            let month_div = self.months / rhs.months;
            if month_div == inner_div as i32 {
                month_div
            } else {
                panic!("not support div TimeDelta when month div and time div is not equal")
            }
        } else {
            panic!("not support div TimeDelta when one of them is nat")
        }
    }
}

// impl PartialEq for TimeDelta {
//     fn eq(&self, other: &Self) -> bool {
//         if self.months != other.months {
//             false
//         } else {
//             self.inner.eq(&other.inner)
//         }
//     }
// }

impl PartialOrd for TimeDelta {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.is_not_nat() {
            // may not as expected
            if self.months != other.months {
                self.months.partial_cmp(&other.months)
            } else {
                self.inner.partial_cmp(&other.inner)
            }
        } else {
            None
        }
    }
}

impl ScalarOperand for TimeDelta {}
impl ScalarOperand for DateTime {}

impl From<&str> for TimeDelta {
    fn from(s: &str) -> Self {
        TimeDelta::parse(s)
    }
}

#[pyclass]
pub struct PyTimeDelta(TimeDelta);

#[pymethods]
impl PyTimeDelta {
    #[staticmethod]
    pub fn parse(rule: &str) -> Self {
        Self(TimeDelta::parse(rule))
    }
}

impl ToPyObject for TimeDelta {
    fn to_object(&self, py: pyo3::Python<'_>) -> pyo3::PyObject {
        PyTimeDelta(self.clone()).into_py(py)
    }
}
