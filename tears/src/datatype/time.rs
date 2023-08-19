use chrono::{Datelike, Duration, Months, NaiveDateTime};
use ndarray::ScalarOperand;
use numpy::{
    datetime::{Datetime as NPDatetime, Unit as NPUnit},
    npyffi::NPY_DATETIMEUNIT,
};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    hash::Hash,
    ops::{Add, Deref, Div, Mul, Neg, Sub},
};

use crate::Cast;

/// The number of nanoseconds in a microsecond.
const NANOS_PER_MICRO: i32 = 1000;
/// The number of nanoseconds in a millisecond.
const NANOS_PER_MILLI: i32 = 1_000_000;
/// The number of nanoseconds in seconds.
const NANOS_PER_SEC: i32 = 1_000_000_000;
/// The number of microseconds per second.
const MICROS_PER_SEC: i64 = 1_000_000;
/// The number of milliseconds per second.
const MILLIS_PER_SEC: i64 = 1000;
/// The number of seconds in a minute.
const SECS_PER_MINUTE: i64 = 60;
/// The number of seconds in an hour.
const SECS_PER_HOUR: i64 = 3600;
/// The number of (non-leap) seconds in days.
const SECS_PER_DAY: i64 = 86400;
/// The number of (non-leap) seconds in a week.
const SECS_PER_WEEK: i64 = 604800;

#[derive(Clone, Copy, Default, Hash, Eq, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct DateTime(pub Option<NaiveDateTime>);

impl std::fmt::Debug for DateTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(dt) = self.0 {
            write!(f, "{dt}")
        } else {
            write!(f, "None")
        }
    }
}

impl From<Option<NaiveDateTime>> for DateTime {
    fn from(dt: Option<NaiveDateTime>) -> Self {
        Self(dt)
    }
}

impl Deref for DateTime {
    type Target = Option<NaiveDateTime>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Cast<i64> for DateTime {
    fn cast(self) -> i64 {
        self.map_or(i64::MIN, |dt| dt.timestamp())
    }
}

// impl PartialEq for DateTime {
//     fn eq(&self, other: &Self) -> bool {
//         self.0.eq(&other.0)
//     }
// }

// impl PartialOrd for DateTime {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         self.0.partial_cmp(&other.0)
//     }
// }

impl DateTime {
    #[inline]
    pub fn from_timestamp_opt(secs: i64, nsecs: u32) -> Self {
        Self(NaiveDateTime::from_timestamp_opt(secs, nsecs))
    }

    #[inline]
    pub fn from_timestamp_ms(ms: i64) -> Option<Self> {
        // if ms == i64::MIN {
        //     return Self::from_timestamp_opt(ms, 0)
        // }
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
        // if us == i64::MIN {
        //     return Self::from_timestamp_opt(us, 0)
        // }
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
        // if ns == i64::MIN {
        //     return Self::from_timestamp_opt(ns, 0)
        // }
        let mut secs = ns / (NANOS_PER_SEC as i64);
        if ns < 0 {
            secs = secs.checked_sub(1)?;
        }

        let nsecs = (ns % (NANOS_PER_SEC as i64)).abs();
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

    pub fn strftime(&self, fmt: Option<String>) -> String {
        if let Some(fmt) = fmt {
            self.0
                .map_or("NaT".to_string(), |dt| dt.format(&fmt).to_string())
        } else {
            self.0.map_or("NaT".to_string(), |dt| dt.to_string())
        }
    }
}

impl ToString for DateTime {
    fn to_string(&self) -> String {
        self.strftime(None)
        // self.0.to_string()
    }
}

impl<U: NPUnit> From<NPDatetime<U>> for DateTime {
    fn from(dt: NPDatetime<U>) -> Self {
        use NPY_DATETIMEUNIT::*;
        let value: i64 = dt.into();
        if value == i64::MIN {
            return DateTime(None);
        }
        match U::UNIT {
            NPY_FR_ms => DateTime::from_timestamp_ms(dt.into()).unwrap_or_default(),
            NPY_FR_us => DateTime::from_timestamp_us(dt.into()).unwrap_or_default(),
            NPY_FR_ns => DateTime::from_timestamp_ns(dt.into()).unwrap_or_default(),
            _ => unimplemented!("not support other timeunit yet"),
        }
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
                NPY_FR_ns => dt.timestamp_nanos().into(),
                _ => unreachable!(),
            }
        } else {
            i64::MIN.into()
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TimeUnit {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    Millisecond,
    #[default]
    Microsecond,
    Nanosecond,
}

impl std::fmt::Debug for TimeUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeUnit::Year => write!(f, "Year"),
            TimeUnit::Month => write!(f, "Month"),
            TimeUnit::Day => write!(f, "Day"),
            TimeUnit::Hour => write!(f, "Hour"),
            TimeUnit::Minute => write!(f, "Minute"),
            TimeUnit::Second => write!(f, "Second"),
            TimeUnit::Millisecond => write!(f, "Millisecond"),
            TimeUnit::Microsecond => write!(f, "Microsecond"),
            TimeUnit::Nanosecond => write!(f, "Nanosecond"),
        }
    }
}

#[serde_with::serde_as]
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct TimeDelta {
    pub months: i32,
    #[serde_as(as = "serde_with::DurationSeconds<i64>")]
    pub inner: Duration,
}

impl Default for TimeDelta {
    fn default() -> Self {
        TimeDelta::nat()
    }
}

impl Add<TimeDelta> for DateTime {
    type Output = DateTime;
    fn add(self, rhs: TimeDelta) -> Self::Output {
        if let Some(dt) = self.0 && rhs.is_not_nat() {
            let out = if rhs.months != 0 {
                if rhs.months > 0 {
                    dt + Months::new(rhs.months as u32)
                } else {
                    dt - Months::new((-rhs.months) as u32)
                }
            } else {
                dt
            };
            DateTime(Some(out + rhs.inner))
        } else {
            DateTime(None)
        }
    }
}

impl Sub<TimeDelta> for DateTime {
    type Output = DateTime;
    fn sub(self, rhs: TimeDelta) -> Self::Output {
        if let Some(dt) = self.0 && rhs.is_not_nat(){
            let out = if rhs.months != 0 {
                if rhs.months > 0 {
                    dt - Months::new(rhs.months as u32)
                } else {
                    dt + Months::new((-rhs.months) as u32)
                }
            } else {
                dt
            };
            DateTime(Some(out + rhs.inner))
        } else {
            DateTime(None)
        }
    }
}

impl Sub<DateTime> for DateTime {
    type Output = TimeDelta;
    fn sub(self, rhs: DateTime) -> Self::Output {
        if let (Some(dt1), Some(dt2)) = (self.0, rhs.0) {
            let r_year = dt2.year();
            let years = dt1.year() - r_year;
            let months = dt1.month() as i32 - dt2.month() as i32;
            let duration =
                dt1.with_year(r_year).unwrap().with_month(1).unwrap() - dt2.with_month(1).unwrap();
            TimeDelta {
                months: 12 * years + months,
                inner: duration,
            }
        } else {
            TimeDelta::nat()
        }
    }
}

impl TimeDelta {
    /// 1ns // 1 nanosecond
    /// 1us // 1 microsecond
    /// 1ms // 1 millisecond
    /// 1s  // 1 second
    /// 1m  // 1 minute
    /// 1h  // 1 hour
    /// 1d  // 1 day
    /// 1w  // 1 week
    /// 1mo // 1 calendar month
    /// 1y  // 1 calendar year
    ///
    /// Parse timedelta from string
    ///
    /// for example: "2y1mo-3d5h-2m3s"
    pub fn parse(duration: &str) -> Self {
        let mut nsecs = 0;
        let mut secs = 0;
        let mut months = 0;
        let mut iter = duration.char_indices();
        let mut start = 0;
        let mut unit = String::with_capacity(2);
        while let Some((i, mut ch)) = iter.next() {
            if !ch.is_ascii_digit() && i != 0 {
                let n = duration[start..i].parse::<i64>().unwrap();
                loop {
                    if ch.is_ascii_alphabetic() {
                        unit.push(ch)
                    } else {
                        break;
                    }
                    match iter.next() {
                        Some((i, ch_)) => {
                            ch = ch_;
                            start = i
                        }
                        None => {
                            break;
                        }
                    }
                }
                if unit.is_empty() {
                    panic!("expected a unit in the duration string")
                }

                match unit.as_str() {
                    "ns" => nsecs += n as i32,
                    "us" => nsecs += n as i32 * NANOS_PER_MICRO,
                    "ms" => nsecs += n as i32 * NANOS_PER_MILLI,
                    "s" => secs += n,
                    "m" => secs += SECS_PER_MINUTE,
                    "h" => secs += n * SECS_PER_HOUR,
                    "d" => secs += n * SECS_PER_DAY,
                    "w" => secs += n * SECS_PER_WEEK,
                    "mo" => months += n as i32,
                    "y" => months += n as i32 * 12,
                    unit => panic!("unit: '{unit}' not supported"),
                }
                unit.clear();
            }
        }
        let duration = Duration::seconds(secs) + Duration::nanoseconds(nsecs as i64);
        TimeDelta {
            months,
            inner: duration,
        }
    }

    fn nat() -> Self {
        Self {
            months: i32::MIN,
            inner: Duration::seconds(0),
        }
    }

    #[allow(dead_code)]
    fn is_nat(&self) -> bool {
        self.months == i32::MIN
    }

    fn is_not_nat(&self) -> bool {
        self.months != i32::MIN
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

impl Div<i32> for TimeDelta {
    type Output = TimeDelta;

    fn div(self, rhs: i32) -> TimeDelta {
        if self.is_not_nat() {
            // may not as expected
            Self {
                months: self.months / rhs,
                inner: self.inner / rhs,
            }
        } else {
            TimeDelta::nat()
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
