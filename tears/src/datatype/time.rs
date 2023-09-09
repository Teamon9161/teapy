use chrono::{Datelike, Duration, DurationRound, Months, NaiveDateTime, NaiveTime, Timelike};
use core::panic;
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

use crate::{Cast, GetNone, OptUsize};
// #[cfg(feature = "option_dtype")]
// use crate::{OptF32, OptF64, OptI32, OptI64};

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

impl From<NaiveDateTime> for DateTime {
    fn from(dt: NaiveDateTime) -> Self {
        Self(Some(dt))
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
        self.into_i64()
    }
}

impl Cast<String> for DateTime {
    fn cast(self) -> String {
        self.to_string()
    }
}

impl Cast<DateTime> for String {
    fn cast(self) -> DateTime {
        for rule in TIME_RULE_VEC {
            if let Ok(dt) = DateTime::parse(&self, rule) {
                return dt;
            }
        }
        panic!("can not parse datetime from string: {self}")
    }
}

impl Cast<DateTime> for &str {
    fn cast(self) -> DateTime {
        for rule in TIME_RULE_VEC {
            if let Ok(dt) = DateTime::parse(self, rule) {
                return dt;
            }
        }
        panic!("can not parse datetime from string: {self}")
    }
}

impl From<i64> for DateTime {
    fn from(dt: i64) -> Self {
        if dt == i64::MIN {
            return DateTime(None);
        }
        DateTime::from_timestamp_us(dt).unwrap_or_default()
    }
}

impl GetNone for DateTime {
    fn none() -> Self {
        Self(None)
    }
}

impl DateTime {
    #[inline]
    pub fn into_i64(self) -> i64 {
        self.map_or(i64::MIN, |dt| dt.timestamp_micros())
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

impl ToString for DateTime {
    fn to_string(&self) -> String {
        self.strftime(None)
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

    #[inline]
    pub fn time(&self) -> Option<NaiveTime> {
        self.0.map(|dt| dt.time())
    }

    #[inline]
    pub fn day(&self) -> OptUsize {
        self.0.map(|dt| dt.day().cast()).into()
    }

    #[inline]
    pub fn month(&self) -> OptUsize {
        self.0.map(|dt| dt.month().cast()).into()
    }

    #[inline]
    pub fn hour(&self) -> OptUsize {
        self.0.map(|dt| dt.hour().cast()).into()
    }

    #[inline]
    pub fn minute(&self) -> OptUsize {
        self.0.map(|dt| dt.minute().cast()).into()
    }

    #[inline]
    pub fn second(&self) -> OptUsize {
        self.0.map(|dt| dt.second().cast()).into()
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

impl From<Duration> for TimeDelta {
    fn from(duration: Duration) -> Self {
        Self {
            months: 0,
            inner: duration,
        }
    }
}

impl From<i64> for TimeDelta {
    fn from(dt: i64) -> Self {
        if dt == i64::MIN {
            return TimeDelta::nat();
        }
        Duration::microseconds(dt).into()
    }
}

impl Cast<i64> for TimeDelta {
    fn cast(self) -> i64 {
        let months = self.months;
        if months != 0 {
            panic!("not support cast TimeDelta to i64 when months is not zero")
        } else {
            self.inner.num_microseconds().unwrap_or(i64::MIN)
        }
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
            let duration = dt1 - dt2;
            TimeDelta {
                months: 0,
                inner: duration,
            }
            // let r_year = dt2.year();
            // let years = dt1.year() - r_year;
            // let months = dt1.month() as i32 - dt2.month() as i32;
            // let duration =
            //     dt1.with_year(r_year).expect(&format!("{dt1} with {r_year}")).with_month(1).expect(&format!("{dt1} with month1")) - dt2.with_month(1).expect(&format!("{dt2} with month1"));
            // TimeDelta {
            //     months: 12 * years + months,
            //     inner: duration,
            // }
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

    pub fn nat() -> Self {
        Self {
            months: i32::MIN,
            inner: Duration::seconds(0),
        }
    }

    #[allow(dead_code)]
    pub fn is_nat(&self) -> bool {
        self.months == i32::MIN
    }

    pub fn is_not_nat(&self) -> bool {
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

impl GetNone for TimeDelta {
    fn none() -> Self {
        TimeDelta::nat()
    }
}

impl Cast<TimeDelta> for &str {
    fn cast(self) -> TimeDelta {
        TimeDelta::parse(self)
    }
}

impl Cast<TimeDelta> for String {
    fn cast(self) -> TimeDelta {
        TimeDelta::parse(&self)
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
