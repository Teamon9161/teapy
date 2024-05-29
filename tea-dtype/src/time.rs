use super::{DateTime, Object, TimeDelta};
use numpy::{
    datetime::{Datetime as NPDatetime, Unit as NPUnit},
    npyffi::NPY_DATETIMEUNIT,
};
use pyo3::prelude::*;
use tevec::dtype::chrono::{DateTime as CrDateTime, Duration, Utc};
use tevec::prelude::*;

impl Cast<Object> for DateTime {
    #[inline]
    fn cast(self) -> Object {
        Python::with_gil(|py| Object(self.0.to_object(py)))
    }
}

impl Cast<DateTime> for Object {
    #[inline]
    fn cast(self) -> DateTime {
        let v: Option<CrDateTime<Utc>> = Python::with_gil(|py| self.0.extract(py).ok());
        DateTime(v)
    }
}

impl Cast<Object> for TimeDelta {
    #[inline]
    fn cast(self) -> Object {
        if self.months != 0 {
            panic!("TimeDelta with months can not be cast to Object")
        }
        Python::with_gil(|py| Object(self.inner.to_object(py)))
    }
}

impl Cast<TimeDelta> for Object {
    #[inline]
    fn cast(self) -> TimeDelta {
        let v: Option<Duration> = Python::with_gil(|py| self.0.extract(py).ok());
        if let Some(v) = v {
            TimeDelta {
                months: 0,
                inner: v,
            }
        } else {
            TimeDelta::nat()
        }
    }
}

pub trait DateTimeToPy {
    fn into_np_datetime<T: NPUnit>(self) -> NPDatetime<T>;
}

impl DateTimeToPy for DateTime {
    #[inline]
    fn into_np_datetime<T: NPUnit>(self) -> NPDatetime<T> {
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
}

pub trait DateTimeToRs {
    fn to_rs(self) -> TResult<DateTime>;
}

impl<U: NPUnit> DateTimeToRs for NPDatetime<U> {
    fn to_rs(self) -> TResult<DateTime> {
        use NPY_DATETIMEUNIT::*;
        let value: i64 = self.into();
        if value == i64::MIN {
            return Ok(DateTime(None));
        }
        match U::UNIT {
            NPY_FR_ms => Ok(DateTime::from_timestamp_ms(value).unwrap_or_default()),
            NPY_FR_us => Ok(DateTime::from_timestamp_us(value).unwrap_or_default()),
            NPY_FR_ns => Ok(DateTime::from_timestamp_ns(value).unwrap_or_default()),
            _ => tbail!("not support cast timeunit {:?} to rust yet", U::UNIT),
        }
    }
}
