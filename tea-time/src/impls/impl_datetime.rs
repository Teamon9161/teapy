use chrono::{DateTime as CrDateTime, Utc};
use numpy::{
    datetime::{Datetime as NPDatetime, Unit as NPUnit},
    npyffi::NPY_DATETIMEUNIT,
};
// use serde::{Deserialize, Serialize};
use crate::DateTime;
use std::ops::Deref;

impl std::fmt::Debug for DateTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(dt) = self.0 {
            write!(f, "{dt}")
        } else {
            write!(f, "None")
        }
    }
}

impl Deref for DateTime {
    type Target = Option<CrDateTime<Utc>>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// impl Cast<i64> for DateTime {
//     fn cast(self) -> i64 {
//         self.into_i64()
//     }
// }

impl From<Option<CrDateTime<Utc>>> for DateTime {
    #[inline(always)]
    fn from(dt: Option<CrDateTime<Utc>>) -> Self {
        Self(dt)
    }
}

impl From<CrDateTime<Utc>> for DateTime {
    #[inline(always)]
    fn from(dt: CrDateTime<Utc>) -> Self {
        Self(Some(dt))
    }
}

impl From<i64> for DateTime {
    #[inline]
    fn from(dt: i64) -> Self {
        if dt == i64::MIN {
            return DateTime(None);
        }
        // DateTime::from_timestamp_us(dt).unwrap_or_default()
        DateTime::from_timestamp_ns(dt).unwrap_or_default()
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

impl std::fmt::Display for DateTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.strftime(None))
    }
}
