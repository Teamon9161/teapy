use super::{Object, DateTime, TimeDelta};
use tevec::dtype::chrono::{DateTime as CrDateTime, Duration, Utc};
use pyo3::prelude::*;
use tevec::prelude::Cast;

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
                inner: v
            }
        } else {
            TimeDelta::nat()
        }
    }
}

// impl ToPyObject for DateTime {
//     fn to_object(&self, py: Python) -> PyObject {
//         self.0.to_object(py)
//     }
// }

// impl ToPyObject for TimeDelta {
//     fn to_object(&self, py: Python) -> PyObject {
//         if self.months != 0 {
//             panic!("TimeDelta with months can not be cast to Object")
//         }
//         self.inner.to_object(py)
//     }
// }