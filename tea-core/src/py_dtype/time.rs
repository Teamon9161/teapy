use crate::prelude::*;
use pyo3::prelude::*;

impl<U: TimeUnitTrait> Cast<Object> for DateTime<U>
where
    CrDateTime<Utc>: From<Self>,
{
    #[inline]
    fn cast(self) -> Object {
        Python::with_gil(|py| Object(self.to_cr().unwrap().to_object(py)))
    }
}

impl<U: TimeUnitTrait> Cast<DateTime<U>> for Object
where
    DateTime<U>: TryFrom<CrDateTime<Utc>> + From<CrDateTime<Utc>>,
{
    #[inline]
    fn cast(self) -> DateTime<U> {
        Python::with_gil(|py| {
            if let Ok(v) = self.extract::<CrDateTime<Utc>>(py) {
                v.into()
            } else if let Ok(s) = self.extract::<std::borrow::Cow<'_, str>>(py) {
                DateTime::parse(s.as_ref(), None).unwrap_or_default()
            } else {
                DateTime::nat()
            }
        })
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
        Python::with_gil(|py| {
            if let Ok(v) = self.extract(py) {
                TimeDelta {
                    months: 0,
                    inner: v,
                }
            } else if let Ok(s) = self.extract::<std::borrow::Cow<'_, str>>(py) {
                TimeDelta::parse(s.as_ref()).unwrap_or_default()
            } else {
                TimeDelta::nat()
            }
        })
    }
}
