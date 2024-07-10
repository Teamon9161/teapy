// use crate::prelude::*;
use crate::prelude::{ArbArray, Arr, ArrBase, ArrOk, WrapNdarray};
use ndarray::{Data, Dimension, Zip};
use numpy::datetime::{units, Datetime as NPDatetime};
use tevec::prelude::{unit, Cast, CrDateTime, DateTime, TimeDelta, TimeUnit, TimeUnitTrait, Utc};

impl<S, D, U: TimeUnitTrait> ArrBase<S, D>
where
    S: Data<Elem = DateTime<U>>,
    D: Dimension,
{
    pub fn sub_datetime<S2>(&self, other: &ArrBase<S2, D>, par: bool) -> Arr<TimeDelta, D>
    where
        S2: Data<Elem = DateTime<U>>,
        DateTime<U>: From<CrDateTime<Utc>> + TryInto<CrDateTime<Utc>>,
    {
        if !par {
            Zip::from(&self.0)
                .and(&other.0)
                .map_collect(|v1, v2| *v1 - *v2)
                .wrap()
        } else {
            Zip::from(&self.0)
                .and(&other.0)
                .par_map_collect(|v1, v2| *v1 - *v2)
                .wrap()
        }
    }
}

impl<S, D, T> ArrBase<S, D>
where
    S: Data<Elem = T>,
    D: Dimension,
{
    #[inline]
    pub fn cast_datetime<U: TimeUnitTrait>(&self) -> Arr<DateTime<U>, D>
    where
        T: Cast<DateTime<U>> + Clone,
    {
        self.view().map(|v| v.clone().cast())
    }

    pub fn to_datetime<'a>(&self, unit: Option<TimeUnit>) -> ArrOk<'a>
    where
        T: Cast<DateTime<unit::Nanosecond>>
            + Cast<DateTime<unit::Microsecond>>
            + Cast<DateTime<unit::Millisecond>>
            + Clone,
    {
        let unit = unit.unwrap_or_default();
        match unit {
            TimeUnit::Nanosecond => self
                .view()
                .map(|v| Cast::<DateTime<unit::Nanosecond>>::cast(v.clone()))
                .into_dyn()
                .into(),
            TimeUnit::Microsecond => self
                .view()
                .map(|v| Cast::<DateTime<unit::Microsecond>>::cast(v.clone()))
                .into_dyn()
                .into(),
            TimeUnit::Millisecond => self
                .view()
                .map(|v| Cast::<DateTime<unit::Millisecond>>::cast(v.clone()))
                .into_dyn()
                .into(),
            _ => unimplemented!("cast to unit {:?} not implemented", unit),
        }
    }
}

impl<'a> From<ArbArray<'a, NPDatetime<units::Milliseconds>>> for ArrOk<'a> {
    #[inline]
    fn from(a: ArbArray<'a, NPDatetime<units::Milliseconds>>) -> Self {
        // safety: datetime and npdatetime has the same size
        let a: ArbArray<'a, DateTime<unit::Millisecond>> = unsafe { a.into_dtype() };
        ArrOk::DateTimeMs(a)
    }
}

impl<'a> From<ArbArray<'a, NPDatetime<units::Microseconds>>> for ArrOk<'a> {
    #[inline]
    fn from(a: ArbArray<'a, NPDatetime<units::Microseconds>>) -> Self {
        // safety: datetime and npdatetime has the same size
        let a: ArbArray<'a, DateTime<unit::Microsecond>> = unsafe { a.into_dtype() };
        ArrOk::DateTimeUs(a)
    }
}

impl<'a> From<ArbArray<'a, NPDatetime<units::Nanoseconds>>> for ArrOk<'a> {
    #[inline]
    fn from(a: ArbArray<'a, NPDatetime<units::Nanoseconds>>) -> Self {
        // safety: datetime and npdatetime has the same size
        let a: ArbArray<'a, DateTime<unit::Nanosecond>> = unsafe { a.into_dtype() };
        ArrOk::DateTimeNs(a)
    }
}
