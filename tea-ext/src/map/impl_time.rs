use ndarray::{Data, Dimension};
use tea_core::prelude::*;

#[ext_trait]
impl<S, D, U: TimeUnitTrait> TimeExt for ArrBase<S, D>
where
    S: Data<Elem = DateTime<U>>,
    D: Dimension,
{
    #[inline]
    fn strftime(&self, fmt: Option<&str>) -> Arr<String, D>
    where
        DateTime<U>: TryInto<CrDateTime<Utc>>,
        <DateTime<U> as TryInto<CrDateTime<Utc>>>::Error: std::fmt::Debug,
    {
        self.map(|dt| dt.strftime(fmt))
    }

    // fn sub_datetime<S2>(&self, other: &ArrBase<S2, D>, par: bool) -> Arr<TimeDelta, D>
    // where
    //     S2: Data<Elem = DateTime>,
    // {
    //     if !par {
    //         Zip::from(&self.0)
    //             .and(&other.0)
    //             .map_collect(|v1, v2| *v1 - *v2)
    //             .wrap()
    //     } else {
    //         Zip::from(&self.0)
    //             .and(&other.0)
    //             .par_map_collect(|v1, v2| *v1 - *v2)
    //             .wrap()
    //     }
    // }
}
