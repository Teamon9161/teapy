use crate::prelude::*;
use ndarray::{Data, Dimension, Zip};

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
