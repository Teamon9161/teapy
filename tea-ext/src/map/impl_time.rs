use ndarray::{Data, Dimension};
use tea_core::prelude::*;

#[ext_trait]
impl<S, D> TimeExt for ArrBase<S, D>
where
    S: Data<Elem = DateTime>,
    D: Dimension,
{
    fn strftime(&self, fmt: Option<&str>) -> Arr<String, D> {
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
