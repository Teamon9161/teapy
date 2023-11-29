use ndarray::{Data, Dimension, Zip};
use tea_core::prelude::*;

#[ext_trait]
impl<S, D> StringExt for ArrBase<S, D>
where
    S: Data<Elem = String>,
    D: Dimension,
{
    fn add_string<S2>(&self, other: &ArrBase<S2, D>) -> Arr<String, D>
    where
        S2: Data<Elem = String>,
    {
        Zip::from(&self.0)
            .and(&other.0)
            .par_map_collect(|s1, s2| s1.to_owned() + s2)
            .wrap()
    }

    fn add_str<'a, S2: Data<Elem = &'a str>>(&self, other: &ArrBase<S2, D>) -> Arr<String, D> {
        Zip::from(&self.0)
            .and(&other.0)
            .par_map_collect(|s1, s2| s1.to_owned() + s2)
            .wrap()
    }

    #[cfg(feature = "time")]
    fn strptime(&self, fmt: String) -> Arr<DateTime, D> {
        self.map(|s| DateTime::parse(s, fmt.as_str()).unwrap_or_default())
    }
}
