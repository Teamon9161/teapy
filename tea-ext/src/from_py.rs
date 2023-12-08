#[cfg(feature = "agg")]
use crate::agg::*;
#[cfg(any(
    feature = "agg",
    feature = "map",
    all(
        feature = "lazy",
        feature = "agg",
        feature = "rolling",
        feature = "time"
    )
))]
use pyo3::{exceptions::PyValueError, FromPyObject, PyAny, PyResult};

#[cfg(feature = "map")]
use crate::map::*;

#[cfg(all(
    feature = "lazy",
    feature = "agg",
    feature = "rolling",
    feature = "time"
))]
use crate::rolling::*;

#[cfg(feature = "agg")]
impl<'source> FromPyObject<'source> for CorrMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("pearson").to_lowercase();
        let out = match s.as_str() {
            "pearson" => CorrMethod::Pearson,
            #[cfg(feature = "map")]
            "spearman" => CorrMethod::Spearman,
            _ => Err(PyValueError::new_err("Not supported corr method: {s}"))?,
        };
        Ok(out)
    }
}

#[cfg(feature = "map")]
impl<'source> FromPyObject<'source> for FillMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("ffill").to_lowercase();
        let out = match s.as_str() {
            "ffill" => FillMethod::Ffill,
            "bfill" => FillMethod::Bfill,
            "vfill" => FillMethod::Vfill,
            _ => Err(PyValueError::new_err("Not supported fillna method: {s}"))?,
        };
        Ok(out)
    }
}

#[cfg(all(feature = "agg", feature = "map"))]
impl<'source> FromPyObject<'source> for WinsorizeMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("quantile").to_lowercase();
        let out = match s.as_str() {
            "quantile" => WinsorizeMethod::Quantile,
            "median" => WinsorizeMethod::Median,
            "sigma" => WinsorizeMethod::Sigma,
            _ => Err(PyValueError::new_err("Not supported winsorize method: {s}"))?,
        };
        Ok(out)
    }
}

#[cfg(feature = "agg")]
impl<'source> FromPyObject<'source> for QuantileMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("linear").to_lowercase();
        let out = match s.as_str() {
            "linear" => QuantileMethod::Linear,
            "lower" => QuantileMethod::Lower,
            "higher" => QuantileMethod::Higher,
            "midpoint" => QuantileMethod::MidPoint,
            _ => Err(PyValueError::new_err("Not supported quantile method: {s}"))?,
        };
        Ok(out)
    }
}

#[cfg(all(feature = "lazy", feature = "map", feature = "agg"))]
impl<'source> FromPyObject<'source> for DropNaMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("any").to_lowercase();
        let out = match s.as_str() {
            "all" => DropNaMethod::All,
            "any" => DropNaMethod::Any,
            _ => Err(PyValueError::new_err("Not supported dropna method: {s}"))?,
        };
        Ok(out)
    }
}

#[cfg(all(
    feature = "lazy",
    feature = "agg",
    feature = "rolling",
    feature = "time"
))]
impl<'source> FromPyObject<'source> for RollingTimeStartBy {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("full").to_lowercase();
        match s.as_str() {
            "full" => Ok(RollingTimeStartBy::Full),
            "duration_start" | "durationstart" | "ds" => Ok(RollingTimeStartBy::DurationStart),
            _ => Err(PyValueError::new_err(format!(
                "Not supported rolling by time start_by method: {s}"
            ))),
        }
    }
}
