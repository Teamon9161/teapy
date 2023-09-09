#[cfg(all(feature = "lazy", feature = "window_func"))]
use crate::lazy::RollingTimeStartBy;
#[cfg(feature = "lazy")]
use crate::lazy::{DropNaMethod, JoinType};
use crate::{CorrMethod, FillMethod, QuantileMethod, WinsorizeMethod};
use pyo3::{exceptions::PyValueError, FromPyObject, PyAny, PyResult};

impl<'source> FromPyObject<'source> for CorrMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("pearson").to_lowercase();
        let out = match s.as_str() {
            "pearson" => CorrMethod::Pearson,
            "spearman" => CorrMethod::Spearman,
            _ => panic!("Not supported method: {s} in correlation"),
        };
        Ok(out)
    }
}

impl<'source> FromPyObject<'source> for FillMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("ffill").to_lowercase();
        let out = match s.as_str() {
            "ffill" => FillMethod::Ffill,
            "bfill" => FillMethod::Bfill,
            "vfill" => FillMethod::Vfill,
            _ => panic!("Not support method: {s} in fillna"),
        };
        Ok(out)
    }
}

impl<'source> FromPyObject<'source> for WinsorizeMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("quantile").to_lowercase();
        let out = match s.as_str() {
            "quantile" => WinsorizeMethod::Quantile,
            "median" => WinsorizeMethod::Median,
            "sigma" => WinsorizeMethod::Sigma,
            _ => panic!("Not support {s} method in winsorize"),
        };
        Ok(out)
    }
}

impl<'source> FromPyObject<'source> for QuantileMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("linear").to_lowercase();
        let out = match s.as_str() {
            "linear" => QuantileMethod::Linear,
            "lower" => QuantileMethod::Lower,
            "higher" => QuantileMethod::Higher,
            "midpoint" => QuantileMethod::MidPoint,
            _ => panic!("Not supported quantile method: {s}"),
        };
        Ok(out)
    }
}

#[cfg(feature = "lazy")]
impl<'source> FromPyObject<'source> for JoinType {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("left").to_lowercase();
        let out = match s.as_str() {
            "left" => JoinType::Left,
            "right" => JoinType::Right,
            "inner" => JoinType::Inner,
            "outer" => JoinType::Outer,
            _ => panic!("Not supported join method: {s}"),
        };
        Ok(out)
    }
}

#[cfg(feature = "lazy")]
impl<'source> FromPyObject<'source> for DropNaMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("any").to_lowercase();
        let out = match s.as_str() {
            "all" => DropNaMethod::All,
            "any" => DropNaMethod::Any,
            _ => panic!("Not supported dropna method: {s}"),
        };
        Ok(out)
    }
}

#[cfg(all(feature = "lazy", feature = "window_func"))]
impl<'source> FromPyObject<'source> for RollingTimeStartBy {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("full").to_lowercase();
        match s.as_str() {
            "full" => Ok(RollingTimeStartBy::Full),
            "duration_start" | "durationstart" | "ds" => Ok(RollingTimeStartBy::DurationStart),
            _ => Err(PyValueError::new_err(
                "Not supported rolling by time start_by method: {s}",
            )),
        }
    }
}
