use pyo3::{FromPyObject, PyAny, PyResult};

#[cfg(feature = "agg")]
use crate::{CorrMethod, QuantileMethod};

#[cfg(feature = "arr_func")]
use crate::FillMethod;
#[cfg(all(feature = "agg", feature = "arr_func"))]
use crate::WinsorizeMethod;

#[cfg(all(feature = "lazy", feature = "arr_func", feature = "agg"))]
use crate::lazy::DropNaMethod;
#[cfg(all(feature = "lazy", feature = "groupby"))]
use crate::lazy::JoinType;

#[cfg(all(feature = "lazy", feature = "window_func"))]
use pyo3::exceptions::PyValueError;

#[cfg(all(feature = "lazy", feature = "window_func"))]
use crate::lazy::RollingTimeStartBy;

#[cfg(feature = "agg")]
impl<'source> FromPyObject<'source> for CorrMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("pearson").to_lowercase();
        let out = match s.as_str() {
            "pearson" => CorrMethod::Pearson,
            #[cfg(feature = "arr_func")]
            "spearman" => CorrMethod::Spearman,
            _ => panic!("Not supported method: {s} in correlation"),
        };
        Ok(out)
    }
}

#[cfg(feature = "arr_func")]
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

#[cfg(all(feature = "agg", feature = "arr_func"))]
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
            _ => panic!("Not supported quantile method: {s}"),
        };
        Ok(out)
    }
}

#[cfg(all(feature = "lazy", feature = "groupby"))]
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

#[cfg(all(feature = "lazy", feature = "arr_func", feature = "agg"))]
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
