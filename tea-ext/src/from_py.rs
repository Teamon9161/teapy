// #[cfg(feature = "agg")]
// use crate::agg::*;
#[cfg(feature = "map")]
use crate::map::*;
#[cfg(all(
    feature = "lazy",
    feature = "agg",
    feature = "rolling",
    feature = "time"
))]
use crate::rolling::*;
use pyo3::exceptions::PyValueError;
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
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

#[inline]
fn extract_str<'py>(ob: &'py Bound<'_, PyAny>) -> PyResult<Option<std::borrow::Cow<'py, str>>> {
    if ob.is_none() {
        Ok(None)
    } else {
        Ok(Some(ob.extract()?))
    }
}

#[cfg(feature = "map")]
impl<'py> FromPyObject<'py> for FillMethod {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let s = extract_str(ob)?;
        let s = s.unwrap_or_else(|| "ffill".into()).to_lowercase();
        let out = match s.as_str() {
            "ffill" => FillMethod::Ffill,
            "bfill" => FillMethod::Bfill,
            "vfill" => FillMethod::Vfill,
            _ => Err(PyValueError::new_err(format!(
                "Not supported fillna method: {}",
                s
            )))?,
        };
        Ok(out)
    }
}

#[cfg(all(feature = "lazy", feature = "map", feature = "agg"))]
impl<'py> FromPyObject<'py> for DropNaMethod {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let s = extract_str(ob)?;
        let s = s.unwrap_or_else(|| "any".into()).to_lowercase();
        let out = match s.as_str() {
            "all" => DropNaMethod::All,
            "any" => DropNaMethod::Any,
            _ => Err(PyValueError::new_err(format!(
                "Not supported dropna method: {}",
                s
            )))?,
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
impl<'py> FromPyObject<'py> for RollingTimeStartBy {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let s = extract_str(ob)?;
        let s = s.unwrap_or_else(|| "full".into()).to_lowercase();
        match s.as_str() {
            "full" => Ok(RollingTimeStartBy::Full),
            "duration_start" | "durationstart" | "ds" => Ok(RollingTimeStartBy::DurationStart),
            _ => Err(PyValueError::new_err(format!(
                "Not supported rolling by time start_by method: {}",
                s
            ))),
        }
    }
}
