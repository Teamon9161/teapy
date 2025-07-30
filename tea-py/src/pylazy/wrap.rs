use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::ops::Deref;
use teapy_core::prelude::*;

#[repr(transparent)]
pub struct Wrap<T>(pub T);

impl<T> Deref for Wrap<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[inline]
fn extract_str<'py>(ob: &'py Bound<'_, PyAny>) -> PyResult<Option<std::borrow::Cow<'py, str>>> {
    if ob.is_none() {
        Ok(None)
    } else {
        Ok(Some(ob.extract()?))
    }
}

#[cfg(feature = "agg")]
impl<'source> FromPyObject<'source> for Wrap<CorrMethod> {
    fn extract_bound(ob: &Bound<'source, PyAny>) -> PyResult<Self> {
        let s = extract_str(ob)?;
        let s = s.unwrap_or("pearson".into()).to_lowercase();
        let out = match s.as_str() {
            "pearson" => Wrap(CorrMethod::Pearson),
            #[cfg(feature = "map")]
            "spearman" => Wrap(CorrMethod::Spearman),
            _ => Err(PyValueError::new_err("Not supported corr method: {s}"))?,
        };
        Ok(out)
    }
}

// #[cfg(feature = "map")]
// impl<'source> FromPyObject<'source> for FillMethod {
//     fn extract(ob: &'source PyAny) -> PyResult<Self> {
//         let s: Option<&str> = ob.extract()?;
//         let s = s.unwrap_or("ffill").to_lowercase();
//         let out = match s.as_str() {
//             "ffill" => FillMethod::Ffill,
//             "bfill" => FillMethod::Bfill,
//             "vfill" => FillMethod::Vfill,
//             _ => Err(PyValueError::new_err("Not supported fillna method: {s}"))?,
//         };
//         Ok(out)
//     }
// }

#[cfg(all(feature = "agg", feature = "map"))]
impl<'source> FromPyObject<'source> for Wrap<WinsorizeMethod> {
    fn extract_bound(ob: &Bound<'source, PyAny>) -> PyResult<Self> {
        let s = extract_str(ob)?;
        let s = s.unwrap_or("quantile".into()).to_lowercase();
        let out = match s.as_str() {
            "quantile" => Wrap(WinsorizeMethod::Quantile),
            "median" => Wrap(WinsorizeMethod::Median),
            "sigma" => Wrap(WinsorizeMethod::Sigma),
            _ => Err(PyValueError::new_err("Not supported winsorize method: {s}"))?,
        };
        Ok(out)
    }
}

#[cfg(feature = "agg")]
impl<'source> FromPyObject<'source> for Wrap<QuantileMethod> {
    fn extract_bound(ob: &Bound<'source, PyAny>) -> PyResult<Self> {
        let s = extract_str(ob)?;
        let s = s.unwrap_or("linear".into()).to_lowercase();
        let out = match s.as_str() {
            "linear" => Wrap(QuantileMethod::Linear),
            "lower" => Wrap(QuantileMethod::Lower),
            "higher" => Wrap(QuantileMethod::Higher),
            "midpoint" => Wrap(QuantileMethod::MidPoint),
            _ => Err(PyValueError::new_err("Not supported quantile method: {s}"))?,
        };
        Ok(out)
    }
}

// #[cfg(all(feature = "lazy", feature = "map", feature = "agg"))]
// impl<'source> FromPyObject<'source> for DropNaMethod {
//     fn extract(ob: &'source PyAny) -> PyResult<Self> {
//         let s: Option<&str> = ob.extract()?;
//         let s = s.unwrap_or("any").to_lowercase();
//         let out = match s.as_str() {
//             "all" => DropNaMethod::All,
//             "any" => DropNaMethod::Any,
//             _ => Err(PyValueError::new_err("Not supported dropna method: {s}"))?,
//         };
//         Ok(out)
//     }
// }
