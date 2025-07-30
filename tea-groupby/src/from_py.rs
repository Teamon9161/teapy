#[cfg(feature = "lazy")]
use super::join::JoinType;
use pyo3::{types::PyAnyMethods, Bound, FromPyObject, PyAny, PyResult};

#[inline]
fn extract_str<'py>(ob: &'py Bound<'_, PyAny>) -> PyResult<Option<std::borrow::Cow<'py, str>>> {
    if ob.is_none() {
        Ok(None)
    } else {
        Ok(Some(ob.extract()?))
    }
}

#[cfg(feature = "lazy")]
impl<'source> FromPyObject<'source> for JoinType {
    fn extract_bound(ob: &Bound<'source, PyAny>) -> PyResult<Self> {
        let s = extract_str(ob)?;
        let s = s.unwrap_or("left".into()).to_lowercase();
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
