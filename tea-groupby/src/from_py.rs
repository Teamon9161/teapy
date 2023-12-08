#[cfg(feature = "lazy")]
use super::join::JoinType;
use pyo3::{FromPyObject, PyAny, PyResult};

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
