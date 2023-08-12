use crate::arr::TimeDelta;
use pyo3::{pyclass, pymethods, IntoPy, ToPyObject};

#[pyclass]
pub struct PyTimeDelta(TimeDelta);

#[pymethods]
// #[allow(clippy::borrow_deref_ref)]
impl PyTimeDelta {
    #[staticmethod]
    pub fn parse(rule: &str) -> Self {
        Self(TimeDelta::parse(rule))
    }
}

impl ToPyObject for TimeDelta {
    fn to_object(&self, py: pyo3::Python<'_>) -> pyo3::PyObject {
        PyTimeDelta(self.clone()).into_py(py)
    }
}
