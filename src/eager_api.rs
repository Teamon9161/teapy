use pyo3::{types::PyModule, PyResult};

pub(crate) fn add_eager(_m: &PyModule) -> PyResult<()> {
    Ok(())
}
