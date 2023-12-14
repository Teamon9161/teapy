#![feature(stmt_expr_attributes)]

#[macro_use]
pub mod from_py;

pub use tea_core;
pub use tea_ext;
#[cfg(feature = "groupby")]
pub use tea_groupby;
#[cfg(feature = "lazy")]
pub use tea_lazy;

#[cfg(feature = "lazy")]
pub mod pylazy;

#[cfg(feature = "lazy")]
mod equity;

use pyo3::{pyfunction, pymodule, types::PyModule, wrap_pyfunction, PyResult, Python};

#[cfg(feature = "lazy")]
use crate::pylazy::add_lazy;

#[cfg(not(feature = "lazy"))]
pub fn add_lazy(_m: &PyModule) -> PyResult<()> {
    Ok(())
}

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[pyfunction]
pub fn get_version() -> &'static str {
    VERSION
}

#[pymodule]
pub fn tears(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    add_lazy(m)?;
    m.add("nan", f64::NAN)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    // #[cfg(feature = "lazy")]
    // m.add_function(wrap_pyfunction!(equity::calc_digital_ret, m)?)?;
    #[cfg(feature = "lazy")]
    m.add_function(wrap_pyfunction!(equity::calc_ret_single, m)?)?;
    #[cfg(feature = "lazy")]
    m.add_function(wrap_pyfunction!(equity::calc_ret_single_with_spread, m)?)?;
    Ok(())
}
