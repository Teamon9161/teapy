#![feature(hash_raw_entry)]
#![feature(arbitrary_self_types)]
#![feature(drain_filter)]

#[macro_use]
pub mod from_py;

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

// #[cfg(any(feature = "intel-mkl-system", feature = "intel-mkl-static"))]
// extern crate intel_mkl_src as _src;

// #[cfg(any(feature = "openblas-system", feature = "openblas-static"))]
// extern crate openblas_src as _src;

#[pymodule]
pub fn teapy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
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
