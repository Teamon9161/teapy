#![feature(hash_raw_entry)]
#![feature(arc_unwrap_or_clone)]
#![feature(arbitrary_self_types)]
#![feature(get_mut_unchecked)]
#![feature(fn_traits)]
#![feature(min_specialization)]
#![feature(drain_filter)]
// #![feature(vec_into_raw_parts)]

#[macro_use]
pub mod arr;
#[macro_use]
pub mod from_py;
// #[cfg(feature = "eager_api")]
// pub(crate) mod eager_macros;

// #[cfg(feature = "eager_api")]
// mod eager_api;

#[cfg(feature = "lazy")]
pub mod pylazy;

#[cfg(feature = "lazy")]
mod equity;

use pyo3::{pyfunction, pymodule, types::PyModule, wrap_pyfunction, PyResult, Python};

// #[cfg(feature = "eager_api")]
// use eager_api::add_eager;

#[cfg(feature = "lazy")]
use crate::pylazy::add_lazy;

#[cfg(not(feature = "lazy"))]
fn add_lazy(_m: &PyModule) -> PyResult<()> {
    Ok(())
}

// #[cfg(not(feature = "eager_api"))]
// fn add_eager(_m: &PyModule) -> PyResult<()> {
//     Ok(())
// }

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[pyfunction]
pub fn get_version() -> &'static str {
    VERSION
}

#[pymodule]
fn teapy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    add_lazy(m)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(equity::calc_digital_ret, m)?)?;
    m.add_function(wrap_pyfunction!(equity::calc_ret_single, m)?)?;
    Ok(())
}
