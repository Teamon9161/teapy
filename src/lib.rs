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
#[cfg(feature = "eager_api")]
pub(crate) mod eager_macros;

#[cfg(feature = "eager_api")]
mod eager_api;

#[cfg(feature = "lazy")]
mod pylazy;

#[cfg(feature = "lazy")]
mod equity;

use pyo3::{pymodule, types::PyModule, wrap_pyfunction, PyResult, Python};

#[cfg(feature = "eager_api")]
use eager_api::add_eager;

#[cfg(feature = "lazy")]
use crate::pylazy::add_lazy;

#[cfg(not(feature = "lazy"))]
fn add_lazy(_m: &PyModule) -> PyResult<()> {
    Ok(())
}

#[cfg(not(feature = "eager_api"))]
fn add_eager(_m: &PyModule) -> PyResult<()> {
    Ok(())
}

#[pymodule]
fn teapy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    add_lazy(m)?;
    add_eager(m)?;
    m.add_function(wrap_pyfunction!(equity::calc_digital_ret, m)?)?;
    Ok(())
}
