// mod datadict;
mod export;
// mod groupby;
mod impl_pyexpr;
mod pyexpr;
mod pyfunc;

// pub use datadict::PyDataDict;
pub use impl_pyexpr::expr_register;
pub use pyexpr::{ExprToPy, IntoPyExpr};
pub use pyexpr::{PyExpr, RefObj};
pub use pyfunc::*;

use pyo3::prelude::{wrap_pyfunction, PyModule, PyResult};

pub(crate) fn add_lazy(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyExpr>()?;
    // m.add_class::<PyDataDict>()?;
    // m.add_class::<PyGroupBy>()?;
    m.add_function(wrap_pyfunction!(expr_register, m)?)?;
    #[cfg(all(feature = "concat", feature = "map"))]
    m.add_function(wrap_pyfunction!(concat_expr_py, m)?)?;
    #[cfg(all(feature = "concat", feature = "map"))]
    m.add_function(wrap_pyfunction!(stack_expr_py, m)?)?;
    m.add_function(wrap_pyfunction!(eval_exprs, m)?)?;
    // m.add_function(wrap_pyfunction!(eval_dicts, m)?)?;
    #[cfg(feature = "agg")]
    m.add_function(wrap_pyfunction!(corr_py, m)?)?;
    #[cfg(feature = "map")]
    m.add_function(wrap_pyfunction!(where_py, m)?)?;
    #[cfg(feature = "create")]
    m.add_function(wrap_pyfunction!(full, m)?)?;
    #[cfg(feature = "create")]
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    #[cfg(feature = "time")]
    m.add_function(wrap_pyfunction!(datetime, m)?)?;
    #[cfg(feature = "time")]
    m.add_function(wrap_pyfunction!(timedelta, m)?)?;
    // m.add_function(wrap_pyfunction!(from_dataframe, m)?)?;
    #[cfg(feature = "blas")]
    m.add_function(wrap_pyfunction!(get_newey_west_adjust_s, m)?)?;
    m.add_function(wrap_pyfunction!(parse_expr, m)?)?;
    m.add_function(wrap_pyfunction!(parse_expr_list, m)?)?;
    m.add_function(wrap_pyfunction!(context, m)?)?;
    // #[cfg(all(feature = "arw", feature = "io"))]
    // m.add_function(wrap_pyfunction!(read_ipc, m)?)?;
    #[cfg(all(feature = "arw", feature = "io"))]
    m.add_function(wrap_pyfunction!(scan_ipc, m)?)?;
    Ok(())
}
