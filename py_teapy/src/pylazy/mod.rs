mod datadict;
mod export;
// mod groupby;
mod impl_pyexpr;
mod pyexpr;
mod pyfunc;

pub use datadict::PyDataDict;
// pub use groupby::PyGroupBy;
pub use impl_pyexpr::expr_register;
pub use pyexpr::{ExprToPy, IntoPyExpr};
pub use pyexpr::{PyExpr, RefObj};
pub use pyfunc::{
    arange, concat_expr, concat_expr_py, context, datetime, eval_dicts, eval_exprs, from_pandas,
    full, get_newey_west_adjust_s, parse_expr, parse_expr_list, parse_expr_nocopy, stack_expr_py,
    timedelta, where_py,
};

#[cfg(feature = "arw")]
use pyfunc::{read_ipc, scan_ipc};

use pyo3::prelude::{wrap_pyfunction, PyModule, PyResult};

pub(crate) fn add_lazy(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyExpr>()?;
    m.add_class::<PyDataDict>()?;
    // m.add_class::<PyGroupBy>()?;
    m.add_function(wrap_pyfunction!(expr_register, m)?)?;
    m.add_function(wrap_pyfunction!(concat_expr_py, m)?)?;
    m.add_function(wrap_pyfunction!(stack_expr_py, m)?)?;
    m.add_function(wrap_pyfunction!(eval_exprs, m)?)?;
    m.add_function(wrap_pyfunction!(eval_dicts, m)?)?;
    m.add_function(wrap_pyfunction!(where_py, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(datetime, m)?)?;
    m.add_function(wrap_pyfunction!(timedelta, m)?)?;
    m.add_function(wrap_pyfunction!(from_pandas, m)?)?;
    m.add_function(wrap_pyfunction!(get_newey_west_adjust_s, m)?)?;
    m.add_function(wrap_pyfunction!(parse_expr, m)?)?;
    m.add_function(wrap_pyfunction!(parse_expr_list, m)?)?;
    m.add_function(wrap_pyfunction!(context, m)?)?;
    #[cfg(feature = "arw")]
    m.add_function(wrap_pyfunction!(read_ipc, m)?)?;
    #[cfg(feature = "arw")]
    m.add_function(wrap_pyfunction!(scan_ipc, m)?)?;
    Ok(())
}
