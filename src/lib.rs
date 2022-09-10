#![feature(macro_metavar_expr)]
pub mod array_func;
pub mod datatype;
#[macro_use]
pub mod pyarray;

#[macro_use]
pub mod func_type;
pub mod algos;
// pub mod window_func;
pub mod arrview;
pub(crate) mod macros;
pub mod window_func;

use crate::pyarray::PyArrayOk;
use func_type::*;
use numpy::PyArrayDyn;
use pyo3::{pyfunction, pymodule, types::PyModule, wrap_pyfunction, PyAny, PyResult, Python};

#[pymodule]
fn teapy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // array function
    impl_py_array_func!(m, argsort, i32);
    impl_py_rank_func!(m, rank, f64);

    // array agg function
    impl_py_array_agg_func!(m, count_nan, usize);
    impl_py_array_agg_func!(m, count_notnan, usize);
    impl_py_array_agg_func!(m, sum, f64);
    impl_py_array_agg_func!(m, mean, f64);
    impl_py_array_agg_func!(m, min, f64);
    impl_py_array_agg_func!(m, max, f64);
    impl_py_array_agg_func!(m, var, f64);
    impl_py_array_agg_func!(m, std, f64);
    impl_py_array_agg_func!(m, skew, f64);
    impl_py_array_agg_func!(m, kurt, f64);

    // array agg function2s
    impl_py_array_agg_func2!(m, cov, f64);
    impl_py_array_agg_func2!(m, corr, f64);

    // feature
    impl_py_ts_func!(m, ts_sma, f64);
    impl_py_ts_func!(m, ts_ewm, f64);
    impl_py_ts_func!(m, ts_wma, f64);
    impl_py_ts_func!(m, ts_sum, f64);
    impl_py_ts_func!(m, ts_prod, f64);
    impl_py_ts_func!(m, ts_prod_mean, f64);
    impl_py_ts_func!(m, ts_std, f64);
    impl_py_ts_func!(m, ts_var, f64);
    impl_py_ts_func!(m, ts_skew, f64);
    impl_py_ts_func!(m, ts_kurt, f64);

    // compare
    impl_py_ts_func!(m, ts_max, f64);
    impl_py_ts_func!(m, ts_min, f64);
    impl_py_ts_func!(m, ts_argmax, f64);
    impl_py_ts_func!(m, ts_argmin, f64);
    impl_py_ts_func!(m, ts_rank, f64);
    impl_py_ts_func!(m, ts_rank_pct, f64);

    // norm
    impl_py_ts_func!(m, ts_stable, f64);
    impl_py_ts_func!(m, ts_minmaxnorm, f64);
    impl_py_ts_func!(m, ts_meanstdnorm, f64);

    // reg
    impl_py_ts_func!(m, ts_reg, f64);
    impl_py_ts_func!(m, ts_tsf, f64);
    impl_py_ts_func!(m, ts_reg_slope, f64);
    impl_py_ts_func!(m, ts_reg_intercept, f64);

    // corr cov
    impl_py_ts_func2!(m, ts_cov, f64);
    impl_py_ts_func2!(m, ts_corr, f64);
    Ok(())
}
