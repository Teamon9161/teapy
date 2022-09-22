pub mod arr;
pub(crate) mod macros;
pub mod pyarray;

use crate::macros::*;
use crate::pyarray::PyArrayOk;
use numpy::{PyArray1, PyArrayDyn};
use pyo3::{pyfunction, pymodule, types::PyModule, wrap_pyfunction, PyAny, PyResult, Python};

use crate::arr::{ArrBase, WrapNdarray};

#[pymodule]
fn teapy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[allow(clippy::needless_return)]
    fn remove_nan(x: PyArrayOk) -> PyResult<&PyAny> {
        match_pyarray!(x, arr, {
            let x_r = arr.readonly();
            assert!(
                x_r.ndim() == 1,
                "remove_nan can only be performed on a 1d array"
            );
            let x_r = x_r.as_array().wrap().to_dim1();
            let out = PyArray1::from_owned_array(arr.py(), x_r.remove_nan_1d().0);
            return Ok(out.into());
        });
    }

    // winsorize pyfunction
    impl_py_view_func!(m, winsorize,
        (
            method: Option<&str>: "None",
            method_params: Option<f64>: "None",
            stable: bool: false,
            axis: usize: "0",
            par: bool: false
        )
    );
    // winsorize pyfunction
    impl_py_inplace_func!(m, winsorize_inplace,
        (
            method: Option<&str>: "None",
            method_params: Option<f64>: "None",
            stable: bool: false,
            axis: usize: "0",
            par: bool: false
        )
    );
    impl_py_view_func!(m, clip, (min: f64, max: f64, axis: usize: "0", par: bool: false));
    impl_py_inplace_func!(m, clip_inplace, (min: f64, max: f64, axis: usize: "0", par: bool: false));
    impl_py_inplace_func!(m, zscore_inplace, (stable: bool: false, axis: usize: "0", par: bool: false));
    impl_py_view_func!(m, fillna, (method: Option<&str>: "None", value: Option<f64>: "None", axis: usize: "0", par: bool: false));
    impl_py_inplace_func!(m, fillna_inplace, (method: Option<&str>: "None", value: Option<f64>: "None", axis: usize: "0", par: bool: false));
    // impl feature without arg `stable`
    impl_py_view_func!(
        m,
        [count_nan, count_notnan, median, min, max, argsort],
        (axis: usize: "0", par: bool: false)
    );
    // impl feature with arg `stable`
    impl_py_view_func!(
        m,
        [sum, mean, var, std, skew, kurt, zscore],
        (stable: bool: false, axis: usize: "0", par: bool: false)
    );
    // quantile pyfunction
    impl_py_view_func!(
        m, quantile,
        (q: f64, method: Option<&str>: "None", axis: usize: "0", par: bool: false)
    );
    // cov and corr pyfunction
    impl_py_view_func2!(
        m,
        [cov, corr],
        (stable: bool: false, axis: usize: "0", par: bool: false)
    );
    // rank pyfunction
    impl_py_view_func!(
        m, rank,
        (pct: bool: false, axis: usize: "0", par: bool: false)
    );
    // window functions with arg `stable` function
    impl_py_view_func!(
        m,
        [   // window-feature
            ts_sma, ts_sum, ts_wma, ts_ewm, ts_std, ts_var, ts_skew, ts_kurt,
            // window-norm
            ts_stable, ts_meanstdnorm,
            // window-reg
            ts_reg, ts_tsf, ts_reg_slope, ts_reg_intercept,
        ],
        (window: usize, min_periods: usize: "1", stable: bool: false, axis: usize: "0", par: bool: false)
    );

    // window functions with nostable_arg
    impl_py_view_func!(
        m,
        [
            ts_prod, ts_prod_mean, // window-feature
            ts_max, ts_min, ts_argmax, ts_argmin, ts_rank, ts_rank_pct, // window-compare
            ts_minmaxnorm, // window-norm
        ],
        (window: usize, min_periods: usize: "1", axis: usize: "0", par: bool: false)
    );

    // window-cov corr
    impl_py_view_func2!(
        m,
        [ts_cov, ts_corr],
        (window: usize, min_periods: usize: "1", stable: bool: false, axis: usize: "0", par: bool: false)
    );

    Ok(())
}
