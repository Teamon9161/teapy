use crate::arr::{ArrBase, CorrMethod, FillMethod, QuantileMethod, WinsorizeMethod, WrapNdarray};
use crate::eager_macros::*;
use crate::from_py::PyArrayOk;
use numpy::{PyArray1, PyArrayDyn};
use pyo3::{pyfunction, types::PyModule, wrap_pyfunction, PyAny, PyResult};

pub(crate) fn add_eager(m: &PyModule) -> PyResult<()> {
    #[pyfunction]
    #[allow(clippy::needless_return)]
    fn remove_nan(x: PyArrayOk) -> PyResult<&PyAny> {
        match_pyarray!(numeric x, arr, {
            assert!(
                arr.ndim() == 1,
                "remove_nan can only be performed on a 1d array"
            );
            return Ok(PyArray1::from_owned_array(arr.py(), arr.readonly().as_array().wrap().to_dim1().unwrap().remove_nan_1d().0).into());
        });
    }
    m.add_function(wrap_pyfunction!(remove_nan, m)?)?;

    #[pyfunction]
    #[pyo3(name="std", signature=(x, stable=false, axis=0, par=false))]
    fn std_(x: PyArrayOk, stable: bool, axis: usize, par: bool) -> PyResult<&PyAny> {
        match_pyarray!(numeric x, arr, {
            let out = ArrBase::new(arr.readonly().as_array()).std(stable, axis, par).0;
            let out = PyArrayDyn::from_owned_array(arr.py(), out);
            Ok(out.into())
        })
    }
    m.add_function(wrap_pyfunction!(std_, m)?)?;

    // winsorize pyfunction
    impl_py_view_func!(m, winsorize,
        (
            method: WinsorizeMethod: WinsorizeMethod::Quantile,
            method_params: Option<f64>: None,
            stable: bool: false,
            axis: usize: 0,
            par: bool: false
        )
    );

    // winsorize pyfunction
    impl_py_inplace_func!(m, winsorize_inplace,
        (
            method: WinsorizeMethod: WinsorizeMethod::Quantile,
            method_params: Option<f64>: None,
            stable: bool: false,
            axis: usize: 0,
            par: bool: false
        )
    );
    impl_py_view_func!(m, clip, (min: f64, max: f64, axis: usize: 0, par: bool: false));
    impl_py_inplace_func!(m, clip_inplace, (min: f64, max: f64, axis: usize: 0, par: bool: false));
    impl_py_inplace_func!(m, zscore_inplace, (stable: bool: false, axis: usize: 0, par: bool: false));
    impl_py_view_func!(m, fillna, (method: FillMethod: FillMethod::Ffill, value: Option<f64>: None, axis: usize: 0, par: bool: false));
    impl_py_inplace_func!(m, fillna_inplace, (method: FillMethod: FillMethod::Ffill, value: Option<f64>: None, axis: usize: 0, par: bool: false));
    impl_py_view_func!(m, argsort, (rev: bool: false, axis: usize: 0, par: bool: false));
    // impl feature without arg `stable`
    impl_py_view_func!(
        m,
        [count_nan, count_notnan, median, min, max],
        (axis: usize: 0, par: bool: false)
    );
    // impl feature with arg `stable`
    impl_py_view_func!(
        m,
        [sum, mean, var, skew, kurt, zscore],
        (stable: bool: false, axis: usize: 0, par: bool: false)
    );
    // quantile pyfunction
    impl_py_view_func!(
        m, quantile,
        (q: f64, method: QuantileMethod: QuantileMethod::Linear, axis: usize: 0, par: bool: false)
    );
    // cov and corr pyfunction
    impl_py_view_func2!( m, cov, (stable: bool: false, axis: usize: 0, par: bool: false));
    impl_py_view_func2!( m, corr, (method: CorrMethod: CorrMethod::Pearson, stable: bool: false, axis: usize: 0, par: bool: false));
    // rank pyfunction
    impl_py_view_func!(
        m, split_group, (group: usize: 10, rev: bool: false, axis: usize: 0, par: bool: false)
    );
    // rank pyfunction
    impl_py_view_func!(
        m, rank, (pct: bool: false, rev: bool: false, axis: usize: 0, par: bool: false)
    );

    if cfg!(feature = "window_func") {
        // impl_py_view_func!(
        //     m, pct_change, (window: usize, axis: usize: "0", par: bool: false)
        // );
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
            (window: usize, min_periods: usize: 1, stable: bool: false, axis: usize: 0, par: bool: false)
        );

        // window functions with nostable_arg
        impl_py_view_func!(
            m,
            [
                ts_prod, ts_prod_mean, // window-feature
                ts_max, ts_min, ts_argmax, ts_argmin, ts_rank, ts_rank_pct, // window-compare
                ts_minmaxnorm, // window-norm
            ],
            (window: usize, min_periods: usize: 1, axis: usize: 0, par: bool: false)
        );

        // window-cov corr
        impl_py_view_func2!(
            m,
            [ts_cov, ts_corr],
            (window: usize, min_periods: usize: 1, stable: bool: false, axis: usize: 0, par: bool: false)
        );
    }

    Ok(())
}
