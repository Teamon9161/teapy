// pub mod ffi;
pub mod datatype;
pub mod pyarray;
pub mod window_func;
pub mod array;

use pyo3::{
    pymodule, pyfunction, wrap_pyfunction, 
    types::PyModule,
    PyAny, PyResult, Python,
};

use crate::pyarray::PyArrayOk;

#[pymodule]
fn teapy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    macro_rules! impl_pyarrayfunc {
        ($func:ident) => {
            #[pyfunction]
            fn $func<'py> ( // 滚动移动平均
                x: PyArrayOk<'py>,
                axis: Option<usize>,
                par: Option<bool>,
            ) -> PyResult<&'py PyAny> {
                let out = x.$func(axis, par);
                return Ok(out.into())
            }
        m.add_function(wrap_pyfunction!($func, m)?)?;
        };
    }
    

    macro_rules! impl_pytsfunc {
        ($func:ident) => {
            #[pyfunction]
            fn $func<'py> ( // 滚动移动平均
                x: PyArrayOk<'py>,
                window: usize,
                axis: Option<usize>,
                min_periods: Option<usize>,
                par: Option<bool>,
            ) -> PyResult<&'py PyAny> {
                let out = x.$func(window, axis, min_periods, par);
                return Ok(out.into())
            }
        m.add_function(wrap_pyfunction!($func, m)?)?;
        };
    }

    // 接收两个array的ts_func2，用于协方差cov和相关系数corr
    macro_rules! impl_pytsfunc2 {
        ($func:ident) => {
            #[pyfunction]
            fn $func<'py> ( // 滚动移动平均
                x: PyArrayOk<'py>,
                y: PyArrayOk<'py>,
                window: usize,
                axis: Option<usize>,
                min_periods: Option<usize>,
                par: Option<bool>,
            ) -> PyResult<&'py PyAny> {
                let out = x.$func(&y, window, axis, min_periods, par);
                return Ok(out.into())
            }
            m.add_function(wrap_pyfunction!($func, m)?)?;
        };
    }

    // array function
    impl_pyarrayfunc!(argsort);
    impl_pyarrayfunc!(rank);
    impl_pyarrayfunc!(rank_pct);
    
    // feature
    impl_pytsfunc!(ts_sma);
    impl_pytsfunc!(ts_ewm);
    impl_pytsfunc!(ts_wma);
    impl_pytsfunc!(ts_sum);
    impl_pytsfunc!(ts_prod);
    impl_pytsfunc!(ts_prod_mean);
    impl_pytsfunc!(ts_std);
    impl_pytsfunc!(ts_skew);
    impl_pytsfunc!(ts_kurt);

    // compare
    impl_pytsfunc!(ts_max);
    impl_pytsfunc!(ts_min);
    impl_pytsfunc!(ts_argmax);
    impl_pytsfunc!(ts_argmin);
    impl_pytsfunc!(ts_rank);
    impl_pytsfunc!(ts_rank_pct);

    // norm
    impl_pytsfunc!(ts_stable);
    impl_pytsfunc!(ts_minmaxnorm);
    impl_pytsfunc!(ts_meanstdnorm);

    // corr cov
    impl_pytsfunc2!(ts_cov);
    impl_pytsfunc2!(ts_corr);

    // reg
    impl_pytsfunc!(ts_reg);
    impl_pytsfunc!(ts_tsf);
    impl_pytsfunc!(ts_reg_slope);
    impl_pytsfunc!(ts_reg_intercept);

    Ok(())
}


