/// impl a python function for `PyArrayOk`, get a readonly view of `PyArray` and call a
/// function of the same name on `ArrBase`.
macro_rules! impl_py_view_func {
    ($m: ident, $func: ident, ($($p:ident: $p_ty:ty $(:$p_default: expr)?),*)) => {
        #[pyfunction(x $($(,$p = $p_default)?)*)]
        fn $func<'py>(x: PyArrayOk<'py> $(, $p: $p_ty)*) -> PyResult<&'py PyAny> {
            match_pyarray!(numeric x, arr, {
                let out = ArrBase::new(arr.readonly().as_array()).$func($($p),*).0;
                let out = PyArrayDyn::from_owned_array(arr.py(), out);
                return Ok(out.into())
            });
        }
        $m.add_function(wrap_pyfunction!($func, $m)?)?;
    };
    ($m: ident, [$($func: ident),* $(,)?], $other: tt) => {
        $(impl_py_view_func!($m, $func, $other);)*
    };
}
pub(crate) use impl_py_view_func;

/// impl a python function for two `PyArrayOk`, get readonly views of the two `PyArray` and call a
/// function of the same name on `ArrBase`.
macro_rules! impl_py_view_func2 {
    ($m: ident, $func: ident, ($($p:ident: $p_ty:ty $(:$p_default: expr)?),*)) => {
        #[pyfunction(x, y $($(,$p = $p_default)?)*)]
        fn $func<'py>(x: PyArrayOk<'py>, y: PyArrayOk $(, $p: $p_ty)*) -> PyResult<&'py PyAny> {
            match_pyarray2!(x, y, arr1, arr2, {
                let out = arr1.readonly().as_array().wrap().$func(&arr2.readonly().as_array().wrap(), $($p),*).0;
                let out = PyArrayDyn::from_owned_array(arr1.py(), out);
                return Ok(out.into())
            });
        }
        $m.add_function(wrap_pyfunction!($func, $m)?)?;
    };
    ($m: ident, [$($func: ident),* $(,)?], $other: tt) => {
        $(impl_py_view_func2!($m, $func, $other);)*
    };
}
pub(crate) use impl_py_view_func2;

/// impl a python inplace function for `PyArrayOk`, get a readwrite view of `PyArray` and call a
/// inplace function of the same name on `ArrBase`.
macro_rules! impl_py_inplace_func {
    ($m: ident, $func: ident, ($($p:ident: $p_ty:ty $(:$p_default: expr)?),*)) => {
        #[pyfunction(x $($(,$p = $p_default)?)*)]
        fn $func<'py>(x: PyArrayOk<'py> $(, $p: $p_ty)*) -> PyResult<()> {
            match_pyarray!(numeric x, arr, {
                arr.readwrite().as_array_mut().wrap().$func($($p),*);
                return Ok(());
            });
        }
        $m.add_function(wrap_pyfunction!($func, $m)?)?;
    };
    ($m: ident, [$($func: ident),* $(,)?], $other: tt) => {
        $(impl_py_inplace_func!($m, $func, $other);)*
    };
}
pub(crate) use impl_py_inplace_func;
