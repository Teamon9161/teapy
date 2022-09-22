/// match the enum `PyArrayOk` to get the discrete dtype of `PyArray` so that we can
/// call functions on a `PyArray` of which dtype is known;
macro_rules! match_pyarray {
    ($array: ident, $arr: ident, $body: tt) => {
        use PyArrayOk::*;
        match $array {
            F32($arr) => $body,
            F64($arr) => $body,
            I32($arr) => $body,
            I64($arr) => $body,
            _ => todo!(),
        }
    };
}
pub(crate) use match_pyarray;

/// match the enum `PyArrayOk` to get the discrete dtype of two `PyArray` so that we can
/// call functions on two `PyArray` of which dtype is known;
macro_rules! match_pyarray2 {
    ($array1: ident, $array2: ident, $arr1: ident, $arr2:ident, $body: tt) => {
        use PyArrayOk::*;
        match ($array1, $array2) {
            (F32($arr1), F32($arr2)) => $body,
            (F64($arr1), F64($arr2)) => $body,
            (I32($arr1), I32($arr2)) => $body,
            (I64($arr1), I64($arr2)) => $body,
            _ => todo!(),
        }
    };
}
pub(crate) use match_pyarray2;

/// impl a python function for `PyArrayOk`, get a readonly view of `PyArray` and call a
/// function of the same name on `ArrBase`.
macro_rules! impl_py_view_func {
    ($m: ident, $func: ident, ($($p:ident: $p_ty:ty $(:$p_default: expr)?),*)) => {
        #[pyfunction(x $($(,$p = $p_default)?)*)]
        fn $func<'py>(x: PyArrayOk<'py> $(, $p: $p_ty)*) -> PyResult<&'py PyAny> {
            match_pyarray!(x, arr, {
                let x = arr.readonly();
                let x_r = x.as_array();
                let out = ArrBase::new(x_r).$func($($p),*).0;
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
                let (x, y) = (arr1.readonly(), arr2.readonly());
                let (x_r, y_r) = (x.as_array(), y.as_array());
                let out = x_r.wrap().$func(&y_r.into(), $($p),*).0;
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
            match_pyarray!(x, arr, {
                let mut x = arr.readwrite();
                let x_r = x.as_array_mut();
                x_r.wrap().$func($($p),*);
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
