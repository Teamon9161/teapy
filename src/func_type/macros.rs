/// Add a function of `func_type` to `PyArrayOk` so that we can directly call
/// the function, then define a `pyfunction` with the same name so that the
/// rust function can be used in Python.
macro_rules! impl_pyfunc {
    ($m:ident, $func:ident, $path:path, $otype:ty, $func_type:ident, $call_func:ident, -p ( $($p_name:ident: $p_type:ty),* $(,)? ), $(-sig $sig:expr)? $(,)?) => {
        // add the function to PyArrayOk
        impl<'py> PyArrayOk<'py> {
            pub fn $func(self, $($p_name: $p_type), *) -> &'py PyArrayDyn<$otype> {
                use PyArrayOk::*;
                use $path::*;
                match self {
                    F32(arr) => arr.$call_func::<$otype>(
                        $func::<f32> as $func_type<f32, $otype>,
                        $($p_name), *
                    ),
                    F64(arr) => arr.$call_func::<$otype>(
                        $func::<f64> as $func_type<f64, $otype>,
                        $($p_name), *
                    ),
                    I32(arr) => arr.$call_func::<$otype>(
                        $func::<i32> as $func_type<i32, $otype>,
                        $($p_name), *
                    ),
                    I64(arr) => arr.$call_func::<$otype>(
                        $func::<i64> as $func_type<i64, $otype>,
                        $($p_name), *
                    ),
                    _ => todo!()
                    // Usize(arr) => arr.$call_func::<$otype>(
                    //     $func::<usize> as $func_type<usize, $otype>,
                    //     $($p_name), *
                    // ),
                }
            }
        }
        // define a pyfunction
        #[pyfunction]
        fn $func<'py>(
            x: PyArrayOk<'py>,
            $($p_name: $p_type), *
        ) -> PyResult<&'py PyAny> {
            let out = x.$func($($p_name), *);
            return Ok(out.into())
        }
        $m.add_function(wrap_pyfunction!($func, $m)?)?;
    }
}

/// Add a function of `func_type` that accept two arrays to `PyArrayOk` so that we
/// can directly call the function, then define a `pyfunction` with the same name
/// so that the rust function can be used in Python.
macro_rules! impl_pyfunc2 {
    ($m:ident, $func:ident, $path:path, $otype:ty, $func_type:ident, $call_func:ident,
        -p ( $($p_name:ident: $p_type:ty),* $(,)? ), $(-sig $sig:expr)? $(,)?) => {
        // add the function to PyArrayOk
        impl<'py> PyArrayOk<'py> {
            pub fn $func(self, other: &PyArrayOk, $($p_name: $p_type), *) -> &'py PyArrayDyn<f64> {
                use PyArrayOk::*;
                use $path::*;
                match (self, other) {
                    (F32(arr), F32(other)) => arr.$call_func::<f32, $otype>(
                        other,
                        $func::<f32, f32> as $func_type<f32, f32, $otype>,
                        $($p_name), *
                    ),
                    (F64(arr), F64(other)) => arr.$call_func::<f64, $otype>(
                        other,
                        $func::<f64, f64> as $func_type<f64, f64, $otype>,
                        $($p_name), *
                    ),
                    (I32(arr), I32(other)) => arr.$call_func::<i32, $otype>(
                        other,
                        $func::<i32, i32> as $func_type<i32, i32, $otype>,
                        $($p_name), *
                    ),
                    (I64(arr), I64(other)) => arr.$call_func::<i64, $otype>(
                        other,
                        $func::<i64, i64> as $func_type<i64, i64, $otype>,
                        $($p_name), *
                    ),
                    (Usize(_arr), Usize(_other)) => todo!(),
                    // arr.$call_func::<usize, $otype>(
                    //     other,
                    //     $func::<usize, usize> as $func_type<usize, usize, $otype>,
                    //     $($p_name), *
                    // ),
                    _ => panic!("dtype of left array doesn't match dtype of right array"),
                }
            }
        }
        // define a pyfunction
        #[pyfunction]
        $(#[pyo3(text_signature=$sig)])?
        fn $func<'py>(
            x: PyArrayOk<'py>,
            y: PyArrayOk<'py>,
            $($p_name: $p_type), *
        ) -> PyResult<&'py PyAny> {
            let out = x.$func(&y, $($p_name), *);
            return Ok(out.into())
        }
        $m.add_function(wrap_pyfunction!($func, $m)?)?;
    }
}

/// Add a functype for `PyArrayDyn<T>` and create a macro_rule to create this type
/// of function, all types of function have two default arguments:
///
/// `&self`: `&PyArrayDyn<T>`
///
/// `f`: a function which has the func type defines in this macro
///
/// Once the function type is created, we can implement a concrete function of this
/// type by
/// `impl_macro_name!(m, func, otype)`
///
/// where m is &PyModule, func is the name of the concrete function and otype is
/// the dtype of the output array
#[allow(clippy::too_many_arguments)]
macro_rules! add_functype {
    ($impl_macro_name: ident, $func_path:path, $func_type:ident, $trait:ident, $call_func:ident,
    -func_p ( $($func_p_name:ident),* $(,)? ), -p ( $($p_name:ident: $p_type:ty),* $(,)? ), $(-sig $sig:expr)? $(,)?,
    ($self: ident, $f: ident), $body: tt) => {
        // define a type of func in rust
        pub type $func_type<T, U> = fn(ArrView1<T>, ArrViewMut1<U>, $($func_p_name),*);
        // define a trait so that PyArray can call functions of the same func type
        pub trait $trait <T: Number> {
            fn $call_func<U: Number> (
                &$self,
                f: $func_type<T, U>,
                $($p_name: $p_type), *
            )  -> &PyArrayDyn<U>;
        }
        // impl the trait defined above for PyArray
        impl<T: Number> $trait<T> for PyArrayDyn<T> {
            fn $call_func<U: Number>(
                &$self,
                $f: $func_type<T, U>,
                $($p_name: $p_type),*
            ) -> &PyArrayDyn<U> $body
        }
        // define a macro to create this type of function
        macro_rules! $impl_macro_name {
            ($$m:ident, $$func:ident, $$otype:ty) => {
                impl_pyfunc!(
                    $$m, $$func, $func_path, $$otype,
                    $func_type, $call_func,
                    -p ( $($p_name: $p_type),*),
                    $(-sig $sig)?
                );
            }
        }
    }
}

/// Add a functype for `PyArrayDyn<T>` and create a macro_rule to create this type
/// of function, all types of function have three default arguments:
///
/// `&self`: `&PyArrayDyn<T>`
///
/// `other`: `&PyArrayDyn<S>`
///
/// `f`: a function which has the func type defines in this macro
///
/// Once the function type is created, we can implement a concrete function of this
/// type by
/// `impl_macro_name!(m, func, otype)`
///
/// where m is &PyModule, func is the name of the concrete function and otype is
/// the dtype of the output array

macro_rules! add_functype2 {
    ($impl_macro_name: ident, $func_path:path, $func_type:ident, $trait:ident, $call_func:ident,
    -func_p ( $($func_p_name:ident),* $(,)? ), -p ( $($p_name:ident: $p_type:ty),* $(,)? ), $(-sig $sig:expr)? $(,)?,
    ($self: ident, $other: ident, $f: ident), $body: tt) => {
        // define a type of func in rust
        pub type $func_type<T, S, U> = fn(ArrView1<T>, ArrView1<S>, ArrViewMut1<U>, $($func_p_name),*);
        // define a trait so that PyArray can call functions of the same func type
        pub trait $trait <T: Number> {
            #![allow(clippy::too_many_arguments)]
            fn $call_func<S: Number, U: Number> (
                &$self,
                $other: &PyArrayDyn<S>,
                $f: $func_type<T, S, U>,
                $($p_name: $p_type), *
            )  -> &PyArrayDyn<U>;
        }
        // impl the trait defined above for PyArray
        impl<T: Number> $trait<T> for PyArrayDyn<T> {
            fn $call_func<S: Number, U: Number>(
                &$self,
                $other: &PyArrayDyn<S>,
                $f: $func_type<T, S, U>,
                $($p_name: $p_type),*
            ) -> &PyArrayDyn<U> $body
        }
        // define a macro to create this type of function
        macro_rules! $impl_macro_name {
            ($$m:ident, $$func:ident, $$otype:ty) => {
                impl_pyfunc2!(
                    $$m, $$func, $func_path, $$otype,
                    $func_type, $call_func,
                    -p ( $($p_name: $p_type),*),
                    $(-sig $sig)?
                );
            }
        }
    }
}

/// expose the function of ArrView1 to a new function.
macro_rules! auto_define {
    (   // array agg func
        agg
        $func:ident, $otype:ty, $($bound: ident,)?
        -a ( $($p_name:ident: $p_type:ty),* $(,)? ) $(,)?
    ) => {
        pub fn $func<T: Number $(+$bound)?>(arr: ArrView1<T>, mut out: ArrViewMut1<$otype>, $($p_name: $p_type),*)
        where
            usize: Number,
        {
            out.apply_mut(|v| *v = arr.$func($($p_name),*));
        }
    };

    (   // array agg func
        agg
        $func: ident, $otype:ty, $($bound: ident,)?
        -a ( $($p_name:ident: $p_type:ty),* $(,)? ),
        -c ( $($c_name:ident),* $(,)? ) $(,)?
    ) => {
        pub fn $func<T: Number $(+$bound)?>(arr: ArrView1<T>, mut out: ArrViewMut1<$otype>, $($p_name: $p_type),*)
        where
            usize: Number,
        {
            out.apply_mut(|v| *v = arr.$func($($c_name),*));
        }
    };

    (   // array agg func
        agg
        $func: ident, $otype:ty, $($bound: ident,)?
        -a ( $($p_name:ident: $p_type:ty),* $(,)? ),
        -noargs $(,)?
    ) => {
        pub fn $func<T: Number $(+$bound)?>(arr: ArrView1<T>, mut out: ArrViewMut1<$otype>, $($p_name: $p_type),*)
        where
            usize: Number,
        {
            out.apply_mut(|v| *v = arr.$func());
        }
    };

    (      // array func
        direct
        $func: ident, $otype:ty, $($bound: ident,)?
        -a ( $($p_name:ident: $p_type:ty),* $(,)? ),
        -noargs $(,)?
    ) => {
        pub fn $func<T: Number $(+$bound)?>(arr: ArrView1<T>, mut out: ArrViewMut1<$otype>, $($p_name: $p_type),*)
        where
            usize: Number,
        {
            arr.$func(&mut out);
        }
    };

    (   // array func
        direct
        $func: ident, $otype:ty, $($bound: ident,)?
        -a ( $($p_name:ident: $p_type:ty),* $(,)? ) $(,)?
    ) => {
        pub fn $func<T: Number $(+$bound)?>(arr: ArrView1<T>, mut out: ArrViewMut1<$otype>, $($p_name: $p_type),*)
        where
            usize: Number,
        {
            arr.$func(&mut out, $($p_name),*);
        }
    };

    (   // array func
        direct
        $func: ident, $otype:ty, $($bound: ident,)?
        -a ( $($p_name:ident: $p_type:ty),* $(,)? ),
        -c ( $($c_name:ident),* $(,)? ) $(,)?
    ) => {
        pub fn $func<T: Number $(+$bound)?>(arr: ArrView1<T>, mut out: ArrViewMut1<$otype>, $($p_name: $p_type),*)
        where
            usize: Number,
        {
            arr.$func(&mut out, $($c_name),*);
        }
    };
}
pub(crate) use auto_define;

/// expose the function of ArrView1 to a new function.
macro_rules! auto_define2 {
    (   // agg func
        agg
        $func:ident, $otype:ty,
        -a ( $($p_name:ident: $p_type:ty),* $(,)? ) $(,)?
    ) => {
        pub fn $func<T: Number, S:Number>(arr1: ArrView1<T>, arr2: ArrView1<S>, mut out: ArrViewMut1<$otype>, $($p_name: $p_type),*)
        where
            usize: Number,
        {
            out.apply_mut(|v| *v = arr1.$func(&arr2, $($p_name),*));
        }
    };

    (   // agg func
        agg
        $func: ident, $otype:ty,
        -a ( $($p_name:ident: $p_type:ty),* $(,)? ),
        -c ( $($c_name:ident),* $(,)? ) $(,)?
    ) => {
        pub fn $func<T: Number, S:Number>(arr1: ArrView1<T>, arr2: ArrView1<S>, mut out: ArrViewMut1<$otype>, $($p_name: $p_type),*)
        where
            usize: Number,
        {
            out.apply_mut(|v| *v = arr1.$func(&arr2, $($c_name),*));
        }
    };

    (   // agg func
        agg
        $func: ident, $otype:ty,
        -a ( $($p_name:ident: $p_type:ty),* $(,)? ),
        -noargs $(,)?
    ) => {
        pub fn $func<T: Number, S:Number>(arr1: ArrView1<T>, arr2: ArrView1<S>, mut out: ArrViewMut1<$otype>, $($p_name: $p_type),*)
        where
            usize: Number,
        {
            out.apply_mut(|v| *v = arr1.$func(&arr2));
        }
    };

    (   // array func
        direct
        $func: ident, $otype:ty,
        -a ( $($p_name:ident: $p_type:ty),* $(,)? ),
        -noargs $(,)?
    ) => {
        pub fn $func<T: Number, S:Number>(arr1: ArrView1<T>, arr2: ArrView1<S>, mut out: ArrViewMut1<$otype>, $($p_name: $p_type),*)
        where
            usize: Number,
        {
            arr1.$func(&arr2, &mut out);
        }
    };

    (   // array func
        direct
        $func: ident, $otype:ty,
        -a ( $($p_name:ident: $p_type:ty),* $(,)? ) $(,)?
    ) => {
        pub fn $func<T: Number, S:Number>(arr1: ArrView1<T>, arr2: ArrView1<S>, mut out: ArrViewMut1<$otype>, $($p_name: $p_type),*)
        where
            usize: Number,
        {
            arr1.$func(&arr2, &mut out, $($p_name),*);
        }
    };
}
pub(crate) use auto_define2;
