use ndarray::{Array1, Axis, Slice};
use pyo3::types::{PyList as PyList3, PyTuple};

use crate::arr::{ArbArray, DateTime, TimeDelta};

use super::super::from_py::{PyArrayOk, PyList};
use super::export::*;

#[pyfunction]
/// A util function to convert python object to PyExpr without copy
pub unsafe fn parse_expr_nocopy(obj: &PyAny) -> PyResult<PyExpr> {
    parse_expr(obj, false)
}

#[pyfunction]
#[pyo3(signature=(obj, copy=false))]
/// A util function to convert python object to PyExpr
///
/// copy: whether to copy numpy.ndarray when creating the PyExpr
pub unsafe fn parse_expr(obj: &PyAny, copy: bool) -> PyResult<PyExpr> {
    if let Ok(expr) = obj.extract::<PyExpr>() {
        Ok(expr)
    } else if obj.get_type().name()? == "PyExpr" {
        // For any crate that extends this crate
        let cell: &PyCell<PyExpr> = PyTryFrom::try_from_unchecked(obj);
        Ok(cell.try_borrow()?.clone())
    } else if obj.get_type().name()? == "DataFrame" {
        // cast pandas.DataFrame or polars DataFrame to PyExpr
        let module_name = obj.getattr("__module__")?.extract::<&str>()?;
        let module_name = module_name.split('.').next().unwrap();
        if module_name == "pandas" {
            let obj = obj.getattr("values")?;
            return parse_expr(obj, copy);
        } else if module_name == "polars" {
            let kwargs = PyDict::new(obj.py());
            kwargs.set_item("writable", false)?;
            let obj = obj.getattr("to_numpy")?.call((), Some(kwargs))?;
            return parse_expr(obj, copy);
        } else {
            return Err(PyValueError::new_err(format!(
                "DataFrame of module {module_name} is not supported"
            )));
        }
    } else if obj.get_type().name()? == "Series" {
        // cast pandas.Series to PyExpr
        let module_name = obj.getattr("__module__")?.extract::<&str>()?;
        let module_name = module_name.split('.').next().unwrap();
        if module_name == "pandas" {
            let obj = obj.getattr("values")?;
            return parse_expr(obj, copy);
        } else if module_name == "polars" {
            let kwargs = PyDict::new(obj.py());
            kwargs.set_item("writable", false)?;
            let obj = obj.getattr("to_numpy")?.call((), Some(kwargs))?;
            return parse_expr(obj, copy);
        } else {
            return Err(PyValueError::new_err(format!(
                "Series of module {module_name} is not supported"
            )));
        }
    } else if let Ok(pyarr) = obj.extract::<PyArrayOk>() {
        // cast numpy.ndarray to PyExpr
        if pyarr.is_object() {
            let arr = pyarr.into_object()?;
            if copy {
                return Ok(Expr::new_from_owned(arr.to_owned_array().wrap(), None).into());
            } else {
                let arr_res = arr.try_readwrite();
                if let Ok(mut arr) = arr_res {
                    let arr_write = arr.as_array_mut();
                    let arb_arr = ArbArray::ViewMut(arr_write.wrap());
                    // This is only safe when the pyarray exists
                    return Ok(std::mem::transmute::<Exprs<'_>, Exprs<'static>>(
                        Expr::new(arb_arr, None).into(),
                    )
                    .to_py(Some(vec![obj.to_object(obj.py())])));
                } else {
                    // not writable
                    let arr_read = arr.as_array();
                    let arb_arr = ArbArray::View(arr_read.wrap());
                    // This is only safe when the pyarray exists
                    return Ok(std::mem::transmute::<Exprs<'_>, Exprs<'static>>(
                        Expr::new(arb_arr, None).into(),
                    )
                    .to_py(Some(vec![obj.to_object(obj.py())])));
                };
            }
        } else if pyarr.is_datetime() {
            // we don't need to reference to the pyobject here because we made a copy
            use PyArrayOk::*;
            let out: PyExpr = match pyarr {
                DateTimeMs(arr) => Expr::new_from_owned(
                    arr.readonly()
                        .as_array()
                        .map(|v| Into::<DateTime>::into(*v))
                        .wrap(),
                    None,
                )
                .into(),
                DateTimeNs(arr) => Expr::new_from_owned(
                    arr.readonly()
                        .as_array()
                        .map(|v| Into::<DateTime>::into(*v))
                        .wrap(),
                    None,
                )
                .into(),
                DateTimeUs(arr) => Expr::new_from_owned(
                    arr.readonly()
                        .as_array()
                        .map(|v| Into::<DateTime>::into(*v))
                        .wrap(),
                    None,
                )
                .into(),
                _ => unreachable!(),
            };
            return Ok(out);
        }

        if copy {
            match_pyarray!(
                pyarr,
                arr,
                { Ok(Expr::new_from_owned(arr.to_owned_array().wrap(), None).into()) },
                F64,
                F32,
                I64,
                I32,
                Bool
            )
        } else {
            match_pyarray!(
                pyarr,
                arr,
                {
                    let arr_res = arr.try_readwrite();
                    if let Ok(mut arr) = arr_res {
                        let arr_write = arr.as_array_mut();
                        let arb_arr = ArbArray::ViewMut(arr_write.wrap());
                        // safe when pyarray exists
                        Ok(std::mem::transmute::<Exprs<'_>, Exprs<'static>>(
                            Expr::new(arb_arr, None).into(),
                        )
                        .to_py(Some(vec![obj.to_object(obj.py())])))
                    } else {
                        let arr_read = arr.as_array();
                        let arb_arr = ArbArray::View(arr_read.wrap());
                        // safe when pyarray exists
                        Ok(std::mem::transmute::<Exprs<'_>, Exprs<'static>>(
                            Expr::new(arb_arr, None).into(),
                        )
                        .to_py(Some(vec![obj.to_object(obj.py())])))
                    }
                },
                F64,
                F32,
                I64,
                I32,
                Bool
            )
        }
    } else if let Ok(pylist) = obj.extract::<PyList>() {
        match_pylist!(pylist, l, {
            Ok(Expr::new_from_owned(Arr1::from_vec(l).to_dimd().unwrap(), None).into())
        })
    } else if let Ok(val) = obj.extract::<i32>() {
        Ok(Expr::new(val.into(), None).into())
    } else if let Ok(val) = obj.extract::<f64>() {
        Ok(Expr::new(val.into(), None).into())
    } else if let Ok(val) = obj.extract::<String>() {
        Ok(Expr::new(val.into(), None).into())
    } else {
        Err(PyValueError::new_err(format!(
            "Not support this type of Pyobject {}",
            obj.get_type()
        )))
    }
}

#[pyfunction]
#[allow(clippy::missing_safety_doc)]
#[pyo3(signature=(obj, copy=false))]
pub unsafe fn parse_expr_list(obj: &PyAny, copy: bool) -> PyResult<Vec<PyExpr>> {
    if obj.is_instance_of::<PyList3>() || obj.is_instance_of::<PyTuple>() {
        if let Ok(seq) = obj.extract::<Vec<&PyAny>>() {
            Ok(seq
                .into_iter()
                .map(|obj| parse_expr(obj, copy).expect("Not support this type of Pyobject"))
                .collect_trusted())
        } else {
            unreachable!()
        }
    } else if let Ok(datadict) = obj.extract::<PyDataDict>() {
        Ok(datadict.into_data())
    } else if let Ok(pyexpr) = parse_expr(obj, copy) {
        Ok(vec![pyexpr])
    } else {
        Err(PyValueError::new_err(
            "Can't parse the Object to a vector of expr",
        ))
    }
}

#[pyfunction]
#[pyo3(signature=(exprs, axis=0))]
#[allow(unreachable_patterns)]
pub fn concat_expr(exprs: Vec<PyExpr>, axis: i32) -> PyResult<PyExpr> {
    let e1 = exprs.get(0).unwrap().clone();
    let obj_vec = exprs.iter().skip(1).map(|e| e.obj()).collect_trusted();
    macro_rules! concat_macro {
        ($($arm: ident => $cast_func: ident $(($arg: expr))?),*) => {
            match e1.inner() {
                $(Exprs::$arm(expr) => {
                    let other = exprs.into_iter().skip(1).map(|e| e.$cast_func($(($arg))?).unwrap()).collect_trusted();
                    expr.clone().concat(other, axis).to_py(e1.obj())
                }),*
                _ => unimplemented!("concat is not implemented for this type.")
            }
        };
    }
    let rtn = concat_macro!(
        F64 => cast_f64, F32 => cast_f32, I32 => cast_i32, I64 => cast_i64,
        Bool => cast_bool, Usize => cast_usize,
        Object => cast_object, String => cast_string, Str => cast_str,
        DateTime => cast_datetime(None), TimeDelta => cast_timedelta
    );
    Ok(rtn.add_obj_vec(obj_vec))
}

#[pyfunction]
#[allow(clippy::missing_safety_doc)]
#[pyo3(name="concat", signature=(exprs, axis=0))]
pub unsafe fn concat_expr_py(exprs: Vec<&PyAny>, axis: i32) -> PyResult<PyExpr> {
    let exprs = exprs
        .into_iter()
        .map(|e| parse_expr_nocopy(e))
        .collect::<PyResult<Vec<PyExpr>>>()?;
    concat_expr(exprs, axis)
}

#[pyfunction]
#[pyo3(signature=(exprs, axis=0))]
#[allow(unreachable_patterns)]
pub fn stack_expr(exprs: Vec<PyExpr>, axis: i32) -> PyResult<PyExpr> {
    let e1 = exprs.get(0).unwrap().clone();
    let obj_vec = exprs.iter().skip(1).map(|e| e.obj()).collect_trusted();
    macro_rules! stack_macro {
        ($($arm: ident => $cast_func: ident $(($arg: expr))?),*) => {
            match e1.inner() {
                $(Exprs::$arm(expr) => {
                    let other = exprs.into_iter().skip(1).map(|e| e.$cast_func($(($arg))?).unwrap()).collect_trusted();
                    expr.clone().stack(other, axis).to_py(e1.obj())
                }),*
                _ => unimplemented!("stack is not implemented for this type.")
            }
        };
    }
    let rtn = stack_macro!(
        F64 => cast_f64, F32 => cast_f32, I32 => cast_i32, I64 => cast_i64,
        Bool => cast_bool, Usize => cast_usize,
        Object => cast_object, String => cast_string, Str => cast_str,
        DateTime => cast_datetime(None), TimeDelta => cast_timedelta
    );
    Ok(rtn.add_obj_vec(obj_vec))
}

#[pyfunction]
#[allow(clippy::missing_safety_doc)]
#[pyo3(name="stack", signature=(exprs, axis=0))]
pub unsafe fn stack_expr_py(exprs: Vec<&PyAny>, axis: i32) -> PyResult<PyExpr> {
    let exprs = exprs
        .into_iter()
        .map(|e| parse_expr_nocopy(e))
        .collect::<PyResult<Vec<PyExpr>>>()?;
    stack_expr(exprs, axis)
}

#[pyfunction]
#[pyo3(signature=(exprs, inplace=false))]
pub fn eval(mut exprs: Vec<PyExpr>, inplace: bool) -> Option<Vec<PyExpr>> {
    exprs.par_iter_mut().for_each(|e| e.eval_inplace());
    if inplace {
        None
    } else {
        Some(exprs)
    }
}

#[pyfunction]
#[allow(clippy::missing_safety_doc)]
#[pyo3(name="where_", signature=(mask, expr, value, par=false))]
pub unsafe fn where_py(mask: &PyAny, expr: &PyAny, value: &PyAny, par: bool) -> PyResult<PyExpr> {
    let expr = parse_expr_nocopy(expr)?;
    let mask = parse_expr_nocopy(mask)?;
    let value = parse_expr_nocopy(value)?;
    where_(mask, expr, value, par)
}

pub fn where_(mask: PyExpr, expr: PyExpr, value: PyExpr, par: bool) -> PyResult<PyExpr> {
    let obj_vec = vec![expr.obj(), mask.obj(), value.obj()];
    let out: PyExpr = match (&expr.inner, &value.inner) {
        (Exprs::F64(_), _) | (_, Exprs::F64(_)) => expr
            .cast_f64()?
            .where_(mask.cast_bool()?, value.cast_f64()?, par)
            .into(),
        (Exprs::F32(_), _) | (_, Exprs::F32(_)) => expr
            .cast_f32()?
            .where_(mask.cast_bool()?, value.cast_f32()?, par)
            .into(),
        (Exprs::I64(_), _) | (_, Exprs::I64(_)) => expr
            .cast_i64()?
            .where_(mask.cast_bool()?, value.cast_i64()?, par)
            .into(),
        (Exprs::I32(_), _) | (_, Exprs::I32(_)) => expr
            .cast_i32()?
            .where_(mask.cast_bool()?, value.cast_i32()?, par)
            .into(),
        (Exprs::Usize(_), Exprs::Usize(_)) => expr
            .cast_usize()?
            .where_(mask.cast_bool()?, value.cast_usize()?, par)
            .into(),
        (Exprs::DateTime(_), Exprs::DateTime(_)) => expr
            .cast_datetime(None)?
            .where_(mask.cast_bool()?, value.cast_datetime(None)?, par)
            .into(),
        (Exprs::String(_), Exprs::String(_)) => expr
            .cast_string()?
            .where_(mask.cast_bool()?, value.cast_string()?, par)
            .into(),
        (Exprs::TimeDelta(_), Exprs::TimeDelta(_)) => expr
            .cast_timedelta()?
            .where_(mask.cast_bool()?, value.cast_timedelta()?, par)
            .into(),
        _ => todo!(),
    };
    Ok(out.add_obj_vec(obj_vec))
}

#[pyfunction]
#[allow(clippy::missing_safety_doc)]
#[allow(unreachable_patterns)]
pub unsafe fn full(shape: &PyAny, value: &PyAny, py: Python) -> PyResult<PyObject> {
    let shape = parse_expr_nocopy(shape)?;
    let value = parse_expr_nocopy(value)?;
    let obj = shape.obj();
    let out = match_exprs!(&value.inner, e, {
        Expr::full(shape.cast_usize()?, e.clone())
            .to_py(obj)
            .add_obj(value.obj())
    });
    Ok(out.into_py(py))
}

#[pyfunction]
#[pyo3(signature=(start, end=None, step=None))]
#[allow(unreachable_patterns, clippy::missing_safety_doc)]
pub unsafe fn arange(start: &PyAny, end: Option<&PyAny>, step: Option<f64>) -> PyResult<PyExpr> {
    let start = parse_expr_nocopy(start)?;
    let obj_start = start.obj();
    if let Some(end) = end {
        let end = parse_expr_nocopy(end)?;
        let obj_end = end.obj();
        Ok(Expr::arange(Some(start.cast_f64()?), end.cast_f64()?, step)
            .to_py(obj_start)
            .add_obj(obj_end))
    } else {
        // we only have start argument, this is actually end argument
        Ok(Expr::arange(None, start.cast_f64()?, step).to_py(obj_start))
    }
    // Ok(Expr::arange(start, end.cast_f64()?, step).to_py(obj))
}

#[pyfunction]
pub fn timedelta(rule: &str) -> PyExpr {
    let e: Expr<'static, TimeDelta> = TimeDelta::parse(rule).into();
    e.into()
}

#[pyfunction]
pub fn datetime(s: &str, fmt: &str) -> PyResult<PyExpr> {
    let e: Expr<'static, DateTime> = DateTime::parse(s, fmt)
        .map_err(PyValueError::new_err)?
        .into();
    Ok(e.into())
}

#[pyfunction]
pub fn get_newey_west_adjust_s(x: PyExpr, resid: PyExpr, lag: PyExpr) -> PyResult<PyExpr> {
    let obj_vec = [x.obj(), resid.obj(), lag.obj()];
    let lag = lag.cast_usize()?;
    let resid = resid.cast_f64()?;
    let mut out: PyExpr = x
        .cast_f64()?
        .chain_owned_f(move |x| {
            let lag = *lag.eval().view_arr().to_dim0().unwrap().0.into_scalar();
            let lag_f64 = lag as f64;
            let resid = resid.eval();
            let resid_view = resid.view_arr().to_dim1().unwrap().0;
            let weights = Array1::range(0., lag_f64 + 1., 1.)
                .mapv(|v| 1. - v / (lag_f64 + 1.))
                .wrap();
            let score = x.0 * resid_view.insert_axis(Axis(1));
            let mut s = score.t().wrap().dot(&score.view().wrap()).0;
            for lag in 1..=lag {
                let temp = score
                    .slice_axis(Axis(0), Slice::new(lag as isize, None, 1))
                    .t()
                    .wrap()
                    .dot(
                        &score
                            .slice_axis(Axis(0), Slice::new(0, Some(-(lag as isize)), 1))
                            .wrap(),
                    )
                    .0;
                s = s + *weights.get(lag).unwrap() * (temp.to_owned() + temp.t());
            }
            ArbArray::Owned(s.wrap())
        })
        .into();
    for obj in obj_vec {
        out = out.add_obj(obj)
    }
    Ok(out)
}

#[pyfunction]
pub fn from_pandas(df: &PyAny) -> PyResult<PyDataDict> {
    let columns = df.getattr("columns")?.extract::<Vec<String>>()?;
    let mut data = Vec::with_capacity(columns.len());
    for col in &columns {
        data.push(unsafe { parse_expr_nocopy(df.get_item(col)?.getattr("values")?)? });
    }
    Ok(PyDataDict::new(data, Some(columns)))
}
