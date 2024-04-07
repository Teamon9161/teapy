use super::super::from_py::{PyArrayOk, PyList};
// use super::datadict::{IntoPyDataDict, PyDataDict, PyVecExprToRs};
use super::export::*;
use pyo3::types::{PyList as PyList3, PyTuple};
use tea_core::prelude::*;
use tea_lazy::{ColumnSelector, Data, Expr};

#[cfg(feature = "agg")]
use tea_ext::agg::{corr, CorrMethod};

#[cfg(feature = "arw")]
use crate::from_py::PyColSelect;
#[cfg(feature = "arw")]
use tea_core::prelude::StrError;
#[cfg(feature = "create")]
use tea_ext::create::*;
#[cfg(feature = "map")]
use tea_ext::map::*;
// #[cfg(feature = "io")]
// use tea_io::*;

#[pyfunction]
/// A util function to convert python object to PyExpr without copy
pub fn parse_expr_nocopy(obj: &PyAny) -> PyResult<PyExpr> {
    unsafe { parse_expr(obj, false) }
}

#[pyfunction]
#[pyo3(signature=(obj, copy=false))]
/// A util function to convert python object to PyExpr
///
/// copy: whether to copy numpy.ndarray when creating the PyExpr
pub unsafe fn parse_expr(obj: &PyAny, copy: bool) -> PyResult<PyExpr> {
    if let Ok(expr) = obj.extract::<PyExpr>() {
        Ok(expr)
    } else if obj.get_type().qualname()? == "Expr" {
        // For any crate that extends this crate
        let module_name = obj.getattr("__module__")?.extract::<&str>()?;
        if module_name == "teapy" {
            // let cell: &PyCell<PyExpr> = PyTryFrom::try_from_unchecked(obj);
            let cell: &PyCell<PyExpr> = obj.downcast_unchecked();
            Ok(cell.try_borrow()?.clone())
        } else {
            Err(PyValueError::new_err(format!(
                "Unknown Expr type from {module_name}"
            )))
        }
    } else if obj.get_type().qualname()? == "Selector" {
        let module_name = obj.getattr("__module__")?.extract::<&str>()?;
        let module_name = module_name.split('.').next().unwrap();
        if module_name == "teapy" {
            let kwargs = PyDict::new(obj.py());
            kwargs.set_item("context", true)?;
            let expr_obj = obj.call_method("to_expr", (), Some(kwargs))?;
            return parse_expr(expr_obj, copy);
        } else {
            Err(PyValueError::new_err(format!(
                "Unknown Selector type from {module_name}"
            )))
        }
    } else if obj.get_type().qualname()? == "DataFrame" {
        // cast pandas.DataFrame or polars DataFrame to PyExpr
        let module_name = obj.getattr("__module__")?.extract::<&str>()?;
        let module_name = module_name.split('.').next().unwrap();
        if module_name == "pandas" {
            let obj = obj.getattr("values")?;
            return parse_expr(obj, copy);
        } else if module_name == "polars" {
            let kwargs = PyDict::new(obj.py());
            kwargs.set_item("writable", false)?;
            let obj = obj.call_method("to_numpy", (), Some(kwargs))?;
            // let obj = obj.getattr("to_numpy")?.call((), Some(kwargs))?;
            return parse_expr(obj, copy);
        } else {
            return Err(PyValueError::new_err(format!(
                "DataFrame of module {module_name} is not supported"
            )));
        }
    } else if obj.get_type().qualname()? == "Series" {
        // cast pandas.Series to PyExpr
        let module_name = obj.getattr("__module__")?.extract::<&str>()?;
        let module_name = module_name.split('.').next().unwrap();
        if module_name == "pandas" {
            dbg!("parse pd Series");
            let obj = obj.getattr("values")?;
            return parse_expr(obj, copy);
        } else if module_name == "polars" {
            let kwargs = PyDict::new(obj.py());
            kwargs.set_item("writable", false)?;
            let dtype = obj.getattr("dtype")?.str()?.to_str()?;
            // let mut obj = obj.getattr("to_numpy")?.call((), Some(kwargs))?;
            let mut obj = obj.call_method("to_numpy", (), Some(kwargs))?;
            if dtype == "Utf8" {
                // obj = obj.getattr("astype")?.call1(("str",))?;
                obj = obj.call_method("astype", ("str",), None)?;
            }
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
                let e: Expr<'static> = arr.to_owned_array().wrap().into();
                return Ok(e.into());
            } else {
                let arr_res = arr.try_readwrite();
                if let Ok(mut arr) = arr_res {
                    let arr_write = arr.as_array_mut();
                    let arb_arr = ArbArray::ViewMut(arr_write.wrap());
                    // This is safe when the pyarray exists on the python side
                    // so we should keep a reference to the pyobject
                    return Ok(
                        std::mem::transmute::<Expr<'_>, Expr<'static>>(arb_arr.into())
                            .to_py(Some(vec![obj.to_object(obj.py())])),
                    );
                } else {
                    // not writable
                    let arr_read = arr.as_array();
                    let arb_arr = ArbArray::View(arr_read.wrap());
                    // This is only safe when the pyarray exists
                    return Ok(
                        std::mem::transmute::<Expr<'_>, Expr<'static>>(arb_arr.into())
                            .to_py(Some(vec![obj.to_object(obj.py())])),
                    );
                };
            }
        } else {
            #[cfg(feature = "time")]
            {
                if pyarr.is_datetime() {
                    // we don't need to reference to the pyobject here because we made a copy
                    use PyArrayOk::*;
                    {
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
                }
            }
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
                        Ok(
                            std::mem::transmute::<Expr<'_>, Expr<'static>>(Expr::new(
                                arb_arr, None,
                            ))
                            .to_py(Some(vec![obj.to_object(obj.py())])),
                        )
                    } else {
                        let arr_read = arr.as_array();
                        let arb_arr = ArbArray::View(arr_read.wrap());
                        // safe when pyarray exists
                        Ok(
                            std::mem::transmute::<Expr<'_>, Expr<'static>>(Expr::new(
                                arb_arr, None,
                            ))
                            .to_py(Some(vec![obj.to_object(obj.py())])),
                        )
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
            Ok(Expr::new_from_owned(Arr1::from_vec(l).to_dimd(), None).into())
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
    // } else if let Ok(datadict) = obj.extract::<PyDataDict>() {
    //     // let data = datadict.dd.data;
    //     let mut obj_map = datadict.obj_map;
    //     Ok(datadict
    //         .dd
    //         .data
    //         .into_iter()
    //         .map(|e| {
    //             if let Some(obj) = obj_map.remove(e.ref_name().unwrap()) {
    //                 e.to_py(obj)
    //             } else {
    //                 e.to_py(None)
    //             }
    //         })
    //         .collect())
    // } else if obj.hasattr("_dd")? && obj.get_type().name()? == "DataDict" {
    //     parse_expr_list(obj.getattr("_dd")?, copy)
    } else if obj.hasattr("exprs")? && obj.get_type().name()? == "DataDict" {
        parse_expr_list(obj.getattr("exprs")?, copy)
    } else if let Ok(pyexpr) = parse_expr(obj, copy) {
        Ok(vec![pyexpr])
    } else {
        Err(PyValueError::new_err(
            "Can't parse the Object to a vector of expr",
        ))
    }
}

#[cfg(all(feature = "concat", feature = "map"))]
#[pyfunction]
#[pyo3(signature=(exprs, axis=0))]
#[allow(unreachable_patterns)]
pub fn concat_expr(exprs: Vec<PyExpr>, axis: i32) -> PyResult<PyExpr> {
    let mut e1 = exprs.get(0).unwrap().clone();
    let obj_vec = exprs.iter().skip(1).map(|e| e.obj()).collect_trusted();
    e1.e.concat(
        exprs.into_iter().skip(1).map(|e| e.e).collect_trusted(),
        axis,
    );
    Ok(e1.add_obj_vec_into(obj_vec))
}

#[cfg(all(feature = "concat", feature = "map"))]
#[pyfunction]
#[allow(clippy::missing_safety_doc)]
#[pyo3(name="concat", signature=(exprs, axis=0))]
pub unsafe fn concat_expr_py(exprs: Vec<&PyAny>, axis: i32) -> PyResult<PyExpr> {
    let exprs = exprs
        .into_iter()
        .map(parse_expr_nocopy)
        .collect::<PyResult<Vec<PyExpr>>>()?;
    concat_expr(exprs, axis)
}

#[cfg(all(feature = "concat", feature = "map"))]
#[pyfunction]
#[pyo3(signature=(exprs, axis=0))]
#[allow(unreachable_patterns)]
pub fn stack_expr(exprs: Vec<PyExpr>, axis: i32) -> PyResult<PyExpr> {
    let mut e1 = exprs.get(0).unwrap().clone();
    let obj_vec = exprs.iter().skip(1).map(|e| e.obj()).collect_trusted();
    e1.e.stack(
        exprs.into_iter().skip(1).map(|e| e.e).collect_trusted(),
        axis,
    );
    Ok(e1.add_obj_vec_into(obj_vec))
}

#[cfg(all(feature = "concat", feature = "map"))]
#[pyfunction]
#[allow(clippy::missing_safety_doc)]
#[pyo3(name="stack", signature=(exprs, axis=0))]
pub unsafe fn stack_expr_py(exprs: Vec<&PyAny>, axis: i32) -> PyResult<PyExpr> {
    let exprs = exprs
        .into_iter()
        .map(|e| parse_expr_nocopy(e).expect("can not parse thid type into Expr"))
        .collect::<Vec<PyExpr>>();
    stack_expr(exprs, axis)
}

#[cfg(feature = "agg")]
#[pyfunction]
#[pyo3(name="corr", signature=(exprs, method=CorrMethod::Pearson, min_periods=1, stable=false))]
pub fn corr_py(
    exprs: Vec<&PyAny>,
    method: CorrMethod,
    min_periods: usize,
    stable: bool,
) -> PyResult<PyExpr> {
    let exprs = exprs
        .into_iter()
        .map(|e| parse_expr_nocopy(e).expect("can not parse thid type into Expr"))
        .collect::<Vec<PyExpr>>();
    let obj_vec = exprs.iter().map(|e| e.obj()).collect_trusted();
    let exprs = exprs.into_iter().map(|e| e.e).collect_trusted();
    let out: PyExpr = corr(exprs, method, min_periods, stable).into();
    Ok(out.add_obj_vec_into(obj_vec))
}

#[pyfunction]
#[pyo3(signature=(exprs, inplace=false, freeze=true))]
pub fn eval_exprs(
    mut exprs: Vec<PyExpr>,
    inplace: bool,
    freeze: bool,
) -> PyResult<Option<Vec<PyExpr>>> {
    exprs
        .par_iter_mut()
        .try_for_each(|e| e.eval_inplace(None, freeze))?;
    if inplace {
        Ok(None)
    } else {
        Ok(Some(exprs))
    }
}

// #[pyfunction]
// #[pyo3(signature=(dds, inplace=true, context=false))]
// pub fn eval_dicts(
//     dds: Vec<&PyAny>,
//     inplace: bool,
//     context: bool,
// ) -> PyResult<Option<Vec<PyDataDict>>> {
//     let mut dds = dds
//         .into_iter()
//         .map(|dd| {
//             if dd.hasattr("_dd").unwrap() && dd.get_type().name().unwrap() == "DataDict" {
//                 dd.getattr("_dd").unwrap().extract::<PyDataDict>().unwrap()
//             } else {
//                 dd.extract::<PyDataDict>().unwrap()
//             }
//         })
//         .collect_trusted();
//     dds.par_iter_mut().try_for_each(|dd| dd.eval_all(context))?;
//     if inplace {
//         Ok(None)
//     } else {
//         Ok(Some(dds))
//     }
// }

#[cfg(feature = "map")]
#[pyfunction]
#[pyo3(name="where_", signature=(mask, expr, value, par=false))]
#[allow(clippy::missing_safety_doc, clippy::redundant_clone)]
pub unsafe fn where_py(mask: &PyAny, expr: &PyAny, value: &PyAny, par: bool) -> PyResult<PyExpr> {
    let expr = parse_expr_nocopy(expr)?;
    let mask = parse_expr_nocopy(mask)?;
    let value = parse_expr_nocopy(value)?;
    where_(expr.clone(), mask, value, par)
}

#[cfg(feature = "map")]
pub fn where_(mut expr: PyExpr, mask: PyExpr, value: PyExpr, par: bool) -> PyResult<PyExpr> {
    let obj_vec = vec![mask.obj(), value.obj()];
    expr.e.where_(mask.e, value.e, par);
    Ok(expr.add_obj_vec_into(obj_vec))
}

#[cfg(feature = "create")]
#[pyfunction]
#[allow(clippy::missing_safety_doc)]
#[allow(unreachable_patterns)]
pub unsafe fn full(shape: &PyAny, value: &PyAny) -> PyResult<PyExpr> {
    let shape = parse_expr_nocopy(shape)?;
    let value = parse_expr_nocopy(value)?;
    let (obj, obj2) = (shape.obj(), value.obj());
    let out = Expr::full(&shape.e, value.e).to_py(obj);
    Ok(out.add_obj_into(obj2))
}

#[cfg(feature = "create")]
#[pyfunction]
#[pyo3(signature=(start, end=None, step=None))]
#[allow(unreachable_patterns, clippy::missing_safety_doc)]
pub unsafe fn arange(start: &PyAny, end: Option<&PyAny>, step: Option<f64>) -> PyResult<PyExpr> {
    let start = parse_expr_nocopy(start)?;
    let step = step.map(|s| s.into());
    let obj_start = start.obj();
    if let Some(end) = end {
        let end = parse_expr_nocopy(end)?;
        let obj_end = end.obj();
        Ok(Expr::arange(Some(start.e), &end.e, step)
            .to_py(obj_start)
            .add_obj_into(obj_end))
    } else {
        // we only have start argument, this is actually end argument
        Ok(Expr::arange(None, &start.e, step).to_py(obj_start))
    }
}

#[cfg(feature = "time")]
#[pyfunction]
pub fn timedelta(rule: &str) -> PyExpr {
    let e: Expr<'static> = TimeDelta::parse(rule).into();
    e.into()
}

#[cfg(feature = "time")]
#[pyfunction]
pub fn datetime(s: &str, fmt: &str) -> PyResult<PyExpr> {
    let e: Expr<'static> = DateTime::parse(s, fmt)
        .map_err(PyValueError::new_err)?
        .into();
    Ok(e.into())
}

#[pyfunction]
#[cfg(feature = "blas")]
#[allow(clippy::redundant_clone)]
pub fn get_newey_west_adjust_s(x: PyExpr, resid: PyExpr, lag: PyExpr) -> PyResult<PyExpr> {
    let obj_vec = [x.obj(), resid.obj(), lag.obj()];
    let mut out = x.clone();
    out.e.get_newey_west_adjust_s(resid.e, lag.e);
    for obj in obj_vec {
        out.add_obj(obj);
    }
    Ok(out)
}

// #[pyfunction]
// pub fn from_dataframe(df: &PyAny) -> PyResult<PyDataDict> {
//     let columns = df.getattr("columns")?.extract::<Vec<String>>()?;
//     let mut data = Vec::with_capacity(columns.len());
//     for col in &columns {
//         data.push(parse_expr_nocopy(df.get_item(col)?)?);
//     }
//     let (data, obj_map) = data.into_rs(Some(columns.clone()))?;
//     Ok(DataDict::new(data, Some(columns)).to_py(obj_map))
// }

#[pyfunction]
pub fn context<'py>(s: Option<&'py PyAny>) -> PyResult<PyExpr> {
    let s: ColumnSelector<'py> = s.into();
    let name = s.name();
    let a: Data = s.into();
    // safety: this function is using in python
    let mut e = unsafe { std::mem::transmute::<Expr<'_>, Expr<'static>>(a.into()) };
    e.set_name(name);
    Ok(e.to_py(None))
}

// #[cfg(all(feature = "arw", feature = "io"))]
// #[pyfunction]
// pub fn read_ipc(path: &str, columns: PyColSelect) -> PyResult<PyDataDict> {
//     let dd = DataDict::read_ipc(path, columns.0).map_err(StrError::to_py)?;
//     Ok(PyDataDict {
//         dd,
//         obj_map: Default::default(),
//     })
// }

#[cfg(all(feature = "arw", feature = "io"))]
#[pyfunction]
pub fn scan_ipc(path: String, columns: PyColSelect) -> PyResult<Vec<PyExpr>> {
    use tea_io::scan_ipc_lazy;
    let out: Vec<PyExpr> = scan_ipc_lazy(path, columns.0)
        .map_err(StrError::to_py)?
        .into_iter()
        .map(|e| e.into())
        .collect();
    Ok(out)
    // let dd = DataDict::scan_ipc(path, columns.0).map_err(StrError::to_py)?;
    // Ok(PyDataDict {
    //     dd,
    //     obj_map: Default::default(),
    // })
}
