use ahash::HashMap;
use pyo3::{PyTraverseError, PyVisit};
use std::borrow::Cow;
use std::fmt::Debug;
use std::ops::{Deref, DerefMut};

use super::export::*;
use super::pyexpr::RefObj;
use tea_core::prelude::StrError;
use tea_core::utils::CollectTrustedToVec;
use tea_lazy::{ColumnSelector, DataDict, Expr, GetOutput};

#[cfg(feature = "agg")]
use tea_ext::agg::*;

#[pyclass]
#[derive(Clone, Default)]
pub struct PyDataDict {
    pub dd: DataDict<'static>,
    pub obj_map: HashMap<String, RefObj>,
}

impl DerefMut for PyDataDict {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.dd
    }
}

impl Deref for PyDataDict {
    type Target = DataDict<'static>;

    fn deref(&self) -> &Self::Target {
        &self.dd
    }
}

impl Debug for PyDataDict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.dd.fmt(f)
    }
}

pub trait IntoPyDataDict {
    fn to_py(self, obj_map: HashMap<String, RefObj>) -> PyDataDict;
}

impl IntoPyDataDict for DataDict<'static> {
    fn to_py(self, obj_map: HashMap<String, RefObj>) -> PyDataDict {
        PyDataDict { dd: self, obj_map }
    }
}

pub trait PyVecExprToRs {
    fn into_rs(
        self,
        names: Option<Vec<String>>,
    ) -> PyResult<(Vec<Expr<'static>>, HashMap<String, RefObj>)>;
}

impl PyVecExprToRs for Vec<PyExpr> {
    fn into_rs(
        self,
        names: Option<Vec<String>>,
    ) -> PyResult<(Vec<Expr<'static>>, HashMap<String, RefObj>)> {
        let obj_map = if let Some(names) = names {
            if names.len() != self.len() {
                return Err(StrError(
                    "The length of names are not equal to the number of expressions".into(),
                )
                .to_py());
            }
            zip(&self, names).map(|(e, name)| (name, e.obj())).collect()
        } else {
            let mut name_auto = -1;
            self.iter()
                .map(|e| {
                    (
                        e.e.name().unwrap_or_else(|| {
                            name_auto += 1;
                            format!("column_{name_auto}")
                        }),
                        e.obj(),
                    )
                })
                .collect()
        };
        let expr_rs = self.into_iter().map(|e| e.e).collect_trusted();
        Ok((expr_rs, obj_map))
    }
}

impl PyDataDict {
    pub fn remove_ref_obj(&mut self, select: Vec<String>) {
        select.into_iter().for_each(|e| {
            self.obj_map.remove(&e);
        });
    }

    pub fn update_ref_obj_name(&mut self, old_name: &str, new_name: &str) {
        if let Some(obj) = self.obj_map.remove(old_name) {
            self.obj_map.insert(new_name.to_string(), obj);
        }
    }

    pub fn add_ref_obj(&mut self, name: &str, obj: RefObj) -> Option<RefObj> {
        self.obj_map.insert(name.to_string(), obj)
    }

    pub fn add_obj_map(&mut self, obj_map: HashMap<String, RefObj>) {
        self.obj_map.extend(obj_map);
    }

    /// Remove the reference of if the Expression owns the data
    pub fn organize_obj_map(&mut self) {
        let cols_to_remove = self
            .dd
            .data
            .iter()
            .filter(|e| e.is_owned())
            .map(|e| e.name().unwrap())
            .collect_trusted();
        self.remove_ref_obj(cols_to_remove);
    }

    pub fn map_expr_to_py(&self, e: Expr<'static>) -> PyExpr {
        let name = &e.name().unwrap();
        e.to_py(self.obj_map.get(name).cloned().unwrap_or(None))
    }

    pub fn get_selector_out_name(&self, cs: ColumnSelector) -> Vec<String> {
        self.dd.get_selector_out_name(cs)
    }
}

/// Be careful when implement a new method as we should also deal with the reference of the object.
#[pymethods]
#[allow(clippy::missing_safety_doc)]
impl PyDataDict {
    #[new]
    #[pyo3(signature=(data, columns=None, copy=false))]
    pub fn init(data: &PyAny, columns: Option<Vec<String>>, copy: bool) -> PyResult<Self> {
        let data = unsafe { parse_expr_list(data, copy)? };
        let (data, obj_map) = data.into_rs(columns.clone())?;
        Ok(DataDict::new(data, columns).to_py(obj_map))
    }

    #[pyo3(name = "is_empty")]
    pub fn is_empty_py(&self) -> bool {
        self.is_empty()
    }

    #[allow(clippy::wrong_self_convention)]
    #[pyo3(signature=(context=false))]
    pub fn to_pd(&mut self, context: bool) -> PyResult<PyObject> {
        self.eval_all(context)?;
        Python::with_gil(|py| {
            let pd = PyModule::import(py, "pandas").unwrap(); //.to_object(py)
            Ok(pd
                .getattr("DataFrame")?
                .call1((self.to_dict(context, py)?,))?
                .into())
        })
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if self.obj_map.is_empty() {
            return Ok(());
        }
        self.obj_map
            .values()
            .flatten()
            .flatten()
            .try_for_each(|e| visit.call(e))
    }

    fn __clear__(&mut self) {
        self.obj_map = Default::default();
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(method=CorrMethod::Pearson, cols=None, min_periods=3, stable=false))]
    pub fn corr(
        &self,
        method: CorrMethod,
        cols: Option<&PyAny>,
        min_periods: usize,
        stable: bool,
    ) -> PyExpr {
        let selector = ColumnSelector::from(cols);
        let out = self.dd.corr(Some(selector), method, min_periods, stable);
        out.to_py(None)
            .add_obj_vec_into(self.obj_map.values().cloned().collect())
    }

    #[pyo3(signature=(columns))]
    pub fn drop(&mut self, columns: &PyAny) -> PyResult<()> {
        let selector: ColumnSelector = columns.into();
        let drop_cols = self.drop_inplace(selector).map_err(StrError::to_py)?;
        self.remove_ref_obj(drop_cols);
        Ok(())
    }

    pub fn __repr__(&self) -> Cow<'_, str> {
        Cow::from(format!("{self:#?}"))
    }

    #[getter]
    pub fn get_dtypes<'py>(&'py self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let res = PyDict::new(py);
        for e in &self.data {
            res.set_item(e.name().unwrap(), e.dtype())?;
        }
        Ok(res)
    }

    #[allow(clippy::wrong_self_convention)]
    #[pyo3(signature=(context=false))]
    pub fn to_dict<'py>(&'py mut self, context: bool, py: Python<'py>) -> PyResult<&'py PyDict> {
        self.eval_all(context)?;
        let dict = PyDict::new(py);
        for expr in &self.data {
            dict.set_item(expr.name(), expr.clone().to_py(None).value(None, None, py)?)?;
        }
        Ok(dict)
    }

    pub fn __len__(&self) -> usize {
        self.len()
    }

    pub fn copy(&self) -> Self {
        PyDataDict {
            dd: self.dd.clone(),
            obj_map: self.obj_map.clone(),
        }
    }

    #[getter]
    pub fn get_len(&self) -> usize {
        self.len()
    }

    #[getter]
    pub fn get_columns(&self) -> Vec<String> {
        self.dd.columns_owned()
    }

    #[getter]
    fn get_map(&self) -> HashMap<String, usize> {
        (*self.map).clone().into()
    }

    #[getter]
    pub fn get_raw_data(&self) -> Vec<PyExpr> {
        // out data must keep the reference
        self.data
            .clone()
            .into_iter()
            .map(|e| self.map_expr_to_py(e))
            .collect_trusted()
    }

    #[getter]
    pub fn get_obj_map(&self) -> HashMap<String, RefObj> {
        self.obj_map.clone()
    }

    #[setter]
    pub fn set_columns(&mut self, columns: Vec<String>) {
        // Be careful when alter the name of the map
        let ori_columns = self.dd.columns_owned();
        let new_columns = columns.iter().map(|e| e.as_str()).collect_trusted();
        for (ori, new) in zip(&ori_columns, new_columns) {
            self.update_ref_obj_name(ori, new);
        }
        self.dd.set_columns(columns)
    }

    #[pyo3(signature=(context=false))]
    pub fn eval_all(&mut self, context: bool) -> PyResult<()> {
        self.eval(None, context)
    }

    #[pyo3(signature=(cols=None, context=false))]
    pub fn eval(&mut self, cols: Option<&PyAny>, context: bool) -> PyResult<()> {
        let cs = ColumnSelector::from(cols);
        let ori_name = self.get_selector_out_name(cs.clone());
        let eval_idx = ori_name
            .iter()
            .map(|e| {
                *self
                    .dd
                    .map
                    .get(e)
                    .unwrap_or_else(|| panic!("Can not find the column: {:?}", e))
            })
            .collect_trusted();
        self.dd.eval_inplace(cs, context).map_err(StrError::to_py)?;

        let new_name = eval_idx
            .into_iter()
            .map(|idx| self.dd.data.get(idx).unwrap().name().unwrap())
            .collect_trusted();
        for (ori, new) in zip(&ori_name, &new_name) {
            if self
                .dd
                .get(new.as_str().into())
                .map_err(StrError::to_py)?
                .into_expr()
                .unwrap()
                .is_owned()
            {
                self.remove_ref_obj(vec![ori.clone()]);
            } else {
                self.update_ref_obj_name(ori, new);
            }
        }
        Ok(())
    }

    fn __getitem__(&self, ob: &PyAny, py: Python) -> PyResult<PyObject> {
        let out = self.get(ob.into()).map_err(StrError::to_py)?;
        match out {
            GetOutput::Expr(expr) => Ok(self.map_expr_to_py(expr.clone()).into_py(py)),
            GetOutput::Exprs(exprs) => Ok(exprs
                .into_iter()
                .map(|e| self.map_expr_to_py(e.clone()))
                .collect_trusted()
                .into_py(py)),
        }
    }

    pub unsafe fn __setitem__(&mut self, key: &PyAny, value: &PyAny) -> PyResult<()> {
        let value = parse_expr_list(value, false)?;
        let cs = ColumnSelector::from(key);
        let ori_name = self.get_selector_out_name(cs.clone());
        if ori_name.len() != value.len() {
            return Err(StrError("The length of keys and values are not equal".into()).to_py());
        }
        let (value, value_obj_map) = value.into_rs(Some(ori_name.clone()))?;
        self.remove_ref_obj(ori_name);
        self.set(cs, value.into()).map_err(StrError::to_py)?;
        self.add_obj_map(value_obj_map);
        Ok(())
    }

    pub fn __delitem__(&mut self, item: &PyAny) -> PyResult<()> {
        self.drop(item)
    }

    // pub unsafe fn insert(&mut self, value: &PyAny) -> PyResult<()> {
    //     let value = parse_expr_list(value, false)?;
    //     for e in value {
    //         self.insert_inplace(e.inner).map_err(StrError::to_py)?;
    //     }
    //     Ok(())
    // }

    #[pyo3(name = "select")]
    pub unsafe fn select_py(&mut self, exprs: &PyAny) -> PyResult<Self> {
        let exprs = parse_expr_list(exprs, false)?;
        let (exprs, obj_map) = exprs.into_rs(None)?;
        Ok(DataDict::new(exprs, None).to_py(obj_map))
    }

    #[pyo3(signature=(exprs))]
    pub unsafe fn with_columns(&mut self, exprs: &PyAny) -> PyResult<()> {
        let exprs = parse_expr_list(exprs, false)?;
        let (exprs, obj_map) = exprs.into_rs(None)?;
        exprs
            .into_iter()
            .try_for_each(|e| self.dd.insert_inplace(e))
            .map_err(StrError::to_py)?;
        self.add_obj_map(obj_map);
        Ok(())
    }

    #[pyo3(signature=(func, exclude=None, **py_kwargs))]
    pub fn apply(
        &mut self,
        func: &PyAny,
        exclude: Option<Vec<&str>>,
        py_kwargs: Option<&PyDict>,
    ) -> PyResult<()> {
        if self.is_empty() {
            return Ok(());
        }
        let ori_name = self.dd.columns_owned();
        let out_data = self
            .dd
            .data
            .iter()
            .map(|e| {
                if let Some(exclude) = &exclude {
                    if !exclude.contains(&e.ref_name().unwrap()) {
                        return parse_expr_nocopy(
                            func.call((e.clone().to_py(None),), py_kwargs)
                                .expect("Call python function error!"),
                        )
                        .expect("Can not parse function return as Expr");
                    } else {
                        e.clone().to_py(None)
                    }
                } else {
                    return parse_expr_nocopy(
                        func.call((e.clone().to_py(None),), py_kwargs)
                            .expect("Call python function error!"),
                    )
                    .expect("Can not parse function return as Expr");
                }
            })
            .collect_trusted();
        let new_name = out_data
            .iter()
            .map(|e| e.e.name().unwrap())
            .collect_trusted();
        for (ori, new) in zip(ori_name, new_name) {
            if ori != new {
                self.update_ref_obj_name(ori.as_str(), new.as_str());
                self.dd.update_column_map(ori, new)?
            }
        }
        let (rs_data, obj_map) = out_data.into_rs(None)?;
        self.add_obj_map(obj_map);
        self.dd.data = rs_data;
        Ok(())
    }
}
