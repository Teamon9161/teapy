use pyo3::exceptions::PyTypeError;
use regex::Regex;
use std::cmp::Ordering;
use std::collections::hash_map::RawEntryMut;
use std::iter::repeat;
use std::sync::{Arc, Mutex};

use crate::arr::{join_left, JoinType};

use super::export::*;

static PYDATADICT_INIT_SIZE: usize = 10;
// use std::collections::HashMap;
use ahash::{HashMap, HashMapExt};

#[pyclass]
#[derive(Clone)]
pub struct PyDataDict {
    data: Vec<PyExpr>,
    column_map: Arc<Mutex<HashMap<String, usize>>>,
}

#[allow(clippy::missing_safety_doc)]
impl PyDataDict {
    pub fn new(mut data: Vec<PyExpr>, columns: Option<Vec<String>>) -> Self {
        if let Some(columns) = columns {
            assert_eq!(data.len(), columns.len());
            let mut column_map =
                HashMap::<String, usize>::with_capacity(data.len().max(PYDATADICT_INIT_SIZE));
            for (col_name, i) in zip(columns, 0..data.len()) {
                column_map.insert(col_name.clone(), i);
                let expr = unsafe { data.get_unchecked_mut(i) };
                expr.set_name(col_name);
            }
            PyDataDict {
                data,
                column_map: Arc::new(Mutex::new(column_map)),
            }
        } else {
            let mut column_map = HashMap::<String, usize>::with_capacity(data.len());
            let mut name_auto = 0;
            for i in 0..data.len() {
                // we must check the name of PyExpr
                let expr = unsafe { data.get_unchecked_mut(i) };
                if let Some(name) = expr.get_name() {
                    column_map.insert(name.to_string(), i);
                } else {
                    // the expr doen't have a name, we must ensure the name of
                    // expr are the same with the name in column_map.
                    column_map.insert(name_auto.to_string(), i);
                    expr.set_name(name_auto.to_string());
                    name_auto += 1;
                }
            }
            PyDataDict {
                data,
                column_map: Arc::new(Mutex::new(column_map)),
            }
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline(always)]
    pub fn into_data(self) -> Vec<PyExpr> {
        self.data
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // fn create_column_map(&mut self) {
    //     let mut column_map = HashMap::<String, usize>::with_capacity(self.data.len());
    //     let mut name_auto = 0;
    //     for i in 0..self.data.len() {
    //         // we must check the name of PyExpr
    //         let expr = unsafe { self.data.get_unchecked_mut(i) };
    //         if let Some(name) = expr.get_name() {
    //             column_map.insert(name.to_string(), i);
    //         } else {
    //             // the expr doen't have a name, we must ensure the name of
    //             // expr are the same with the name in column_map.
    //             column_map.insert(name_auto.to_string(), i);
    //             expr.set_name(name_auto.to_string());
    //             name_auto += 1;
    //         }
    //     }
    //     self.column_map = Arc::new(Mutex::new(column_map));
    // }

    /// If the name of expression has changed after evaluating, we should update the column_map.
    fn update_column_map(
        &mut self,
        ori_name: Option<String>,
        new_name: Option<String>,
    ) -> PyResult<()> {
        if ori_name != new_name {
            let mut map = self.column_map.lock().unwrap();
            let i = map.remove(&ori_name.unwrap()).unwrap();
            let insert_res = map.insert(new_name.unwrap(), i);
            // the new name is equal to an existed expression
            if let Some(_idx) = insert_res {
                Err(PyValueError::new_err(
                    "The name of the new expression is duplicated",
                ))
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }

    /// Copy the column map
    pub fn map_to_own(&mut self) {
        let map = self.column_map.lock().unwrap().clone();
        self.column_map = Arc::new(Mutex::new(map));
    }

    pub fn _drop(&mut self, columns: Vec<&str>, inplace: bool) -> Option<Self> {
        let columns: Vec<String> = columns
            .into_iter()
            .flat_map(|col| self.get_matched_regex_column(col))
            .collect();
        if inplace {
            let mut column_map = self.column_map.lock().unwrap();
            for col in &columns {
                column_map.remove(col);
            }
            self.data = self
                .data
                .drain_filter(|e| !columns.contains(&e.get_name().unwrap()))
                .collect::<Vec<_>>();
            None
        } else {
            let mut column_map = self.column_map.lock().unwrap().clone();
            for col in &columns {
                column_map.remove(col);
            }
            let data = self
                .data
                .clone()
                .drain_filter(|e| !columns.contains(&e.get_name().unwrap()))
                .collect::<Vec<_>>();
            Some(PyDataDict {
                data,
                column_map: Arc::new(Mutex::new(column_map)),
            })
        }
    }

    /// Adjust when idx < 0
    fn valid_idx(&self, col_idx: i32) -> usize {
        let mut col_idx = col_idx;
        if col_idx < 0 {
            col_idx += self.len() as i32;
            assert!(col_idx >= 0, "Column doesn't exist!");
        }
        col_idx as usize
    }

    pub fn eval_by_str(&mut self, col_name: &str) -> PyResult<()> {
        let expr = self.get_mut_by_str(col_name);
        let mut expr_own = mem::take(expr);
        let ori_name = expr_own.get_name();
        expr_own.eval_py(true);
        let new_name = expr_own.get_name();
        *expr = expr_own;
        self.update_column_map(ori_name, new_name)?;
        Ok(())
    }

    pub fn eval_by_idx(&mut self, col_idx: i32) -> PyResult<()> {
        let expr = self.get_mut_by_idx(col_idx);
        let mut expr_own = mem::take(expr);
        let ori_name = expr_own.get_name();
        expr_own.eval_py(true);
        let new_name = expr_own.get_name();
        *expr = expr_own;
        self.update_column_map(ori_name, new_name)?;
        Ok(())
    }

    pub fn get_by_str(&self, col_name: &str) -> &PyExpr {
        self.data
            .get(
                *self
                    .column_map
                    .lock()
                    .unwrap()
                    .get(col_name)
                    .unwrap_or_else(|| panic!("{col_name} isn't a column")),
            )
            .unwrap()
    }

    pub fn get_matched_regex_column(&self, col_name: &str) -> Vec<String> {
        if col_name.starts_with('^') & col_name.ends_with('$') {
            let re = Regex::new(col_name).expect("Invalid regex");
            self.data
                .iter()
                .filter(|e| re.is_match(e.get_name().unwrap().as_str()))
                .map(|e| e.get_name().unwrap())
                .collect()
        } else {
            vec![col_name.to_string()]
        }
    }

    pub fn get_by_regex(&self, col_name: &str) -> Result<Vec<PyExpr>, &'static str> {
        if col_name.starts_with('^') & col_name.ends_with('$') {
            let re = Regex::new(col_name).expect("Invalid regex");
            let out: Vec<PyExpr> = self
                .data
                .iter()
                .filter(|e| re.is_match(e.get_name().unwrap().as_str()))
                .cloned()
                // .map(|e| e.clone())
                .collect();
            Ok(out)
        } else {
            Ok(vec![self.get_by_str(col_name).clone()])
        }
    }

    pub fn get_mut_by_str(&mut self, col_name: &str) -> &mut PyExpr {
        self.data
            .get_mut(
                *self
                    .column_map
                    .lock()
                    .unwrap()
                    .get(col_name)
                    .unwrap_or_else(|| panic!("{col_name} isn't a column")),
            )
            .unwrap()
    }

    pub fn get_by_idx(&self, col_idx: i32) -> &PyExpr {
        let col_idx = self.valid_idx(col_idx);
        self.data
            .get(col_idx)
            .unwrap_or_else(|| panic!("col index: {col_idx} doesn't exist"))
    }

    pub fn get_mut_by_idx(&mut self, col_idx: i32) -> &mut PyExpr {
        let col_idx = self.valid_idx(col_idx);
        self.data
            .get_mut(col_idx)
            .unwrap_or_else(|| panic!("col index: {col_idx} doesn't exist"))
    }

    /// Insert a new value or update the old value,
    /// the column name will be the name of the value expression.
    /// The caller must ensure that the name of the value is not None;
    pub fn _insert(&mut self, value: PyExpr) {
        let mut column_map = self.column_map.lock().unwrap();
        let col_name = value
            .get_name()
            .expect("The value expression must have a name when insert");
        let col_idx = *column_map.get(col_name.as_str()).unwrap_or(&self.len());
        // let col_name = col_name.to_string();
        if col_idx == self.len() {
            // insert a new column
            column_map.insert(col_name, self.len());
            self.data.push(value);
        } else {
            // update a existed column
            let col = self.data.get_mut(col_idx).unwrap();
            let col_name_to_drop = col.get_name().unwrap(); // record the old name
            *col = value;
            let res = column_map.insert(col_name, col_idx);
            if res.is_none() {
                // the name is different, so we must drop the old name
                column_map.remove(&col_name_to_drop);
            }
        }
    }

    /// Set a new column or replace an existed column using col name
    pub fn set_by_name(&mut self, col_name: String, value: PyExpr) {
        let mut value = value;
        value.set_name(col_name);
        self._insert(value);
    }

    /// Set a new column or replace an existed column using col index
    pub fn set_by_idx(&mut self, col_idx: i32, value: PyExpr) {
        let col_idx = self.valid_idx(col_idx);
        let mut value = value;
        assert!(
            col_idx <= self.len(),
            "col index: {} should be set first",
            self.len()
        );
        if col_idx < self.len() {
            // replace a column
            // We use the value's name by default if the value expression has a name.
            if value.get_name().is_none() {
                // if the value expression doesn't have a name, we use the old name
                value.set_name(self.data[col_idx].get_name().unwrap())
            }
            self._insert(value);
        } else {
            // insert a new column
            if value.get_name().is_none() {
                value.set_name(self.len().to_string())
            }
            self._insert(value);
        }
    }

    #[allow(unreachable_patterns)]
    pub fn select_on_axis(&self, slc: Vec<usize>, axis: Option<i32>) -> PyDataDict {
        let mut out_data = Vec::<PyExpr>::with_capacity(slc.len());
        let axis = axis.unwrap_or(0);
        let slc_expr: Expr<'static, usize> = slc.into();
        for expr in &self.data {
            out_data.push(match_exprs!(&expr.inner, e, {
                e.clone()
                    .select_on_axis_by_expr(slc_expr.clone(), axis.into())
                    .to_py(expr.obj())
            }))
        }
        PyDataDict {
            data: out_data,
            column_map: self.column_map.clone(),
        }
    }

    /// Evaluate some columns of the datadict in parallel
    pub fn eval_multi(&mut self, cols: &[&str]) -> PyResult<()> {
        self.data.par_iter_mut().for_each(|e| {
            if cols.contains(&e.get_name().unwrap().as_str()) {
                e.eval_inplace()
            }
        });
        Ok(())
    }

    /// Evaluate the whole datadict
    pub fn eval_all(&mut self) -> PyResult<()> {
        self.data.par_iter_mut().for_each(|e| e.eval_inplace());
        Ok(())
    }

    /// Evaluate some columns of the datadict in parallel
    pub fn eval_multi_by_idx(&mut self, cols: &[i32]) -> PyResult<()> {
        let len = self.data.len();
        let cols = cols
            .iter()
            .map(|idx| {
                if idx < &0 {
                    len + *idx as usize
                } else {
                    *idx as usize
                }
            })
            .collect_trusted();
        self.data.par_iter_mut().enumerate().for_each(|(i, e)| {
            if cols.contains(&i) {
                e.eval_inplace()
            }
        });
        Ok(())
    }
}

#[pymethods]
#[allow(clippy::missing_safety_doc)]
impl PyDataDict {
    #[new]
    #[pyo3(signature=(data, columns=None, copy=false))]
    pub fn init(data: &PyAny, columns: Option<Vec<String>>, copy: bool) -> PyResult<Self> {
        let data = unsafe { parse_expr_list(data, copy)? };
        Ok(Self::new(data, columns))
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_pd(&mut self) -> PyResult<PyObject> {
        self.eval_all()?;
        Python::with_gil(|py| {
            let pd = PyModule::import(py, "pandas").unwrap(); //.to_object(py)
            Ok(pd.getattr("DataFrame")?.call1((self.to_dict(py)?,))?.into())
        })
    }

    // #[args(inplace = false)]
    #[pyo3(signature=(columns, inplace=false))]
    pub fn drop(&mut self, columns: &PyAny, inplace: bool) -> PyResult<Option<Self>> {
        if let Ok(columns) = columns.extract::<&str>() {
            Ok(self._drop(vec![columns], inplace))
        } else if let Ok(columns) = columns.extract::<Vec<&str>>() {
            Ok(self._drop(columns, inplace))
        } else {
            Err(PyTypeError::new_err(
                "The type of parameter columns is invalid!",
            ))
        }
    }

    #[getter]
    pub fn get_dtypes<'py>(&'py self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let res = PyDict::new(py);
        for e in &self.data {
            res.set_item(e.get_name().unwrap(), e.dtype())?;
        }
        Ok(res)
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_dict<'py>(&'py mut self, py: Python<'py>) -> PyResult<&'py PyDict> {
        self.eval_all()?;
        let dict = PyDict::new(py);
        for expr in &self.data {
            dict.set_item(expr.get_name(), expr.value(py)?)?;
        }
        Ok(dict)
    }

    pub fn __len__(&self) -> usize {
        self.len()
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }

    #[getter]
    pub fn get_len(&self) -> usize {
        self.len()
    }

    #[getter]
    pub fn get_columns(&self) -> Vec<String> {
        self.data.iter().map(|e| e.get_name().unwrap()).collect()
    }

    #[getter]
    fn get_column_map(&self) -> HashMap<String, usize> {
        self.column_map.lock().unwrap().clone()
    }

    #[getter]
    pub fn get_raw_data(&self) -> Vec<PyExpr> {
        self.data.clone()
    }

    #[setter]
    pub fn set_columns(&mut self, columns: Vec<String>) {
        assert_eq!(self.len(), columns.len());
        let mut column_map = self.column_map.lock().unwrap();
        zip(self.data.iter_mut(), columns.into_iter()).for_each(|(e, name)| {
            // We should get the original name at first, and then find
            // the column index and insert a new map in `column_map`
            let name_ori = e.get_name().unwrap();
            let idx = column_map.remove(&name_ori).unwrap();
            column_map.insert(name.clone(), idx);
            e.set_name(name);
        });
    }

    #[pyo3(signature=(ob=None, inplace=true))]
    pub fn eval(&mut self, ob: Option<&PyAny>, inplace: bool) -> PyResult<Option<Self>> {
        if let Some(ob) = ob {
            if let Ok(col_name) = ob.extract::<&str>() {
                self.eval_by_str(col_name)?;
            } else if let Ok(col_idx) = ob.extract::<i32>() {
                self.eval_by_idx(col_idx)?;
            } else if let Ok(col_name_vec) = ob.extract::<Vec<&str>>() {
                // todo: evaluate in parallel
                self.eval_multi(&col_name_vec)?;
                // col_name_vec.into_iter().for_each(|col_name| self.eval_by_str(col_name).unwrap())
            } else if let Ok(col_idx_vec) = ob.extract::<Vec<i32>>() {
                self.eval_multi_by_idx(&col_idx_vec)?;
                // col_idx_vec.into_iter().for_each(|col_idx| self.eval_by_idx(col_idx).unwrap())
            } else {
                return Err(PyValueError::new_err("Not support type in eval"));
            }
        } else if ob.is_none() {
            // eval the whole datadict
            self.eval_all()?;
        }
        if inplace {
            Ok(None)
        } else {
            // may not be expected because this is not a deep clone,
            // but one should not use the original DataDict after evaluating it.
            Ok(Some(self.clone()))
        }
    }

    fn __getitem__(&self, ob: &PyAny, py: Python) -> PyResult<PyObject> {
        if let Ok(col_name) = ob.extract::<&str>() {
            let out = self.get_by_regex(col_name).map_err(PyValueError::new_err)?;
            if out.len() == 1 {
                Ok(out[0].clone().into_py(py))
            } else {
                Ok(PyDataDict::new(out, None).into_py(py))
            }
        } else if let Ok(col_idx) = ob.extract::<i32>() {
            Ok(self.get_by_idx(col_idx).clone().into_py(py))
        } else if let Ok(col_name_vec) = ob.extract::<Vec<&str>>() {
            let out = col_name_vec
                .into_iter()
                .flat_map(|col_name| self.get_by_regex(col_name).unwrap())
                .collect();
            Ok(PyDataDict::new(out, None).into_py(py))
        } else if let Ok(col_idx_vec) = ob.extract::<Vec<i32>>() {
            let out = col_idx_vec
                .into_iter()
                .map(|col_idx| self.get_by_idx(col_idx).clone())
                .collect_trusted();
            Ok(PyDataDict::new(out, None).into_py(py))
        } else {
            Err(PyValueError::new_err("Not support type in get item"))
        }
    }

    pub unsafe fn __setitem__(&mut self, key: &PyAny, value: &PyAny, py: Python) -> PyResult<()> {
        if let Ok(col_name) = key.extract::<String>() {
            let col_name_vec = self.get_matched_regex_column(&col_name);
            match col_name_vec.len().cmp(&1) {
                Ordering::Greater => {
                    let key = col_name_vec.into_py(py);
                    self.__setitem__(key.as_ref(py), value, py)
                }
                Ordering::Equal => {
                    let value = parse_expr_nocopy(value)?;
                    self.set_by_name(col_name_vec[0].clone(), value);
                    Ok(())
                }
                Ordering::Less => Err(PyValueError::new_err("The column name doesn't exist")),
            }
        } else if let Ok(col_idx) = key.extract::<i32>() {
            let value = parse_expr_nocopy(value)?;
            self.set_by_idx(col_idx, value);
            Ok(())
        } else if let Ok(col_name_vec) = key.extract::<Vec<String>>() {
            let col_name_vec: Vec<String> = col_name_vec
                .into_iter()
                .flat_map(|col| self.get_matched_regex_column(&col))
                .collect();
            let value_vec = parse_expr_list(value, false)?;
            if value_vec.len() != col_name_vec.len() {
                if value_vec.len() == 1 {
                    col_name_vec
                        .into_iter()
                        .for_each(|col_name| self.set_by_name(col_name, value_vec[0].deep_copy()));
                } else {
                    return Err(PyValueError::new_err(
                        "The length of columns and values are not equal",
                    ));
                }
            } else {
                zip(col_name_vec, value_vec)
                    .for_each(|(col_name, value)| self.set_by_name(col_name, value));
            }
            Ok(())
        } else if let Ok(col_idx_vec) = key.extract::<Vec<i32>>() {
            let value_vec = parse_expr_list(value, false)?;
            if value_vec.len() != col_idx_vec.len() {
                if value_vec.len() == 1 {
                    col_idx_vec
                        .into_iter()
                        .for_each(|col_idx| self.set_by_idx(col_idx, value_vec[0].deep_copy()));
                } else {
                    return Err(PyValueError::new_err(
                        "The length of columns and values are not equal",
                    ));
                }
            } else {
                zip(col_idx_vec, value_vec)
                    .for_each(|(col_idx, value)| self.set_by_idx(col_idx, value));
            }
            Ok(())
        } else {
            Err(PyValueError::new_err("Not support type in set item"))
        }
    }

    pub fn __delitem__(&mut self, item: &PyAny) -> PyResult<()> {
        if let Ok(item) = item.extract::<&str>() {
            self._drop(vec![item], true);
        } else if let Ok(item) = item.extract::<Vec<&str>>() {
            self._drop(item, true);
        } else {
            return Err(PyValueError::new_err("Not support this type of item."));
        }
        Ok(())
    }

    pub fn insert(&mut self, value: Vec<PyExpr>) {
        value.into_iter().for_each(|e| self._insert(e));
    }

    pub unsafe fn select(&mut self, exprs: &PyAny) -> PyResult<Self> {
        let exprs = parse_expr_list(exprs, false)?;
        Ok(PyDataDict::new(exprs, None))
    }

    #[pyo3(signature=(exprs, inplace=false))]
    pub unsafe fn with_columns(&mut self, exprs: &PyAny, inplace: bool) -> PyResult<Option<Self>> {
        let exprs = parse_expr_list(exprs, false)?;
        if !inplace {
            let mut out = self.clone();
            // note that we didn't deep clone the expression, we only clone a column_map.
            // This is faster but some inplace functions may affect both DataDicts.
            out.map_to_own();
            exprs.into_iter().for_each(|e| out._insert(e));
            Ok(Some(out))
        } else {
            exprs.into_iter().for_each(|e| self._insert(e));
            Ok(None)
        }
    }

    #[pyo3(signature=(func, **py_kwargs))]
    pub fn apply(&self, func: &PyAny, py_kwargs: Option<&PyDict>) -> PyResult<Self> {
        if self.is_empty() {
            return Ok(self.clone());
        }
        let out_data = unsafe {
            self.data
                .iter()
                .map(|e| {
                    parse_expr_nocopy(
                        func.call((e.clone(),), py_kwargs)
                            .expect("Call python function error!"),
                    )
                    .expect("Can not parse fucntion return as Expr")
                })
                .collect_trusted()
        };
        Ok(PyDataDict::new(out_data, None))
    }

    #[pyo3(signature=(window, func, axis=0, check=true, **py_kwargs))]
    #[allow(unreachable_patterns)]
    pub fn rolling_apply(
        &mut self,
        window: usize,
        func: &PyAny,
        axis: i32,
        check: bool,
        py_kwargs: Option<&PyDict>,
    ) -> PyResult<Self> {
        if self.is_empty() {
            return Ok(self.clone());
        }
        if window == 0 {
            return Err(PyValueError::new_err("Window should be greater than 0"));
        }
        self.eval_all()?;
        let axis = match_exprs!(&self.data[0].inner, expr, {
            expr.view_arr().norm_axis(axis)
        });
        let axis_i = axis.index() as i32;
        let length = match_exprs!(&self.data[0].inner, expr, {
            expr.view_arr().shape()[axis.index()]
        });
        if check {
            for e in &self.data {
                let len_ = match_exprs!(&e.inner, expr, { expr.view_arr().shape()[axis.index()] });
                if length != len_ {
                    return Err(PyValueError::new_err(
                        "Each Expressions should have the same length on given axis",
                    ));
                }
            }
        }
        let mut column_num = 0;
        let mut output = zip(repeat(0).take(window - 1), 0..window - 1)
            .chain((window - 1..length).enumerate())
            .map(|(start, end)| {
                let mut step_df = Vec::with_capacity(self.len());
                self.data.iter().for_each(|pyexpr| unsafe {
                    step_df.push(pyexpr.take_by_slice(Some(axis_i), start, end, None));
                });
                let step_df = PyDataDict {
                    data: step_df,
                    column_map: self.column_map.clone(),
                };
                let res = func
                    .call((step_df,), py_kwargs)
                    .expect("Call python function error!");
                let res = unsafe {
                    parse_expr_list(res, false).expect("Can not parse fucntion return as Expr list")
                };
                column_num = res.len();
                res
            })
            .collect_trusted();
        output
            .par_iter_mut()
            .for_each(|vec_e| vec_e.par_iter_mut().for_each(|e| e.eval_inplace()));

        let out_data = (0..column_num)
            .into_par_iter()
            .map(|i| {
                let group_vec = output
                    .iter()
                    .map(|single_output_exprs| single_output_exprs.get(i).unwrap().no_dim0())
                    .collect_trusted();
                concat_expr(group_vec, axis_i).expect("Concat expr error")
            })
            .collect();
        Ok(PyDataDict::new(out_data, None))
    }

    #[pyo3(signature=(duration, func, index=None, axis=0, check=true, **py_kwargs))]
    #[allow(unreachable_patterns)]
    pub fn rolling_apply_by_time(
        &mut self,
        duration: &str,
        func: &PyAny,
        index: Option<&str>,
        axis: i32,
        check: bool,
        py_kwargs: Option<&PyDict>,
    ) -> PyResult<Self> {
        if self.is_empty() {
            return Ok(self.clone());
        }
        self.eval_all()?;
        let index = if let Some(index) = index {
            self.get_by_str(index)
        } else {
            let mut dt_num = 0; // number of datetime expression in datadict
            let mut out = None;
            for expr in &self.data {
                if expr.dtype() == "DateTime" {
                    out = Some(expr);
                    dt_num += 1;
                }
            }
            if dt_num != 1 {
                return Err(PyValueError::new_err(
                    "The Number of DateTime Expression in DataDict should be 1",
                ));
            } else {
                out.unwrap()
            }
        };
        let axis = match_exprs!(&self.data[0].inner, expr, {
            expr.view_arr().norm_axis(axis)
        });
        let axis_i = axis.index() as i32;
        let length = match_exprs!(&self.data[0].inner, expr, {
            expr.view_arr().shape()[axis.index()]
        });
        if check {
            for e in &self.data {
                let len_ = match_exprs!(&e.inner, expr, { expr.view_arr().shape()[axis.index()] });
                if length != len_ {
                    return Err(PyValueError::new_err(
                        "Each Expressions should have the same length on given axis",
                    ));
                }
            }
        }
        let mut rolling_idx = index.clone().cast_datetime(None)?.time_rolling(duration);
        rolling_idx.eval_inplace();
        let mut column_num = 0;
        let mut output = rolling_idx
            .view_arr()
            .to_dim1()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(end, start)| {
                let mut step_df = Vec::with_capacity(self.len());
                self.data.iter().for_each(|pyexpr| unsafe {
                    step_df.push(pyexpr.take_by_slice(Some(axis_i), start, end, None));
                });
                let step_df = PyDataDict {
                    data: step_df,
                    column_map: self.column_map.clone(),
                };
                let res = func
                    .call((step_df,), py_kwargs)
                    .expect("Call python function error!");
                let res = unsafe {
                    parse_expr_list(res, false).expect("Can not parse fucntion return as Expr list")
                };
                column_num = res.len();
                res
            })
            .collect_trusted();
        output
            .par_iter_mut()
            .for_each(|vec_e| vec_e.par_iter_mut().for_each(|e| e.eval_inplace()));
        let out_data = (0..column_num)
            .into_par_iter()
            .map(|i| {
                let group_vec = output
                    .iter()
                    .map(|single_output_exprs| single_output_exprs.get(i).unwrap().no_dim0())
                    .collect_trusted();
                concat_expr(group_vec, axis_i).expect("Concat expr error")
            })
            .collect();
        Ok(PyDataDict::new(out_data, None))
    }

    #[pyo3(signature=(by, rev=false, inplace=false))]
    pub fn sort_by(&mut self, by: Vec<&str>, rev: bool, inplace: bool) -> Option<Self> {
        let key: Vec<PyExpr> = by.into_iter().map(|n| self.get_by_str(n).clone()).collect();
        if self.data.is_empty() {
            return Some(self.clone());
        }
        let idx = self.data.get(0).unwrap().clone().sort_by_expr_idx(key, rev);
        unsafe {
            self.data.iter_mut().for_each(|e| {
                *e = e
                    .clone()
                    ._take_on_axis_by_expr_unchecked(idx.clone(), 0, false)
                    .unwrap()
            });
        }
        if !inplace {
            Some(self.clone())
        } else {
            None
        }
    }

    #[pyo3(signature=(by, axis=0, sort=true, par=false))]
    pub fn groupby(&mut self, by: &PyAny, axis: i32, sort: bool, par: bool) -> PyResult<PyGroupBy> {
        let by = parse_one_or_more_str(by)?;
        Ok(PyGroupBy::new(self._groupby(by, axis, sort, par)?, axis))
    }

    /// Groupby and consume the dfs generated during the groupby process
    /// this is faster as we don't need to clone `Vec<PyDataDict>`
    #[pyo3(signature=(py_func, by, axis=0, sort=true, par=false, **py_kwargs))]
    pub fn groupby_apply(
        &mut self,
        py_func: &PyAny,
        by: &PyAny,
        axis: i32,
        sort: bool,
        par: bool,
        py_kwargs: Option<&PyDict>,
    ) -> PyResult<PyDataDict> {
        let by = parse_one_or_more_str(by)?;
        let group_dfs = self._groupby(by, axis, sort, par)?;
        super::groupby::groupby_apply(group_dfs, py_func, axis, py_kwargs)
    }

    #[allow(unreachable_patterns)]
    #[pyo3(signature=(right, left_on, right_on, method=JoinType::Left))]
    pub fn join(
        &mut self,
        mut right: Self,
        left_on: &PyAny,
        right_on: &PyAny,
        method: JoinType,
    ) -> PyResult<Self> {
        let left_on = parse_one_or_more_str(left_on)?;
        let right_on = parse_one_or_more_str(right_on)?;
        self.eval_multi(&left_on)?;
        right.eval_multi(&right_on)?;
        let left_on_exprs = left_on
            .into_iter()
            .map(|key| &self.get_by_str(key).inner)
            .collect_trusted();
        let right_on_exprs = right_on
            .clone()
            .into_iter()
            .map(|key| &right.get_by_str(key).inner)
            .collect_trusted();
        match method {
            JoinType::Left => {
                let idx = join_left(left_on_exprs, right_on_exprs);
                let idx: Expr<'static, Option<usize>> = idx.into();
                unsafe {
                    right.data = right
                        .data
                        .into_iter()
                        .map(|e| {
                            use Exprs::*;
                            match &e.inner {
                                F32(_) | F64(_) | String(_) => match_exprs!(
                                    &e.inner,
                                    expr,
                                    {
                                        expr.clone()
                                            .take_option_on_axis_by_expr_unchecked(
                                                idx.clone(),
                                                0,
                                                false,
                                            )
                                            .to_py(e.obj())
                                    },
                                    F32,
                                    F64,
                                    String
                                ),
                                I32(_) | I64(_) | Usize(_) => match_exprs!(
                                    &e.inner,
                                    expr,
                                    {
                                        expr.clone()
                                            .cast::<f64>()
                                            .take_option_on_axis_by_expr_unchecked(
                                                idx.clone(),
                                                0,
                                                false,
                                            )
                                            .to_py(e.obj())
                                    },
                                    I32,
                                    I64,
                                    Usize
                                ),
                                _ => {
                                    // other dtypes that doesn't support a option index, we assumed that the index are all Some
                                    let idx: Expr<'static, usize> =
                                        std::mem::transmute(idx.clone());
                                    match_exprs!(&e.inner, expr, {
                                        expr.clone()
                                            .take_on_axis_by_expr_unchecked(idx, 0, false)
                                            .to_py(e.obj())
                                    })
                                }
                            }
                        })
                        .collect_trusted();
                }
                let mut data = [self.data.clone(), right.data].concat();
                let right_map = right.column_map.lock().unwrap();
                let len = self.len();
                let mut map = self.column_map.lock().unwrap().clone();
                for (k, v) in right_map.iter() {
                    if right_on.contains(&k.as_str()) {
                        continue;
                    }
                    if map.get(k).is_none() {
                        map.insert(k.clone(), *v + len);
                    } else {
                        // duplicate column name
                        let new_name = k.clone() + "_right";
                        map.insert(new_name.clone(), *v + len);
                        data.get_mut(*v + len).unwrap().set_name(new_name);
                    }
                }
                Ok(PyDataDict {
                    data,
                    column_map: Arc::new(Mutex::new(map)),
                })
            }
            _ => todo!(),
        }
    }

    #[pyo3(signature=(subset, keep="first", inplace=false, axis=0))]
    #[allow(unreachable_patterns)]
    pub fn unique(
        &mut self,
        subset: &PyAny,
        keep: &str,
        inplace: bool,
        axis: i32,
    ) -> PyResult<Option<Self>> {
        let subset = parse_one_or_more_str(subset)?;
        if subset.is_empty() {
            return Err(PyValueError::new_err("subset cannot be empty"));
        }
        self.eval_multi(&subset)?;
        let subset_exprs = subset
            .into_iter()
            .map(|key| &self.get_by_str(key).inner)
            .collect_trusted();
        let (len, hasher, hashed_keys) = crate::arr::groupby::prepare_groupby(&subset_exprs, None);
        let mut out_idx = Vec::with_capacity(len);
        if keep == "first" {
            let mut map = HashMap::<u64, u8>::with_capacity_and_hasher(len, hasher.clone());
            if subset_exprs.len() == 1 {
                let hashed_key = &hashed_keys[0];
                for i in 0..len {
                    let hash = unsafe { *hashed_key.uget(i) };
                    let entry = map.raw_entry_mut().from_key_hashed_nocheck(hash, &hash);
                    if let RawEntryMut::Vacant(entry) = entry {
                        entry.insert_hashed_nocheck(hash, hash, 1);
                        out_idx.push(i);
                    }
                }
            } else {
                for i in 0..len {
                    let tuple_keys = hashed_keys
                        .iter()
                        .map(|keys| unsafe { *keys.uget(i) })
                        .collect_trusted();
                    let hash = hasher.hash_one(&tuple_keys);
                    let entry = map.raw_entry_mut().from_key_hashed_nocheck(hash, &hash);
                    if let RawEntryMut::Vacant(entry) = entry {
                        entry.insert_hashed_nocheck(hash, hash, 1);
                        out_idx.push(i);
                    }
                }
            }
        } else if keep == "last" {
            let mut map = HashMap::<u64, usize>::with_capacity_and_hasher(len, hasher.clone());
            if subset_exprs.len() == 1 {
                let hashed_key = &hashed_keys[0];
                for i in 0..len {
                    let hash = unsafe { *hashed_key.uget(i) };
                    let entry = map.raw_entry_mut().from_key_hashed_nocheck(hash, &hash);
                    match entry {
                        RawEntryMut::Vacant(entry) => {
                            entry.insert_hashed_nocheck(hash, hash, i);
                        }
                        RawEntryMut::Occupied(mut entry) => {
                            let v = entry.get_mut();
                            *v = i;
                        }
                    }
                }
            } else {
                for i in 0..len {
                    let tuple_keys = hashed_keys
                        .iter()
                        .map(|keys| unsafe { *keys.uget(i) })
                        .collect_trusted();
                    let hash = hasher.hash_one(&tuple_keys);
                    let entry = map.raw_entry_mut().from_key_hashed_nocheck(hash, &hash);
                    match entry {
                        RawEntryMut::Vacant(entry) => {
                            entry.insert_hashed_nocheck(hash, hash, i);
                        }
                        RawEntryMut::Occupied(mut entry) => {
                            let v = entry.get_mut();
                            *v = i;
                        }
                    }
                }
            }
            out_idx = map.into_values().collect_trusted();
            out_idx.sort_unstable()
        } else {
            return Err(PyValueError::new_err("keep must be either first or last"));
        }
        let new_len = out_idx.len();
        let idx: Expr<'static, usize> = out_idx.into();
        if inplace {
            unsafe {
                self.data.iter_mut().for_each(|s| {
                    if new_len < len {
                        *s = match_exprs!(&s.inner, expr, {
                            expr.clone()
                                .take_on_axis_by_expr_unchecked(idx.clone(), axis, false)
                                .to_py(s.obj())
                        })
                    }
                });
            }
            Ok(None)
        } else {
            let data = self
                .data
                .iter()
                .map(|s| {
                    if new_len < len {
                        unsafe {
                            match_exprs!(&s.inner, expr, {
                                expr.clone()
                                    .take_on_axis_by_expr_unchecked(idx.clone(), axis, false)
                                    .to_py(s.obj())
                            })
                        }
                    } else {
                        // fast path for no duplicates
                        s.clone()
                    }
                })
                .collect_trusted();
            Ok(Some(PyDataDict {
                data,
                column_map: self.column_map.clone(),
            }))
        }
    }
}

fn parse_one_or_more_str(s: &PyAny) -> PyResult<Vec<&str>> {
    if let Ok(s) = s.extract::<&str>() {
        Ok(vec![s])
    } else if let Ok(s) = s.extract::<Vec<&str>>() {
        Ok(s)
    } else {
        Err(PyValueError::new_err(
            "the param cann't be parsed as a vector of string",
        ))
    }
}
