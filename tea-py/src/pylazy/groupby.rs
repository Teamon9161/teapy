use pyo3::exceptions::PyRuntimeError;
use tea_groupby::groupby;
use super::export::*;

impl PyDataDict {
    pub(crate) fn _groupby(
        &mut self,
        keys: Vec<&str>,
        axis: i32,
        sort: bool,
        par: bool,
    ) -> PyResult<Vec<PyDataDict>> {
        self.eval_multi(&keys)?;
        let keys = keys
            .into_iter()
            .map(|key| &self.get_by_str(key).inner)
            .collect_trusted();
        let group_idx_vec = if par {
            // groupby_par(&keys, sort)
            // todo: use parallel groupby
            groupby(&keys, sort)
        } else {
            groupby(&keys, sort)
        };
        let mut output = Vec::<PyDataDict>::with_capacity(self.len());
        group_idx_vec.into_iter().for_each(|(_idx, idx_vec)| {
            output.push(self.select_on_axis_unchecked(idx_vec, Some(axis)))
        });
        Ok(output)
    }
}

pub(super) fn groupby_apply(
    group_dfs: Vec<PyDataDict>,
    py_func: &PyAny,
    axis: i32,
    py_kwargs: Option<&PyDict>,
) -> PyResult<PyDataDict> {
    assert!(
        py_func.is_callable(),
        "must pass a callable object to apply"
    );
    let mut exprs = Vec::<Vec<PyExpr>>::with_capacity(group_dfs.len());
    for df in group_dfs {
        let expr_list = py_func
            .call((df,), py_kwargs)
            .expect("call python function error!");
        let expr_list = unsafe { parse_expr_list(expr_list, false) }?;
        exprs.push(expr_list);
    }
    groupby_eval(exprs, axis)
}

fn groupby_eval(mut exprs: Vec<Vec<PyExpr>>, axis: i32) -> PyResult<PyDataDict> {
    let column_num = exprs[0].len(); // each group should have the same column num
    let eval_res: Vec<_> = exprs
        .par_iter_mut()
        .flatten()
        .map(|e| e.eval_inplace())
        .collect();
    if eval_res.iter().any(|e| e.is_err()) {
        return Err(PyRuntimeError::new_err(
            "Some of the expressions can't be evaluated",
        ));
    }

    let out_data = (0..column_num)
        .into_par_iter()
        .map(|i| {
            let group_vec = exprs
                .iter()
                .map(|single_group_exprs| single_group_exprs.get(i).unwrap().no_dim0())
                .collect_trusted();
            concat_expr(group_vec, axis).expect("concat expr error")
        })
        .collect();
    Ok(PyDataDict::new(out_data, None))
}

#[pyclass]
pub struct PyGroupBy {
    data: Vec<PyDataDict>,
    axis: i32,
}

#[pymethods]
impl PyGroupBy {
    #[new]
    #[pyo3(signature=(data, axis=0))]
    pub fn new(data: Vec<PyDataDict>, axis: i32) -> Self {
        PyGroupBy { data, axis }
    }

    pub fn data(&self) -> Vec<PyDataDict> {
        self.data.clone()
    }

    #[allow(clippy::borrow_deref_ref)]
    pub fn apply(&self, py_func: &PyAny, py_kwargs: Option<&PyDict>) -> PyResult<PyDataDict> {
        groupby_apply(self.data.clone(), py_func, self.axis, py_kwargs)
    }
}
