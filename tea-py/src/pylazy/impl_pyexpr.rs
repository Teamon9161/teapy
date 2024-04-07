use super::export::*;
use super::pyfunc::{parse_expr, parse_expr_list, parse_expr_nocopy};
use crate::from_py::{NoDim0, PyContext};
// use ahash::{HashMap, HashMapExt};
use ndarray::SliceInfoElem;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
#[cfg(feature = "ops")]
use pyo3::pyclass::CompareOp;
use pyo3::types::PyDict;
use pyo3::{
    exceptions::{PyAttributeError, PyTypeError},
    PyTraverseError, PyVisit,
};
use tea_hash::TpHashMap;

use tea_core::prelude::*;

#[cfg(feature = "map")]
use super::pyfunc::where_;
use tea_lazy::{Data, Expr};

#[cfg(feature = "agg")]
use tea_ext::agg::*;
#[cfg(feature = "map")]
use tea_ext::map::*;
#[cfg(feature = "rolling")]
use tea_ext::rolling::*;
#[cfg(feature = "blas")]
use tea_ext::ExprMapExt;
#[cfg(feature = "blas")]
use tea_ext::ExprStatExt;
#[cfg(all(feature = "map", feature = "time"))]
use tea_ext::ExprTimeExt;
#[cfg(all(feature = "rolling", feature = "agg", feature = "time"))]
use tea_ext::RollingTimeStartBy;

#[cfg(feature = "groupby")]
use tea_groupby::*;

static PYEXPR_ATTRIBUTE: Lazy<Mutex<TpHashMap<String, PyObject>>> =
    Lazy::new(|| Mutex::new(TpHashMap::<String, PyObject>::with_capacity(10)));

#[pyfunction]
#[pyo3(signature=(name, f))]
pub fn expr_register(name: String, f: PyObject) -> PyResult<()> {
    let mut attr_dict = PYEXPR_ATTRIBUTE.lock();
    let _ = attr_dict.insert(name, f);
    Ok(())
}

#[pymethods]
#[allow(clippy::missing_safety_doc)]
impl PyExpr {
    #[new]
    #[pyo3(signature=(expr=None, name=None, copy=false))]
    pub unsafe fn new(expr: Option<&PyAny>, name: Option<String>, copy: bool) -> PyResult<Self> {
        let mut out = if let Some(expr) = expr {
            parse_expr(expr, copy)?
        } else {
            Default::default()
        };
        if let Some(name) = name {
            out.set_name(name);
        }
        Ok(out)
    }

    #[getter]
    pub fn dtype(&self) -> String {
        // self.inner.dtype()
        self.e.dtype()
    }

    #[getter]
    pub fn get_base_type(&self) -> &'static str {
        // match_exprs!(&self.inner, e, { e.get_base_type() })
        self.e.base_type()
    }

    // #[getter]
    // pub fn get_base_strong_count(&self) -> PyResult<usize> {
    //     match_exprs!(&self.inner, e, {
    //         e.get_base_strong_count().map_err(StrError::to_py)
    //     })
    // }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(obj_vec) = &self.obj {
            for obj in obj_vec {
                visit.call(obj)?
            }
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        // Clear reference, this decrements ref counter.
        self.obj = None;
    }

    fn __repr__(&self) -> String {
        format!("{:?}", &self.e)
    }

    pub fn simplify(&mut self) {
        self.e.simplify()
    }

    #[cfg(all(feature = "map", feature = "agg", feature = "ops"))]
    pub unsafe fn __getitem__(&self, obj: &PyAny, py: Python) -> PyResult<Self> {
        use pyo3::types::{PyList, PySlice, PyTuple};
        if let Ok(length) = obj.len() {
            let mut slc_vec = Vec::with_capacity(10);
            let mut no_slice_idx_vec = Vec::with_capacity(length);
            let mut no_slice_slc_obj_vec = Vec::with_capacity(length);
            // if item is slice, slice first
            if obj.is_instance_of::<PyList>() {
                no_slice_idx_vec.push(0);
                no_slice_slc_obj_vec.push(obj);
                slc_vec.push(SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                });
            } else if obj.is_instance_of::<PyTuple>() {
                let obj_vec = obj.extract::<Vec<&PyAny>>().unwrap();
                for (idx, obj) in obj_vec.into_iter().enumerate() {
                    if let Ok(slc) = obj.extract::<&PySlice>() {
                        let start = slc
                            .getattr("start")?
                            .extract::<Option<isize>>()?
                            .unwrap_or(0);
                        let end = slc.getattr("stop")?.extract::<Option<isize>>()?;
                        let step = slc
                            .getattr("step")?
                            .extract::<Option<isize>>()?
                            .unwrap_or(1);
                        slc_vec.push(SliceInfoElem::Slice { start, end, step });
                    } else if obj.is_none() {
                        slc_vec.push(SliceInfoElem::NewAxis);
                    } else if let Ok(idx) = obj.extract::<isize>() {
                        slc_vec.push(SliceInfoElem::Index(idx))
                    } else {
                        no_slice_idx_vec.push(idx);
                        no_slice_slc_obj_vec.push(obj);
                        slc_vec.push(SliceInfoElem::Slice {
                            start: 0,
                            end: None,
                            step: 1,
                        });
                    }
                }
            }

            let mut out = self.clone();
            let ori_dim = out.ndim();
            out.e.view_by_slice(slc_vec);
            for (idx, slc_obj) in zip(no_slice_idx_vec, no_slice_slc_obj_vec) {
                let slc = parse_expr_nocopy(slc_obj)?;
                let obj = slc.obj();
                let idx: Expr<'static> = (idx as i32).into();
                let idx = idx - (ori_dim.e.clone() - out.ndim().e);
                out.e.select(slc.e, idx, true);
                out.add_obj(obj);
            }
            Ok(out)
        } else {
            let obj = PyTuple::new(py, [obj]);
            self.__getitem__(obj.into(), py)
        }
    }

    #[cfg(feature = "map")]
    pub fn __setitem__(&mut self, item: &PyAny, value: &PyAny) -> PyResult<()> {
        use pyo3::types::{PySlice, PyTuple};
        if let Ok(length) = item.len() {
            let value = parse_expr_nocopy(value)?;
            let value_ref = value.obj();
            let mut slc_vec = Vec::with_capacity(length);
            let obj_vec = item.extract::<Vec<&PyAny>>().unwrap();
            for obj in obj_vec {
                if let Ok(slc) = obj.extract::<&PySlice>() {
                    let start = slc
                        .getattr("start")?
                        .extract::<Option<isize>>()?
                        .unwrap_or(0);
                    let end = slc.getattr("stop")?.extract::<Option<isize>>()?;
                    let step = slc
                        .getattr("step")?
                        .extract::<Option<isize>>()?
                        .unwrap_or(1);
                    slc_vec.push(SliceInfoElem::Slice { start, end, step });
                } else if obj.is_none() {
                    slc_vec.push(SliceInfoElem::NewAxis)
                } else if let Ok(idx) = obj.extract::<isize>() {
                    slc_vec.push(SliceInfoElem::Index(idx))
                } else {
                    return Err(PyAttributeError::new_err(format!(
                        "item must be slice or None, not {}",
                        obj.get_type().name()?
                    )));
                }
            }
            self.e.set_item_by_slice(slc_vec, value.e);
            self.add_obj(value_ref);
        } else {
            let item = PyTuple::new(value.py(), [item]);
            self.__setitem__(item.into(), value)?;
        }
        Ok(())
    }

    #[pyo3(name="eval", signature=(inplace=false, context=None, freeze=true))]
    #[allow(unreachable_patterns)]
    pub fn eval_py(
        &mut self,
        inplace: bool,
        context: Option<&PyAny>,
        freeze: bool,
    ) -> PyResult<Option<Self>> {
        if !inplace {
            let mut e = self.clone();
            e.eval_inplace(context, freeze)?;
            Ok(Some(e))
        } else {
            self.eval_inplace(context, freeze)?;
            Ok(None)
        }
    }

    #[pyo3(signature=(context=None))]
    // #[cfg(feature = "time")]
    pub fn view_in(
        // slf: PyRefMut<'_, Self>,
        slf: Bound<'_, PyExpr>,
        context: Option<&Bound<'_, PyAny>>,
        py: Python,
    ) -> PyResult<PyObject> {
        let ct: PyContext<'static> = if let Some(context) = context {
            unsafe { std::mem::transmute(context.extract::<PyContext>()?) }
        } else {
            Default::default()
        };
        let (ct_rs, _obj_map) = (ct.ct, ct.obj_map);
        let slf_ref = slf.borrow();
        let data = slf_ref
            .e
            .view_data(ct_rs.as_ref())
            .map_err(StrError::to_py)?;
        let container = slf.into_any();
        if matches!(&data, Data::ArrVec(_)) {
            if let Data::ArrVec(arr_vec) = data {
                let out = arr_vec
                    .iter()
                    .map(|arr| {
                        match_arrok!(pyelement arr, a, {
                            unsafe{
                                // PyArray::borrow_from_array(&a.view().0, container)
                                PyArray::borrow_from_array_bound(&a.view().0, container.clone())
                                .no_dim0(py)
                                .unwrap()
                            }
                        })
                    })
                    .collect_trusted();
                return Ok(out.into_py(py));
            }
        }
        let arr = data.view_arr(ct_rs.as_ref())?;
        if matches!(arr, ArrOk::Str(_) | ArrOk::String(_) | ArrOk::TimeDelta(_)) {
            let arr = match_arrok!(
                arr,
                a,
                { a.view().to_object(py) },
                Str,
                String,
                #[cfg(feature = "time")]
                TimeDelta
            );
            return PyArray::from_owned_array_bound(py, arr.0).no_dim0(py);
        }
        #[cfg(feature = "time")]
        if let ArrOk::DateTime(arr) = arr {
            let arr = arr
                .view()
                .map(|v| v.into_np_datetime::<numpy::datetime::units::Microseconds>());
            return PyArray::from_owned_array_bound(py, arr.0).no_dim0(py);
        }
        match_arrok!(
            pyelement arr,
            a,
            {
                unsafe {
                    Ok(PyArray::borrow_from_array_bound(
                        &a.view().0,
                        container,
                    )
                    .no_dim0(py)?)
                }
            }
        )
    }

    #[getter]
    pub fn get_view(slf: Bound<'_, Self>, py: Python) -> PyResult<PyObject> {
        PyExpr::view_in(slf, None, py)
    }

    // #[allow(unreachable_patterns)]
    // #[pyo3(signature=(context=None))]
    // /// eval and view, used for fast test
    // pub fn eview(
    //     mut self: PyRefMut<Self>,
    //     context: Option<&PyAny>,
    //     py: Python,
    // ) -> PyResult<PyObject> {
    //     self.eval_inplace(context.clone())?;
    //     self.view_in(context, py)
    // }

    #[allow(unreachable_patterns)]
    #[pyo3(signature=(unit=None, context=None))]
    // #[cfg(feature = "time")]
    pub fn value<'py>(
        &'py mut self,
        unit: Option<&'py str>,
        context: Option<&Bound<'py, PyAny>>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let ct: PyContext<'static> = if let Some(context) = context {
            unsafe { std::mem::transmute(context.extract::<PyContext>()?) }
        } else {
            Default::default()
        };
        let (ct_rs, _obj_map) = (ct.ct, ct.obj_map);
        self.e.eval_inplace(ct_rs.clone())?;
        let data = self.e.view_data(ct_rs.as_ref()).map_err(StrError::to_py)?;
        if matches!(&data, Data::ArrVec(_)) {
            if let Data::ArrVec(_) = data {
                let arr_vec = data.view_arr_vec(ct_rs.as_ref()).map_err(StrError::to_py)?;
                let out = arr_vec
                    .into_iter()
                    .map(|arr| {
                        match_arrok!(pyelement arr, a, {
                            PyArray::from_owned_array_bound(py, a.view().to_owned().0)
                            .no_dim0(py)
                            .unwrap()
                        })
                    })
                    .collect_trusted();
                return Ok(out.into_py(py));
            }
        }
        let arr = data.view_arr(ct_rs.as_ref())?;
        if matches!(&arr, ArrOk::Str(_) | ArrOk::String(_) | ArrOk::TimeDelta(_)) {
            let arr = match_arrok!(
                arr,
                a,
                { a.view().to_object(py) },
                Str,
                String,
                #[cfg(feature = "time")]
                TimeDelta
            );
            return PyArray::from_owned_array_bound(py, arr.0).no_dim0(py);
        }
        #[cfg(feature = "time")]
        if let ArrOk::DateTime(arr) = &arr {
            match unit.unwrap_or("us").to_lowercase().as_str() {
                "ms" => {
                    let arr = arr
                        .view()
                        .map(|v| v.into_np_datetime::<numpy::datetime::units::Milliseconds>());
                    return PyArray::from_owned_array_bound(py, arr.0).no_dim0(py);
                }
                "us" => {
                    let arr = arr
                        .view()
                        .map(|v| v.into_np_datetime::<numpy::datetime::units::Microseconds>());
                    return PyArray::from_owned_array_bound(py, arr.0).no_dim0(py);
                }
                "ns" => {
                    let arr = arr
                        .view()
                        .map(|v| v.into_np_datetime::<numpy::datetime::units::Nanoseconds>());
                    return PyArray::from_owned_array_bound(py, arr.0).no_dim0(py);
                }
                _ => unimplemented!("not support datetime unit"),
            }
        }
        match_arrok!(
            pyelement arr,
            a,
            {
                Ok(PyArray::from_owned_array_bound(
                    py,
                    a.view().to_owned().0
                )
                .no_dim0(py)?)
            }
        )
    }

    #[getter]
    pub fn step(&self) -> usize {
        self.e.step_acc()
    }

    #[getter]
    pub fn get_name(&self) -> Option<String> {
        self.e.name()
    }

    #[setter]
    pub fn set_name(&mut self, name: String) {
        self.e.rename(name)
    }

    #[cfg(feature = "map")]
    #[pyo3(name = "copy")]
    pub fn deep_copy(&self) -> Self {
        let mut out = self.clone();
        out.e.deep_copy();
        out
    }

    pub(crate) fn is_owned(&self) -> bool {
        self.e.is_owned()
    }

    #[cfg(feature = "map")]
    pub unsafe fn reshape(&self, shape: &PyAny) -> PyResult<Self> {
        let shape = parse_expr_nocopy(shape)?;
        let obj = shape.obj();
        let mut out = self.clone();
        out.e.reshape(shape.e);
        Ok(out.add_obj_into(obj))
    }

    pub fn __getattr__<'py>(
        slf: PyRef<'py, Self>,
        attr: &'py str,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let attr_dict = PYEXPR_ATTRIBUTE.lock();
        let res = attr_dict.get(attr);
        if let Some(res) = res {
            let func = res.clone();
            let functools = py.import("functools")?;
            let partial = functools.getattr("partial")?;
            partial.call1((func, slf))
        } else {
            Err(PyAttributeError::new_err(format!(
                "'PyExpr' object has no attribute {attr}"
            )))
        }
    }

    #[cfg(feature = "ops")]
    fn __add__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((self.clone().e + other.e).to_py(obj).add_obj_into(obj2))
    }

    #[cfg(feature = "ops")]
    fn __radd__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((other.e + self.clone().e).to_py(obj).add_obj_into(obj2))
    }

    #[cfg(feature = "ops")]
    unsafe fn __iadd__(&mut self, other: &PyAny) {
        *self = self.__add__(other).unwrap()
    }

    #[cfg(feature = "ops")]
    fn __sub__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((self.clone().e - other.e).to_py(obj).add_obj_into(obj2))
    }

    #[cfg(feature = "ops")]
    fn __rsub__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((other.e - self.clone().e).to_py(obj).add_obj_into(obj2))
    }

    #[cfg(feature = "ops")]
    unsafe fn __isub__(&mut self, other: &PyAny) {
        *self = self.__sub__(other).unwrap()
    }

    #[cfg(feature = "ops")]
    fn __mul__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((self.clone().e * other.e).to_py(obj).add_obj_into(obj2))
    }

    #[cfg(feature = "ops")]
    unsafe fn __rmul__(&self, other: &PyAny) -> PyResult<Self> {
        self.__mul__(other)
    }

    #[cfg(feature = "ops")]
    unsafe fn __imul__(&mut self, other: &PyAny) {
        *self = self.__mul__(other).unwrap()
    }

    #[cfg(feature = "ops")]
    fn __truediv__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((self.clone().e / other.e).to_py(obj).add_obj_into(obj2))
    }

    #[cfg(feature = "ops")]
    fn __rtruediv__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((other.e / self.clone().e).to_py(obj).add_obj_into(obj2))
    }

    #[cfg(feature = "ops")]
    unsafe fn __itruediv__(&mut self, other: &PyAny) {
        *self = self.__truediv__(other).unwrap()
    }

    #[cfg(feature = "ops")]
    fn __and__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((self.clone().e & other.e).to_py(obj).add_obj_into(obj2))
    }

    #[cfg(feature = "ops")]
    fn __or__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((self.clone().e | other.e).to_py(obj).add_obj_into(obj2))
    }

    #[cfg(feature = "ops")]
    unsafe fn __rand__(&self, other: &PyAny) -> PyResult<Self> {
        self.__and__(other)
    }

    #[cfg(feature = "ops")]
    unsafe fn __iand__(&mut self, other: &PyAny) {
        *self = self.__and__(other).unwrap()
    }

    #[cfg(feature = "ops")]
    unsafe fn __ror__(&self, other: &PyAny) -> PyResult<Self> {
        self.__or__(other)
    }

    #[cfg(feature = "ops")]
    unsafe fn __ior__(&mut self, other: &PyAny) {
        *self = self.__or__(other).unwrap()
    }

    #[cfg(feature = "ops")]
    unsafe fn __richcmp__(&self, other: &PyAny, op: CompareOp, _py: Python) -> PyResult<PyExpr> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        let mut lhs = self.e.clone();
        let rhs = other.e;
        match op {
            CompareOp::Eq => lhs.eq(rhs, false),
            CompareOp::Lt => lhs.lt(rhs, false),
            CompareOp::Le => lhs.le(rhs, false),
            CompareOp::Ne => lhs.ne(rhs, false),
            CompareOp::Gt => lhs.gt(rhs, false),
            CompareOp::Ge => lhs.ge(rhs, false),
        }
        Ok(lhs.to_py(obj).add_obj_into(obj2))
    }

    #[cfg(feature = "map")]
    unsafe fn __pow__(&self, other: &PyAny, _mod: &PyAny) -> PyResult<Self> {
        self.pow_py(other, false)
    }

    #[cfg(feature = "map")]
    pub fn __round__(&self, precision: u32) -> PyResult<Self> {
        self.round(precision)
    }

    #[cfg(feature = "map")]
    #[pyo3(name="pow", signature=(other, par=false))]
    unsafe fn pow_py(&self, other: &PyAny, par: bool) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let mut out = self.clone();
        out.e.pow(other.e, par);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(feature = "map")]
    fn abs(&self) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.abs();
        Ok(out)
    }

    #[cfg(feature = "map")]
    fn __abs__(&self) -> PyResult<Self> {
        self.abs()
    }

    // unsafe fn __rshift__(&self, other: &PyAny) -> PyResult<Self> {
    //     let other = parse_expr_nocopy(other)?;
    //     let obj = other.obj();
    //     match_exprs!((&self.inner, e1, F64, I32), (other.inner, e2, I32), {
    //         (e1.clone() * e2.cast::<usize>().clone()).into()
    //     })
    // }

    // fn __lshift__(&self, #[pyo3(from_py_with = "parse_expr")] other: Self) -> PyResult<Self> {
    //     impl_py_ops!(all_and_i32 (self, other), (e1, e2), {
    //         (e1.clone() << e2.clone()).into()
    //     })
    // }

    #[cfg(feature = "ops")]
    fn __neg__(&self) -> PyResult<Self> {
        let e = self.clone();
        Ok((-e.e).to_py(self.obj()))
    }

    #[cfg(feature = "ops")]
    fn __invert__(&self) -> PyResult<Self> {
        let e = self.clone();
        Ok((!e.e).to_py(self.obj()))
        // match_exprs!(&self.inner, e, { Ok((!e.clone()).to_py(self.obj())) }, Bool)
    }

    #[cfg(feature = "blas")]
    unsafe fn __matmul__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let mut out = self.clone();
        out.e.dot(other.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(feature = "blas")]
    #[allow(clippy::redundant_clone)]
    unsafe fn __rmatmul__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = self.obj();
        let mut out = other.clone();
        out.e.dot(self.e.clone());
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(name, inplace=false))]
    pub fn alias(&mut self, name: String, inplace: bool) -> Option<Self> {
        if inplace {
            self.e.rename(name);
            None
        } else {
            let mut e = self.clone();
            e.e.rename(name);
            Some(e)
        }
    }

    #[pyo3(signature=(name, inplace=false))]
    pub fn suffix(&mut self, name: String, inplace: bool) -> Option<Self> {
        let ori_name = self.e.name().unwrap();
        if inplace {
            self.e.rename(ori_name + &name);
            None
        } else {
            let mut e = self.clone();
            e.e.rename(ori_name + &name);
            Some(e)
        }
    }

    #[pyo3(signature=(name, inplace=false))]
    pub fn prefix(&mut self, name: String, inplace: bool) -> Option<Self> {
        let ori_name = self.e.ref_name().unwrap();
        if inplace {
            self.e.rename(name + ori_name);
            None
        } else {
            let mut e = self.clone();
            e.e.rename(name + ori_name);
            Some(e)
        }
    }

    // #[classmethod]
    pub fn __setstate__(&mut self, state: &PyAny) -> PyResult<()> {
        if let Ok(state) = state.downcast::<PyDict>() {
            let name = state
                .get_item("name")?
                .unwrap()
                .extract::<Option<String>>()?;
            let arr = state.get_item("arr")?.unwrap();
            if let Some(name) = name {
                self.set_name(name);
            }
            let new_e = parse_expr_nocopy(arr)?;
            let obj = new_e.obj();
            self.e = new_e.e;
            self.add_obj(obj);
            Ok(())
        } else {
            Err(PyTypeError::new_err("state must be dict"))
        }

        // match state.extract::<&pyo3::types::PyBytes>(py) {
        //     Ok(s) => {
        //         let e: Expr<'static> = bincode::deserialize(s.as_bytes()).unwrap();
        //         Ok(())
        //     }
        //     Err(e) => Err(e),
        // }
    }

    #[pyo3(signature=())]
    pub fn __getstate__(&mut self, py: Python) -> PyResult<PyObject> {
        let name = self.e.name();
        let arr = self.value(None, None, py)?;
        let state = PyDict::new(py);
        state.set_item("name", name)?;
        state.set_item("arr", arr)?;
        Ok(state.to_object(py))
        // Ok(PyBytes::new(py, &bincode::serialize(&self.e).unwrap()).to_object(py))
    }

    #[cfg(feature = "map")]
    pub fn is_in(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let mut out = self.clone();
        out.e.is_in(other.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(feature = "map")]
    /// Returns the square root of a number.
    ///
    /// Returns NaN if self is a negative number other than -0.0.
    pub fn sqrt(&self) -> Self {
        let mut e = self.clone();
        e.e.sqrt();
        e
    }

    #[cfg(feature = "map")]
    /// Returns the cube root of each element.
    pub fn cbrt(&self) -> Self {
        let mut e = self.clone();
        e.e.cbrt();
        e
    }

    #[cfg(feature = "map")]
    /// Returns the sign of each element.
    fn sign(&self) -> Self {
        let mut e = self.clone();
        e.e.sign();
        e
    }

    #[cfg(feature = "map")]
    /// Returns the natural logarithm of each element.
    pub fn ln(&self) -> Self {
        let mut e = self.clone();
        e.e.ln();
        e
    }

    #[cfg(feature = "map")]
    /// Returns ln(1+n) (natural logarithm) more accurately than if the operations were performed separately.
    pub fn ln_1p(&self) -> Self {
        let mut e = self.clone();
        e.e.ln_1p();
        e
    }

    #[cfg(feature = "map")]
    /// Returns the base 2 logarithm of each element.
    pub fn log2(&self) -> Self {
        let mut e = self.clone();
        e.e.log2();
        e
    }

    #[cfg(feature = "map")]
    /// Returns the base 10 logarithm of each element.
    pub fn log10(&self) -> Self {
        let mut e = self.clone();
        e.e.log10();
        e
    }

    #[cfg(feature = "map")]
    /// Returns e^(self) of each element, (the exponential function).
    pub fn exp(&self) -> Self {
        let mut e = self.clone();
        e.e.exp();
        e
    }

    #[cfg(feature = "map")]
    /// Returns 2^(self) of each element.
    pub fn exp2(&self) -> Self {
        let mut e = self.clone();
        e.e.exp2();
        e
    }

    #[cfg(feature = "map")]
    /// Returns e^(self) - 1 in a way that is accurate even if the number is close to zero.
    pub fn exp_m1(&self) -> Self {
        let mut e = self.clone();
        e.e.exp_m1();
        e
    }

    #[cfg(feature = "map")]
    /// Computes the arccosine of each element. Return value is in radians in the range 0,
    /// pi or NaN if the number is outside the range -1, 1.
    pub fn acos(&self) -> Self {
        let mut e = self.clone();
        e.e.acos();
        e
    }

    #[cfg(feature = "map")]
    /// Computes the arcsine of each element. Return value is in radians in the range -pi/2,
    /// pi/2 or NaN if the number is outside the range -1, 1.
    pub fn asin(&self) -> Self {
        let mut e = self.clone();
        e.e.asin();
        e
    }

    #[cfg(feature = "map")]
    /// Computes the arctangent of each element. Return value is in radians in the range -pi/2, pi/2;
    pub fn atan(&self) -> Self {
        let mut e = self.clone();
        e.e.atan();
        e
    }

    #[cfg(feature = "map")]
    /// Computes the sine of each element (in radians).
    pub fn sin(&self) -> Self {
        let mut e = self.clone();
        e.e.sin();
        e
    }

    #[cfg(feature = "map")]
    /// Computes the cosine of each element (in radians).
    pub fn cos(&self) -> Self {
        let mut e = self.clone();
        e.e.cos();
        e
    }

    #[cfg(feature = "map")]
    /// Computes the tangent of each element (in radians).
    pub fn tan(&self) -> Self {
        let mut e = self.clone();
        e.e.tan();
        e
    }

    #[cfg(feature = "map")]
    /// Returns the smallest integer greater than or equal to `self`.
    pub fn ceil(&self) -> Self {
        let mut e = self.clone();
        e.e.ceil();
        e
    }

    #[cfg(feature = "map")]
    /// Returns the largest integer less than or equal to `self`.
    pub fn floor(&self) -> Self {
        let mut e = self.clone();
        e.e.floor();
        e
    }

    #[cfg(feature = "map")]
    /// Returns the fractional part of each element.
    pub fn fract(&self) -> Self {
        let mut e = self.clone();
        e.e.fract();
        e
    }

    #[cfg(feature = "map")]
    /// Returns the integer part of each element. This means that non-integer numbers are always truncated towards zero.
    pub fn trunc(&self) -> Self {
        let mut e = self.clone();
        e.e.trunc();
        e
    }

    #[cfg(feature = "map")]
    /// Returns true if this number is neither infinite nor NaN
    pub fn is_finite(&self) -> Self {
        let mut e = self.clone();
        e.e.is_finite();
        e
    }

    #[cfg(feature = "map")]
    /// Returns true if this value is positive infinity or negative infinity, and false otherwise.
    pub fn is_inf(&self) -> Self {
        let mut e = self.clone();
        e.e.is_infinite();
        e
    }

    #[cfg(feature = "map")]
    /// Returns the logarithm of the number with respect to an arbitrary base.
    ///
    /// The result might not be correctly rounded owing to implementation details;
    /// `self.log2()` can produce more accurate results for base 2,
    /// and `self.log10()` can produce more accurate results for base 10.
    pub fn log(&self, base: f64) -> Self {
        let mut e = self.clone();
        e.e.log(base);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(axis=0, par=false))]
    pub fn first(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.first(axis, par);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(axis=0, par=false))]
    pub fn last(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.last(axis, par);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(axis=0, par=false))]
    pub fn valid_first(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.valid_first(axis, par);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(axis=0, par=false))]
    pub fn valid_last(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.valid_last(axis, par);
        e
    }

    #[cfg(feature = "map")]
    #[pyo3(signature=(n=1, fill=None, axis=0, par=false))]
    pub fn shift(
        &self,
        n: i32,
        fill: Option<&PyAny>,
        axis: i32,
        par: bool,
        _py: Python,
    ) -> PyResult<Self> {
        let fill = if let Some(fill) = fill {
            if fill.is_none() {
                None
            } else {
                Some(parse_expr_nocopy(fill)?)
            }
        } else {
            None
        };
        let obj = fill.as_ref().map(|e| e.obj());
        let mut out = self.clone();
        out.e.shift(n.into(), fill.map(|f| f.e), axis, par);
        if let Some(obj) = obj {
            Ok(out.add_obj_into(obj))
        } else {
            Ok(out)
        }
    }

    #[cfg(feature = "map")]
    #[pyo3(signature=(n=1, fill=None, axis=0, par=false))]
    pub fn diff(&self, n: i32, fill: Option<&PyAny>, axis: i32, par: bool) -> PyResult<Self> {
        let fill = if let Some(fill) = fill {
            if fill.is_none() {
                None
            } else {
                Some(parse_expr_nocopy(fill)?)
            }
        } else {
            None
        };
        let obj = fill.as_ref().map(|e| e.obj());
        let mut e = self.clone();
        e.e.diff(n.into(), fill.map(|f| f.e), axis, par);
        if let Some(obj) = obj {
            Ok(e.add_obj_into(obj))
        } else {
            Ok(e)
        }
    }

    #[cfg(feature = "map")]
    #[pyo3(signature=(n=1, axis=0, par=false))]
    pub fn pct_change(&self, n: i32, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.pct_change(n, axis, par);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(axis=0, par=false))]
    pub fn count_nan(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.count_nan(axis, par);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(axis=0, par=false))]
    pub fn count_notnan(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.count_notnan(axis, par);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(value, axis=0, par=false))]
    pub fn count_value(&self, value: &PyAny, axis: i32, par: bool) -> PyResult<Self> {
        let value = parse_expr_nocopy(value)?;
        let obj = value.obj();
        let mut e = self.clone();
        e.e.count_value(value.e, axis, par);
        Ok(e.add_obj_into(obj))
    }

    #[cfg(all(feature = "map", feature = "agg"))]
    #[pyo3(signature=(mask, axis=None, par=false))]
    pub unsafe fn filter(&self, mask: &PyAny, axis: Option<&PyAny>, par: bool) -> PyResult<Self> {
        let mask = parse_expr_nocopy(mask)?;
        let axis = if let Some(axis) = axis {
            parse_expr_nocopy(axis)?
        } else {
            0.into_pyexpr()
        };
        let obj = mask.obj();
        let obj2 = axis.obj();
        let mut out = self.clone();
        out.e.filter(mask.e, axis.e, par);
        Ok(out.add_obj_into(obj).add_obj_into(obj2))
    }

    #[cfg(all(feature = "map", feature = "agg"))]
    #[pyo3(signature=(axis=None, how=DropNaMethod::Any, par=false))]
    pub unsafe fn dropna(
        &self,
        axis: Option<&PyAny>,
        how: DropNaMethod,
        par: bool,
    ) -> PyResult<Self> {
        let axis = if let Some(axis) = axis {
            parse_expr_nocopy(axis)?
        } else {
            0.into_pyexpr()
        };
        let obj = axis.obj();
        let mut e = self.clone();
        e.e.dropna(axis.e, how, par);
        Ok(e.add_obj_into(obj))
    }

    #[cfg(feature = "map")]
    pub fn is_nan(&self) -> Self {
        let mut out = self.clone();
        out.e.is_nan();
        out
    }

    #[cfg(feature = "map")]
    pub fn not_nan(&self) -> Self {
        let mut out = self.clone();
        out.e.not_nan();
        out
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(axis=0, par=false))]
    pub fn median(&self, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.median(axis, par);
        out
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(axis=0, par=false))]
    pub fn all(&self, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.all(axis, par);
        out
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(axis=0, par=false))]
    pub fn any(&self, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.any(axis, par);
        out
    }

    #[cfg(feature = "map")]
    /// Return a view of the diagonal elements of the array.
    ///
    /// The diagonal is simply the sequence indexed by (0, 0, .., 0), (1, 1, ..., 1) etc as long as all axes have elements.
    ///
    /// Safety
    /// the data for the array view should exist
    pub unsafe fn diag(&self) -> Self {
        let mut out = self.clone();
        out.e.diag();
        out
    }

    #[cfg(feature = "map")]
    /// Insert new array axis at axis and return the result.
    pub unsafe fn insert_axis(&self, axis: i32) -> Self {
        let mut out = self.clone();
        out.e.insert_axis(axis);
        out
    }

    #[cfg(feature = "map")]
    /// Remove new array axis at axis and return the result.
    pub unsafe fn remove_axis(&self, axis: i32) -> Self {
        let mut out = self.clone();
        out.e.remove_axis(axis);
        out
    }

    #[cfg(feature = "map")]
    /// Return a transposed view of the array.
    pub unsafe fn t(&self) -> Self {
        let mut out = self.clone();
        out.e.t();
        out
    }

    #[cfg(feature = "map")]
    /// Swap axes ax and bx.
    ///
    /// This does not move any data, it just adjusts the array’s dimensions and strides.
    pub unsafe fn swap_axes(&self, ax: i32, bx: i32) -> Self {
        let mut out = self.clone();
        out.e.swap_axes(ax, bx);
        out
    }

    #[cfg(feature = "map")]
    /// Permute the axes.
    ///
    /// This does not move any data, it just adjusts the array’s dimensions and strides.
    ///
    ///i in the j-th place in the axes sequence means self's i-th axis becomes self.permuted_axes()'s j-th axis
    pub unsafe fn permuted_axes(&self, axes: &PyAny) -> PyResult<Self> {
        let axes = parse_expr_nocopy(axes)?;
        let obj = axes.obj();
        let mut out = self.clone();
        out.e.permuted_axes(axes.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(feature = "agg")]
    /// Return the number of dimensions (axes) in the array
    pub fn ndim(&self) -> Self {
        let mut out = self.clone();
        out.e.ndim();
        out
    }

    #[cfg(feature = "agg")]
    #[getter]
    /// Return the shape of the array as a usize Expr.
    pub fn shape(&self) -> Self {
        let mut out = self.clone();
        out.e.shape();
        out
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(axis=0, par=false))]
    pub fn max(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.max(axis, par);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(axis=0, par=false))]
    pub fn min(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.min(axis, par);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn sum(&self, stable: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.sum(stable, axis, par);
        e
    }

    #[cfg(feature = "map")]
    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn cumsum(&self, stable: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.cumsum(stable, axis, par);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(axis=0, par=false))]
    pub fn prod(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.prod(axis, par);
        e
    }

    #[cfg(feature = "map")]
    #[pyo3(signature=(axis=0, par=false))]
    pub fn cumprod(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.cumprod(axis, par);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(min_periods=1, stable=false, axis=0, par=false))]
    pub fn mean(&self, min_periods: usize, stable: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.mean(min_periods, stable, axis, par);
        e
    }

    #[cfg(all(feature = "map", feature = "agg"))]
    #[pyo3(signature=(min_periods=3, stable=false, axis=0, par=false, _warning=true))]
    pub fn zscore(
        &self,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
        _warning: bool,
        _py: Python,
    ) -> PyResult<Self> {
        // if warning && !self.is_float().unwrap() {
        //     let warnings = py.import("warnings")?;
        //     warnings.call_method1(
        //         "warn",
        //         ("The dtype of input is not Float, so note that the result is not float too",),
        //     )?;
        // }
        let mut e = self.clone();
        e.e.zscore(min_periods, stable, axis, par);
        Ok(e)
    }

    #[cfg(all(feature = "map", feature = "agg"))]
    #[pyo3(signature=(method=WinsorizeMethod::Quantile, method_params=0.01, stable=false, axis=0, par=false))]
    #[allow(clippy::too_many_arguments)]
    pub fn winsorize(
        &self,
        method: WinsorizeMethod,
        method_params: Option<f64>,
        stable: bool,
        axis: i32,
        par: bool,
        // _warning: bool,
        _py: Python,
    ) -> PyResult<Self> {
        let mut e = self.clone();
        e.e.winsorize(method, method_params, stable, axis, par);
        Ok(e)
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(min_periods=1, stable=false, axis=0, par=false))]
    pub fn var(&self, min_periods: usize, stable: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.var(min_periods, stable, axis, par);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(min_periods=2, stable=false, axis=0, par=false))]
    pub fn std(&self, min_periods: usize, stable: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.std(min_periods, stable, axis, par);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(min_periods=3, stable=false, axis=0, par=false))]
    pub fn skew(&self, min_periods: usize, stable: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.skew(min_periods, stable, axis, par);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(min_periods=4, stable=false, axis=0, par=false))]
    pub fn kurt(&self, min_periods: usize, stable: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.kurt(min_periods, stable, axis, par);
        e
    }

    #[cfg(all(feature = "map", feature = "agg"))]
    #[pyo3(signature=(pct=false, rev=false, axis=0, par=false))]
    pub fn rank(&self, pct: bool, rev: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.rank(pct, rev, axis, par);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(q, method=QuantileMethod::Linear, axis=0, par=false))]
    pub fn quantile(&self, q: f64, method: QuantileMethod, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.quantile(q, method, axis, par);
        e
    }

    #[cfg(feature = "map")]
    #[pyo3(signature=(rev=false, axis=0, par=false))]
    pub fn argsort(&self, rev: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.argsort(rev, axis, par);
        e
    }

    #[cfg(all(feature = "map", feature = "agg"))]
    #[pyo3(signature=(kth, sort=true, rev=false, axis=0, par=false))]
    pub fn arg_partition(&self, kth: usize, sort: bool, rev: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.arg_partition(kth, sort, rev, axis, par);
        e
    }

    #[cfg(all(feature = "map", feature = "agg"))]
    #[pyo3(signature=(kth, sort=true, rev=false, axis=0, par=false))]
    pub fn partition(&self, kth: usize, sort: bool, rev: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.partition(kth, sort, rev, axis, par);
        e
    }

    #[cfg(all(feature = "map", feature = "agg"))]
    #[pyo3(signature=(group, rev=false, axis=0, par=false))]
    pub fn split_group(&self, group: usize, rev: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.split_group(group, rev, axis, par);
        e
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(other, min_periods=3, stable=false, axis=0, par=false))]
    pub unsafe fn cov(
        &self,
        other: &PyAny,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let mut e = self.clone();
        e.e.cov(other.e, min_periods, stable, axis, par);
        Ok(e.add_obj_into(obj))
    }

    #[cfg(feature = "agg")]
    #[pyo3(signature=(other, method=CorrMethod::Pearson, min_periods=3, stable=false, axis=0, par=false))]
    pub unsafe fn corr(
        &self,
        other: &PyAny,
        method: CorrMethod,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let mut e = self.clone();
        e.e.corr(other.e, method, min_periods, stable, axis, par);
        Ok(e.add_obj_into(obj))
    }

    #[cfg(feature = "groupby")]
    #[pyo3(signature=(keep="first".to_string()))]
    pub fn _get_sorted_unique_idx(&self, keep: String) -> Self {
        let mut out = self.clone();
        out.e.get_sorted_unique_idx(keep);
        out
        // match_exprs!(&self.inner, expr, {
        //     expr.clone().get_sorted_unique_idx(keep).to_py(self.obj())
        // })
    }

    #[cfg(feature = "groupby")]
    pub fn sorted_unique(&self) -> Self {
        let mut out = self.clone();
        out.e.sorted_unique();
        out
    }

    #[cfg(feature = "groupby")]
    #[pyo3(signature=(others=None, keep="first".to_string()))]
    pub fn _get_unique_idx(&self, others: Option<&PyAny>, keep: String) -> PyResult<Self> {
        let (obj_vec, others) = if let Some(others) = others {
            let others = unsafe { parse_expr_list(others, false)? };
            let obj_vec = others.iter().map(|e| e.obj()).collect_trusted();
            (obj_vec, Some(others))
        } else {
            (vec![], None)
        };
        let others = others.map(|v| v.into_iter().map(|e| e.e).collect_trusted());
        let mut out = self.clone();
        out.e.get_unique_idx(others, keep);
        Ok(out.add_obj_vec_into(obj_vec))
    }

    #[cfg(feature = "groupby")]
    pub unsafe fn _get_left_join_idx(&self, left_other: &PyAny, right: &PyAny) -> PyResult<Self> {
        let left_other = if left_other.is_none() {
            None
        } else {
            Some(parse_expr_list(left_other, false)?)
        };
        let right = parse_expr_list(right, false)?;
        let obj_vec1 = left_other
            .as_ref()
            .map(|lo| lo.iter().map(|e| e.obj()).collect_trusted());
        let obj_vec2 = right.iter().map(|e| e.obj()).collect_trusted();
        let left_other = left_other.map(|lo| lo.into_iter().map(|e| e.e).collect_trusted());
        let right = right.into_iter().map(|e| e.e).collect_trusted();
        let mut out = self.clone();
        out.e.get_left_join_idx(left_other, right);
        if let Some(obj_vec1) = obj_vec1 {
            Ok(out.add_obj_vec_into(obj_vec1).add_obj_vec_into(obj_vec2))
        } else {
            Ok(out.add_obj_vec_into(obj_vec2))
        }
    }

    #[cfg(feature = "groupby")]
    #[pyo3(signature=(left_other, right, sort=true, rev=false, split=true))]
    pub unsafe fn _get_outer_join_idx(
        &self,
        left_other: &PyAny,
        right: &PyAny,
        sort: bool,
        rev: bool,
        split: bool,
        py: Python,
    ) -> PyResult<PyObject> {
        let left_other = if left_other.is_none() {
            None
        } else {
            Some(parse_expr_list(left_other, false)?)
        };
        let right = parse_expr_list(right, false)?;
        let obj_vec1 = left_other
            .as_ref()
            .map(|lo| lo.iter().map(|e| e.obj()).collect_trusted());
        let obj_vec2 = right.iter().map(|e| e.obj()).collect_trusted();
        let left_other = left_other.map(|lo| lo.into_iter().map(|e| e.e).collect_trusted());
        let right = right.into_iter().map(|e| e.e).collect_trusted();
        let len = right.len() + 2;
        let mut out = self.clone();
        out.e.get_outer_join_idx(left_other, right, sort, rev);
        if let Some(obj_vec1) = obj_vec1 {
            out.add_obj_vec(obj_vec1).add_obj_vec(obj_vec2);
        } else {
            out.add_obj_vec(obj_vec2);
        }
        if split {
            let obj = out.obj();
            let out = out
                .e
                .split_vec_base(len)
                .into_iter()
                .map(|e| e.to_py(obj.clone()))
                .collect_trusted();
            Ok(out.into_py(py))
        } else {
            Ok(out.into_py(py))
        }
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_argmin(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.ts_argmin(window, min_periods, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_argmax(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.ts_argmax(window, min_periods, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_min(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.ts_min(window, min_periods, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_max(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.ts_max(window, min_periods, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, pct=false, rev=false, axis=0, par=false))]
    pub fn ts_rank(
        &self,
        window: usize,
        min_periods: usize,
        pct: bool,
        rev: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e.ts_rank(window, min_periods, pct, rev, axis, par);
        // if !pct {
        //     out.e.ts_rank(window, min_periods, axis, par);
        // } else {
        //     out.e.ts_rank_pct(window, min_periods, axis, par);
        // }
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_prod(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.ts_prod(window, min_periods, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_prod_mean(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.ts_prod_mean(window, min_periods, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_minmaxnorm(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.ts_minmaxnorm(window, min_periods, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(other, window, min_periods=1, stable=false, axis=0, par=false))]
    pub unsafe fn ts_cov(
        &self,
        other: &PyAny,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let mut out = self.clone();
        out.e
            .ts_cov(other.e, window, min_periods, stable, axis, par);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(other, window, min_periods=1, stable=false, axis=0, par=false))]
    pub unsafe fn ts_corr(
        &self,
        other: &PyAny,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let mut out = self.clone();
        out.e
            .ts_corr(other.e, window, min_periods, stable, axis, par);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(other, window, min_periods=1, stable=false, axis=0, par=false))]
    pub unsafe fn ts_regx_alpha(
        &self,
        other: &PyAny,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let mut out = self.clone();
        out.e
            .ts_regx_alpha(other.e, window, min_periods, stable, axis, par);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(other, window, min_periods=1, stable=false, axis=0, par=false))]
    pub unsafe fn ts_regx_beta(
        &self,
        other: &PyAny,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let mut out = self.clone();
        out.e
            .ts_regx_beta(other.e, window, min_periods, stable, axis, par);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(other, window, min_periods=1, axis=0, par=false))]
    pub unsafe fn ts_regx_resid_mean(
        &self,
        other: &PyAny,
        window: usize,
        min_periods: usize,
        axis: i32,
        par: bool,
    ) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let mut out = self.clone();
        out.e
            .ts_regx_resid_mean(other.e, window, min_periods, axis, par);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(other, window, min_periods=1, axis=0, par=false))]
    pub unsafe fn ts_regx_resid_std(
        &self,
        other: &PyAny,
        window: usize,
        min_periods: usize,
        axis: i32,
        par: bool,
    ) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let mut out = self.clone();
        out.e
            .ts_regx_resid_std(other.e, window, min_periods, axis, par);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(other, window, min_periods=1, axis=0, par=false))]
    pub unsafe fn ts_regx_resid_skew(
        &self,
        other: &PyAny,
        window: usize,
        min_periods: usize,
        axis: i32,
        par: bool,
    ) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let mut out = self.clone();
        out.e
            .ts_regx_resid_skew(other.e, window, min_periods, axis, par);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_sum(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e.ts_sum(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_sma(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e.ts_sma(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_mean(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        self.ts_sma(window, min_periods, stable, axis, par)
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_ewm(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e.ts_ewm(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_wma(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e.ts_wma(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_std(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e.ts_std(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_var(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e.ts_var(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_skew(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e.ts_skew(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_kurt(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e.ts_kurt(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_stable(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e.ts_stable(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_meanstdnorm(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e.ts_meanstdnorm(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_reg(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e.ts_reg(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_tsf(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e.ts_tsf(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_reg_slope(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e.ts_reg_slope(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_reg_resid_mean(
        &self,
        window: usize,
        min_periods: usize,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e.ts_reg_resid_mean(window, min_periods, axis, par);
        out
    }

    #[cfg(feature = "rolling")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_reg_intercept(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        let mut out = self.clone();
        out.e
            .ts_reg_intercept(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "map")]
    #[pyo3(signature=(method=FillMethod::Vfill, value=None, axis=0, par=false))]
    pub fn fillna(
        &self,
        method: FillMethod,
        value: Option<&PyAny>,
        axis: i32,
        par: bool,
    ) -> PyResult<Self> {
        let mut out = self.clone();
        let (value, obj) = if let Some(value) = value {
            let v = parse_expr_nocopy(value)?;
            let obj = v.obj();
            (Some(v.e), obj)
        } else {
            (None, None)
        };
        out.e.fillna(method, value, axis, par);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(feature = "map")]
    #[pyo3(signature=(min, max, axis=0, par=false))]
    pub fn clip(&self, min: f64, max: f64, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.clip(min.into(), max.into(), axis, par);
        out
    }

    #[cfg(all(feature = "map", feature = "agg"))]
    #[pyo3(signature=(slc, axis=None, check=true))]
    pub unsafe fn select(&self, slc: &PyAny, axis: Option<&PyAny>, check: bool) -> PyResult<Self> {
        let axis = if let Some(axis) = axis {
            parse_expr_nocopy(axis)?
        } else {
            0.into_pyexpr()
        };
        let slc = parse_expr_nocopy(slc)?;
        let (obj1, obj2) = (slc.obj(), axis.obj());
        let mut out = self.clone();
        out.e.select(slc.e, axis.e, check);
        Ok(out.add_obj_into(obj1).add_obj_into(obj2))
    }

    #[cfg(feature = "blas")]
    #[pyo3(signature=(df, loc=None, scale=None))]
    pub unsafe fn t_cdf(&self, df: &PyAny, loc: Option<f64>, scale: Option<f64>) -> PyResult<Self> {
        let df = parse_expr_nocopy(df)?;
        let obj = df.obj();
        let mut out = self.clone();
        out.e.t_cdf(df.e, loc, scale);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(feature = "blas")]
    #[pyo3(signature=(mean=None, std=None))]
    pub unsafe fn norm_cdf(&self, mean: Option<f64>, std: Option<f64>) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.norm_cdf(mean, std);
        Ok(out)
    }

    #[cfg(feature = "blas")]
    #[pyo3(signature=(df1, df2))]
    pub unsafe fn f_cdf(&self, df1: &PyAny, df2: &PyAny) -> PyResult<Self> {
        let df1 = parse_expr_nocopy(df1)?;
        let df2 = parse_expr_nocopy(df2)?;
        let (obj1, obj2) = (df1.obj(), df2.obj());
        let mut out = self.clone();
        out.e.f_cdf(df1.e, df2.e);
        Ok(out.add_obj_into(obj1).add_obj_into(obj2))
    }

    #[cfg(all(feature = "map", feature = "agg"))]
    #[pyo3(signature=(by, rev=false, return_idx=false))]
    pub fn sort(&self, by: &PyAny, rev: bool, return_idx: bool) -> PyResult<Self> {
        let by = unsafe { parse_expr_list(by, false) }?;
        let obj_vec = by.iter().map(|e| e.obj()).collect_trusted();
        let by = by.into_iter().map(|e| e.e).collect_trusted();
        if return_idx {
            let out = Expr::get_sort_idx(by, rev);
            Ok(out.to_py(None).add_obj_vec_into(obj_vec))
        } else {
            let mut out = self.clone();
            out.e.sort(by, rev);
            Ok(out.add_obj_vec_into(obj_vec))
        }
    }

    pub fn cast(&self, ty: &PyAny, py: Python) -> PyResult<Self> {
        if let Ok(ty_name) = ty.extract::<&str>() {
            self.cast_by_str(ty_name, py)
        } else if let Ok(py_type) = ty.extract::<&pyo3::types::PyType>() {
            self.cast_by_str(&py_type.name().unwrap(), py)
        } else {
            unimplemented!("Incorrect type for casting")
        }
    }

    #[cfg(all(feature = "map", feature = "agg"))]
    #[pyo3(signature=(mask, value, axis=0, par=false, inplace=false))]
    pub unsafe fn put_mask(
        &mut self,
        mask: &PyAny,
        value: &PyAny,
        axis: i32,
        par: bool,
        inplace: bool,
        _py: Python,
    ) -> PyResult<Option<Self>> {
        let (mask, value) = (parse_expr_nocopy(mask)?, parse_expr_nocopy(value)?);
        let (obj1, obj2) = (mask.obj(), value.obj());
        if !inplace {
            let mut e = self.clone();
            e.e.put_mask(mask.e, value.e, axis, par);
            e.add_obj(obj1).add_obj(obj2);
            Ok(Some(e))
        } else {
            self.e.put_mask(mask.e, value.e, axis, par);
            self.add_obj(obj1).add_obj(obj2);
            Ok(None)
        }
    }

    #[cfg(feature = "map")]
    #[pyo3(signature=(mask, value, par=false))]
    pub unsafe fn where_(&self, mask: &PyAny, value: &PyAny, par: bool) -> PyResult<PyExpr> {
        let mask = parse_expr_nocopy(mask)?;
        let value = parse_expr_nocopy(value)?;
        where_(self.clone(), mask, value, par)
    }

    #[cfg(feature = "map")]
    #[pyo3(signature=(con, then))]
    pub unsafe fn if_then(&self, con: &PyAny, then: &PyAny) -> PyResult<PyExpr> {
        let con = parse_expr_nocopy(con)?;
        let then = parse_expr_nocopy(then)?;
        let con_obj = con.obj();
        let then_obj = then.obj();
        let mut out = self.clone();
        out.e.if_then(con.e, then.e);
        Ok(out.add_obj_into(con_obj).add_obj_into(then_obj))
    }

    #[cfg(feature = "blas")]
    pub unsafe fn lstsq(&self, y: &PyAny) -> PyResult<Self> {
        let y = parse_expr_nocopy(y)?;
        let obj = y.obj();
        let mut out = self.clone();
        out.e.lstsq(y.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(feature = "blas")]
    pub fn params(&self) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.params();
        Ok(out)
    }

    #[cfg(feature = "blas")]
    pub fn singular_values(&self) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.singular_values();
        Ok(out)
    }

    #[cfg(feature = "blas")]
    pub fn ols_rank(&self) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.ols_rank();
        Ok(out)
    }

    #[cfg(feature = "blas")]
    pub fn sse(&self) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.sse();
        Ok(out)
    }

    #[cfg(feature = "blas")]
    pub fn fitted_values(&self) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.fitted_values();
        Ok(out)
    }

    #[cfg(feature = "blas")]
    #[pyo3(signature=(full=true, calc_uvt=true, split=true))]
    pub fn svd(&self, full: bool, calc_uvt: bool, split: bool, py: Python) -> PyResult<PyObject> {
        let mut out = self.clone();
        if split {
            let len = 1 + 2 * calc_uvt as usize;
            out.e.svd(full, calc_uvt);
            let mut out = out
                .e
                .split_vec_base(len)
                .into_iter()
                .map(|e| e.to_py(self.obj()))
                .collect_trusted();
            if out.len() == 1 {
                Ok(out.pop().unwrap().into_py(py))
            } else {
                Ok(out.into_py(py))
            }
        } else {
            out.e.svd(full, calc_uvt);
            Ok(out.into_py(py))
        }
    }

    #[cfg(feature = "blas")]
    #[pyo3(signature=(return_s=false, r_cond=None, split=true))]
    pub fn pinv(
        &self,
        return_s: bool,
        r_cond: Option<f64>,
        split: bool,
        py: Python,
    ) -> PyResult<PyObject> {
        let mut out = self.clone();
        if split {
            out.e.pinv(r_cond, return_s);
            let mut out = out
                .e
                .split_vec_base(1 + return_s as usize)
                .into_iter()
                .map(|e| e.to_py(self.obj()))
                .collect_trusted();
            if out.len() == 1 {
                Ok(out.pop().unwrap().into_py(py))
            } else {
                Ok(out.into_py(py))
            }
        } else {
            out.e.pinv(r_cond, return_s);
            Ok(out.into_py(py))
        }
    }

    #[cfg(feature = "map")]
    pub unsafe fn broadcast(&self, shape: &PyAny) -> PyResult<Self> {
        let shape = parse_expr_nocopy(shape)?;
        let obj = shape.obj();
        let mut out = self.clone();
        out.e.broadcast(shape.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "map", feature = "agg"))]
    pub unsafe fn broadcast_with(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let shape = other.shape();
        let obj = shape.obj();
        let mut out = self.clone();
        out.e.broadcast_with(other.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(feature = "concat")]
    #[pyo3(name = "concat")]
    #[pyo3(signature=(other, axis=0))]
    pub unsafe fn concat_py(&self, other: &PyAny, axis: i32) -> PyResult<Self> {
        let other = parse_expr_list(other, false)?;
        if other.is_empty() {
            return Ok(self.clone());
        }
        let obj_vec = other.iter().map(|e| e.obj()).collect_trusted();
        let other = other.into_iter().map(|e| e.e).collect_trusted();
        let mut out = self.clone();
        out.e.concat(other, axis);
        Ok(out.add_obj_vec_into(obj_vec))
    }

    #[cfg(feature = "concat")]
    #[pyo3(name = "stack")]
    #[pyo3(signature=(other, axis=0))]
    pub unsafe fn stack_py(&self, other: &PyAny, axis: i32) -> PyResult<Self> {
        let other = parse_expr_list(other, false)?;
        if other.is_empty() {
            return Ok(self.clone());
        }
        let obj_vec = other.iter().map(|e| e.obj()).collect_trusted();
        let other = other.into_iter().map(|e| e.e).collect_trusted();
        let mut out = self.clone();
        out.e.stack(other, axis);
        Ok(out.add_obj_vec_into(obj_vec))
    }

    #[cfg(feature = "time")]
    pub fn offset_by(&self, delta: &str) -> PyResult<Self> {
        let delta: Expr<'static> = TimeDelta::parse(delta).into();
        let out = self.clone();
        Ok((out.e + delta).to_py(self.obj()))
    }

    #[cfg(all(feature = "map", feature = "time"))]
    pub fn strptime(&self, fmt: String) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.strptime(fmt);
        Ok(out)
    }

    #[cfg(all(feature = "map", feature = "time"))]
    pub fn strftime(&self, fmt: Option<String>) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.strftime(fmt);
        Ok(out)
    }

    #[cfg(feature = "map")]
    pub fn round(&self, precision: u32) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.round(precision);
        Ok(out)
    }

    // pub fn round_string(&self, precision: usize) -> PyResult<Self> {
    //     let out = match_exprs!(numeric & self.inner, e, {
    //         e.clone()
    //             .chain_view_f::<_, String>(
    //                 move |arr| Ok(arr.map(|v| format!("{v:.precision$}")).into()),
    //                 RefType::False,
    //             )
    //             .to_py(self.obj())
    //     });
    //     Ok(out)
    // }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    pub unsafe fn _get_fix_window_rolling_idx(&self, window: &PyAny) -> PyResult<Self> {
        let mut length = self.clone();
        length.e.len();
        let mut window = parse_expr(window, true)?;
        Expr::get_fix_window_rolling_idx(&mut window.e, length.e);
        Ok(window.add_obj_into(self.obj()))
    }

    #[pyo3(signature=(duration, start_by=RollingTimeStartBy::Full))]
    #[cfg(all(feature = "rolling", feature = "time", feature = "agg"))]
    pub unsafe fn _get_time_rolling_idx(
        &self,
        duration: &str,
        start_by: RollingTimeStartBy,
    ) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.get_time_rolling_idx(duration, start_by);
        Ok(out)
    }

    #[cfg(feature = "groupby")]
    #[pyo3(signature=(others=None, sort=true, par=false))]
    pub unsafe fn _get_group_by_idx(
        &self,
        others: Option<&PyAny>,
        sort: bool,
        par: bool,
    ) -> PyResult<Self> {
        let others = if let Some(others) = others {
            parse_expr_list(others, false)?
        } else {
            vec![]
        };
        let others_obj_vec = others.iter().map(|e| e.obj()).collect_trusted();
        let others = others.into_iter().map(|e| e.e).collect_trusted();
        let mut out = self.clone();
        out.e.get_group_by_idx(others, sort, par);
        Ok(out.add_obj_vec_into(others_obj_vec))
    }

    #[cfg(feature = "groupby")]
    #[pyo3(signature=(agg_expr, idxs, others=None))]
    pub unsafe fn apply_with_vecusize(
        &self,
        agg_expr: &PyAny,
        idxs: &PyAny,
        others: Option<&PyAny>,
    ) -> PyResult<Self> {
        let agg_expr = parse_expr_nocopy(agg_expr)?;
        let idxs = parse_expr_nocopy(idxs)?;
        let others = if let Some(others) = others {
            parse_expr_list(others, false)?
        } else {
            vec![]
        };
        let obj1 = agg_expr.obj();
        let obj2 = idxs.obj();
        let others_obj_vec = others.iter().map(|e| e.obj()).collect_trusted();
        let others = others.into_iter().map(|e| e.e).collect_trusted();
        let mut out = self.clone();
        if let Some(name) = agg_expr.e.name() {
            out.e.rename(name);
        }
        out.e.apply_with_vecusize(agg_expr.e, idxs.e, others);
        out.add_obj(obj1).add_obj(obj2).add_obj_vec(others_obj_vec);
        Ok(out)
    }

    #[pyo3(signature=(agg_expr, roll_start, others=None))]
    #[cfg(all(feature = "rolling", feature = "concat"))]
    pub unsafe fn rolling_apply_with_start(
        &self,
        agg_expr: &PyAny,
        roll_start: &PyAny,
        others: Option<&PyAny>,
    ) -> PyResult<Self> {
        let agg_expr = parse_expr_nocopy(agg_expr)?;
        let roll_start = parse_expr_nocopy(roll_start)?;
        let others = if let Some(others) = others {
            Some(parse_expr_list(others, false)?)
        } else {
            None
        };
        let obj1 = agg_expr.obj();
        let obj2 = roll_start.obj();
        let others_obj_vec = if let Some(others) = &others {
            others.iter().map(|e| e.obj()).collect_trusted()
        } else {
            vec![]
        };
        let others = if let Some(others) = others {
            others.into_iter().map(|e| e.e).collect_trusted()
        } else {
            vec![]
        };
        let mut out = self.clone();
        if let Some(name) = agg_expr.e.name() {
            out.e.rename(name);
        }
        out.e
            .rolling_apply_with_start(agg_expr.e, roll_start.e, others, false);
        out.add_obj(obj1).add_obj(obj2).add_obj_vec(others_obj_vec);
        Ok(out)
    }

    #[pyo3(signature=(duration, closed="right".to_owned(), split=true))]
    #[cfg(all(feature = "time", feature = "groupby"))]
    pub unsafe fn _get_group_by_time_idx(
        &self,
        duration: &str,
        closed: String,
        split: bool,
        py: Python,
    ) -> PyResult<PyObject> {
        let mut out = self.clone();
        out.e.get_group_by_time_idx(duration, closed);
        if split {
            let out = out
                .e
                .split_vec_base(2)
                .into_iter()
                .map(|e| e.to_py(self.obj()))
                .collect_trusted();
            Ok(out.into_py(py))
        } else {
            Ok(out.into_py(py))
        }
    }

    #[cfg(all(feature = "agg", feature = "groupby"))]
    #[pyo3(signature=(agg_expr, idx, others=None))]
    pub unsafe fn group_by_startidx(
        &self,
        agg_expr: &PyAny,
        idx: &PyAny,
        others: Option<&PyAny>,
    ) -> PyResult<Self> {
        let agg_expr = parse_expr_nocopy(agg_expr)?;
        let idx = parse_expr_nocopy(idx)?;
        let others = if let Some(others) = others {
            Some(parse_expr_list(others, false)?)
        } else {
            None
        };
        let obj1 = agg_expr.obj();
        let obj2 = idx.obj();
        let others_obj_vec = if let Some(others) = &others {
            others.iter().map(|e| e.obj()).collect_trusted()
        } else {
            vec![]
        };
        let others = if let Some(others) = others {
            others.into_iter().map(|e| e.e).collect_trusted()
        } else {
            vec![]
        };
        let mut out = self.clone();
        if let Some(name) = agg_expr.e.name() {
            out.e.rename(name);
        }
        out.e.group_by_startidx(agg_expr.e, idx.e, others);
        out.add_obj(obj1).add_obj(obj2).add_obj_vec(others_obj_vec);
        Ok(out)
    }

    #[cfg(all(feature = "groupby", feature = "agg"))]
    #[pyo3(signature=(idx, min_periods=1, stable=false))]
    pub unsafe fn _group_by_startidx_mean(
        &self,
        idx: &PyAny,
        min_periods: usize,
        stable: bool,
    ) -> PyResult<Self> {
        let idx = parse_expr_nocopy(idx)?;
        let obj = idx.obj();
        let mut out = self.clone();
        out.e.group_by_startidx_mean(idx.e, min_periods, stable);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "groupby", feature = "agg"))]
    #[pyo3(signature=(idx, stable=false))]
    pub unsafe fn _group_by_startidx_sum(&self, idx: &PyAny, stable: bool) -> PyResult<Self> {
        let idx = parse_expr_nocopy(idx)?;
        let obj = idx.obj();
        let mut out = self.clone();
        out.e.group_by_startidx_sum(idx.e, stable);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "groupby", feature = "agg"))]
    #[pyo3(signature=(idx, min_periods=2, stable=false))]
    pub unsafe fn _group_by_startidx_std(
        &self,
        idx: &PyAny,
        min_periods: usize,
        stable: bool,
    ) -> PyResult<Self> {
        let idx = parse_expr_nocopy(idx)?;
        let obj = idx.obj();
        let mut out = self.clone();
        out.e.group_by_startidx_std(idx.e, min_periods, stable);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "groupby", feature = "agg"))]
    #[pyo3(signature=(idx, min_periods=2, stable=false))]
    pub unsafe fn _group_by_startidx_var(
        &self,
        idx: &PyAny,
        min_periods: usize,
        stable: bool,
    ) -> PyResult<Self> {
        let idx = parse_expr_nocopy(idx)?;
        let obj = idx.obj();
        let mut out = self.clone();
        out.e.group_by_startidx_var(idx.e, min_periods, stable);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "groupby", feature = "agg"))]
    #[pyo3(signature=(idx))]
    pub unsafe fn _group_by_startidx_min(&self, idx: &PyAny) -> PyResult<Self> {
        let idx = parse_expr_nocopy(idx)?;
        let obj = idx.obj();
        let mut out = self.clone();
        out.e.group_by_startidx_min(idx.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "groupby", feature = "agg"))]
    #[pyo3(signature=(idx))]
    pub unsafe fn _group_by_startidx_max(&self, idx: &PyAny) -> PyResult<Self> {
        let idx = parse_expr_nocopy(idx)?;
        let obj = idx.obj();
        let mut out = self.clone();
        out.e.group_by_startidx_max(idx.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "groupby", feature = "agg"))]
    #[pyo3(signature=(idx))]
    pub unsafe fn _group_by_startidx_first(&self, idx: &PyAny) -> PyResult<Self> {
        let idx = parse_expr_nocopy(idx)?;
        let obj = idx.obj();
        let mut out = self.clone();
        out.e.group_by_startidx_first(idx.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "groupby", feature = "agg"))]
    #[pyo3(signature=(idx))]
    pub unsafe fn _group_by_startidx_last(&self, idx: &PyAny) -> PyResult<Self> {
        let idx = parse_expr_nocopy(idx)?;
        let obj = idx.obj();
        let mut out = self.clone();
        out.e.group_by_startidx_last(idx.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "groupby", feature = "agg"))]
    #[pyo3(signature=(idx))]
    pub unsafe fn _group_by_startidx_valid_first(&self, idx: &PyAny) -> PyResult<Self> {
        let idx = parse_expr_nocopy(idx)?;
        let obj = idx.obj();
        let mut out = self.clone();
        out.e.group_by_startidx_valid_first(idx.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "groupby", feature = "agg"))]
    #[pyo3(signature=(idx))]
    pub unsafe fn _group_by_startidx_valid_last(&self, idx: &PyAny) -> PyResult<Self> {
        let idx = parse_expr_nocopy(idx)?;
        let obj = idx.obj();
        let mut out = self.clone();
        out.e.group_by_startidx_valid_last(idx.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "groupby", feature = "agg"))]
    #[pyo3(signature=(idx, other, min_periods=3, stable=false))]
    pub unsafe fn _group_by_startidx_cov(
        &self,
        idx: &PyAny,
        other: &PyAny,
        min_periods: usize,
        stable: bool,
    ) -> PyResult<Self> {
        let idx = parse_expr_nocopy(idx)?;
        let other = parse_expr_nocopy(other)?;
        let obj = idx.obj();
        let obj2 = other.obj();
        let mut out = self.clone();
        out.e
            .group_by_startidx_cov(other.e, idx.e, min_periods, stable);
        out.add_obj(obj).add_obj(obj2);
        Ok(out)
    }

    #[cfg(all(feature = "groupby", feature = "agg"))]
    #[pyo3(signature=(idx, other, method=CorrMethod::Pearson, min_periods=3, stable=false))]
    pub unsafe fn _group_by_startidx_corr(
        &self,
        idx: &PyAny,
        other: &PyAny,
        method: CorrMethod,
        min_periods: usize,
        stable: bool,
    ) -> PyResult<Self> {
        let idx = parse_expr_nocopy(idx)?;
        let other = parse_expr_nocopy(other)?;
        let obj = idx.obj();
        let obj2 = other.obj();
        let mut out = self.clone();
        out.e
            .group_by_startidx_corr(other.e, idx.e, method, min_periods, stable);
        out.add_obj(obj).add_obj(obj2);
        Ok(out)
    }

    #[pyo3(signature=(window, offset))]
    #[cfg(all(feature = "rolling", feature = "agg", feature = "time"))]
    pub unsafe fn _get_time_rolling_offset_idx(
        &self,
        window: &str,
        offset: &str,
    ) -> PyResult<Self> {
        let obj = self.obj();
        let mut out = self.clone();
        out.e.get_time_rolling_offset_idx(window, offset);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(roll_start, min_periods=1, stable=false))]
    pub unsafe fn _rolling_select_mean(
        &self,
        roll_start: &PyAny,
        min_periods: usize,
        stable: bool,
    ) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_mean(roll_start.e, min_periods, stable);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(roll_start, stable=false))]
    pub unsafe fn _rolling_select_sum(&self, roll_start: &PyAny, stable: bool) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_sum(roll_start.e, stable);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(roll_start, min_periods=2, stable=false))]
    pub unsafe fn _rolling_select_std(
        &self,
        roll_start: &PyAny,
        min_periods: usize,
        stable: bool,
    ) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_std(roll_start.e, min_periods, stable);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(roll_start, other, min_periods=2, stable=false))]
    pub unsafe fn _rolling_select_cov(
        &self,
        roll_start: &PyAny,
        other: &PyAny,
        min_periods: usize,
        stable: bool,
    ) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let other = parse_expr_nocopy(other)?;
        let obj = roll_start.obj();
        let obj2 = other.obj();
        let mut out = self.clone();
        out.e
            .rolling_select_cov(other.e, roll_start.e, min_periods, stable);
        out.add_obj(obj).add_obj(obj2);
        Ok(out)
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(roll_start, other, method=CorrMethod::Pearson, min_periods=2, stable=false))]
    pub unsafe fn _rolling_select_corr(
        &self,
        roll_start: &PyAny,
        other: &PyAny,
        method: CorrMethod,
        min_periods: usize,
        stable: bool,
    ) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let other = parse_expr_nocopy(other)?;
        let obj = roll_start.obj();
        let obj2 = other.obj();
        let mut out = self.clone();
        out.e
            .rolling_select_corr(other.e, roll_start.e, method, min_periods, stable);
        out.add_obj(obj).add_obj(obj2);
        Ok(out)
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(roll_start, other, min_periods=2, stable=false))]
    pub unsafe fn _rolling_select_weight_mean(
        &self,
        roll_start: &PyAny,
        other: &PyAny,
        min_periods: usize,
        stable: bool,
    ) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let other = parse_expr_nocopy(other)?;
        let obj = roll_start.obj();
        let obj2 = other.obj();
        let mut out = self.clone();
        out.e
            .rolling_select_weight_mean(other.e, roll_start.e, min_periods, stable);
        out.add_obj(obj).add_obj(obj2);
        Ok(out)
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(roll_start, mask, min_periods=1))]
    pub unsafe fn _rolling_select_cut_mean(
        &self,
        roll_start: &PyAny,
        mask: &PyAny,
        min_periods: usize,
    ) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let mask = parse_expr_nocopy(mask)?;
        let obj = roll_start.obj();
        let obj2 = mask.obj();
        let mut out = self.clone();
        out.e
            .rolling_select_cut_mean(mask.e, roll_start.e, min_periods);
        out.add_obj(obj).add_obj(obj2);
        Ok(out)
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(roll_start, min_periods=2, stable=false))]
    pub unsafe fn _rolling_select_var(
        &self,
        roll_start: &PyAny,
        min_periods: usize,
        stable: bool,
    ) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_var(roll_start.e, min_periods, stable);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(roll_start))]
    pub unsafe fn _rolling_select_first(&self, roll_start: &PyAny) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_first(roll_start.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(roll_start))]
    pub unsafe fn _rolling_select_last(&self, roll_start: &PyAny) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_last(roll_start.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(roll_start))]
    pub unsafe fn _rolling_select_valid_first(&self, roll_start: &PyAny) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_valid_first(roll_start.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(roll_start))]
    pub unsafe fn _rolling_select_valid_last(&self, roll_start: &PyAny) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_valid_last(roll_start.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(roll_start))]
    pub unsafe fn _rolling_select_min(&self, roll_start: &PyAny) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_min(roll_start.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(roll_start))]
    pub unsafe fn _rolling_select_max(&self, roll_start: &PyAny) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_max(roll_start.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg", feature = "map"))]
    #[pyo3(signature=(roll_start))]
    pub unsafe fn _rolling_select_umin(&self, roll_start: &PyAny) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_umin(roll_start.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg", feature = "map"))]
    #[pyo3(signature=(roll_start))]
    pub unsafe fn _rolling_select_umax(&self, roll_start: &PyAny) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_umax(roll_start.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(roll_start, q, method=QuantileMethod::Linear))]
    pub unsafe fn _rolling_select_quantile(
        &self,
        roll_start: &PyAny,
        q: f64,
        method: QuantileMethod,
    ) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_quantile(roll_start.e, q, method);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(idxs, stable=false))]
    pub unsafe fn _rolling_select_by_vecusize_sum(
        &self,
        idxs: &PyAny,
        stable: bool,
    ) -> PyResult<Self> {
        let idxs = parse_expr_nocopy(idxs)?;
        let obj = idxs.obj();
        let mut out = self.clone();
        out.e.rolling_select_by_vecusize_sum(idxs.e, stable);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(idxs, min_periods=1, stable=false))]
    pub unsafe fn _rolling_select_by_vecusize_mean(
        &self,
        idxs: &PyAny,
        min_periods: usize,
        stable: bool,
    ) -> PyResult<Self> {
        let idxs = parse_expr_nocopy(idxs)?;
        let obj = idxs.obj();
        let mut out = self.clone();
        out.e
            .rolling_select_by_vecusize_mean(idxs.e, min_periods, stable);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(idxs, min_periods=1, stable=false))]
    pub unsafe fn _rolling_select_by_vecusize_var(
        &self,
        idxs: &PyAny,
        min_periods: usize,
        stable: bool,
    ) -> PyResult<Self> {
        let idxs = parse_expr_nocopy(idxs)?;
        let obj = idxs.obj();
        let mut out = self.clone();
        out.e
            .rolling_select_by_vecusize_var(idxs.e, min_periods, stable);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(idxs, min_periods=1, stable=false))]
    pub unsafe fn _rolling_select_by_vecusize_std(
        &self,
        idxs: &PyAny,
        min_periods: usize,
        stable: bool,
    ) -> PyResult<Self> {
        let idxs = parse_expr_nocopy(idxs)?;
        let obj = idxs.obj();
        let mut out = self.clone();
        out.e
            .rolling_select_by_vecusize_std(idxs.e, min_periods, stable);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(idxs))]
    pub unsafe fn _rolling_select_by_vecusize_first(&self, idxs: &PyAny) -> PyResult<Self> {
        let idxs = parse_expr_nocopy(idxs)?;
        let obj = idxs.obj();
        let mut out = self.clone();
        out.e.rolling_select_by_vecusize_first(idxs.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(idxs))]
    pub unsafe fn _rolling_select_by_vecusize_last(&self, idxs: &PyAny) -> PyResult<Self> {
        let idxs = parse_expr_nocopy(idxs)?;
        let obj = idxs.obj();
        let mut out = self.clone();
        out.e.rolling_select_by_vecusize_last(idxs.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(idxs))]
    pub unsafe fn _rolling_select_by_vecusize_valid_first(&self, idxs: &PyAny) -> PyResult<Self> {
        let idxs = parse_expr_nocopy(idxs)?;
        let obj = idxs.obj();
        let mut out = self.clone();
        out.e.rolling_select_by_vecusize_valid_first(idxs.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(idxs))]
    pub unsafe fn _rolling_select_by_vecusize_valid_last(&self, idxs: &PyAny) -> PyResult<Self> {
        let idxs = parse_expr_nocopy(idxs)?;
        let obj = idxs.obj();
        let mut out = self.clone();
        out.e.rolling_select_by_vecusize_valid_last(idxs.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(idxs))]
    pub unsafe fn _rolling_select_by_vecusize_max(&self, idxs: &PyAny) -> PyResult<Self> {
        let idxs = parse_expr_nocopy(idxs)?;
        let obj = idxs.obj();
        let mut out = self.clone();
        out.e.rolling_select_by_vecusize_max(idxs.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg", feature = "map"))]
    #[pyo3(signature=(idxs))]
    pub unsafe fn _rolling_select_by_vecusize_umax(&self, idxs: &PyAny) -> PyResult<Self> {
        let idxs = parse_expr_nocopy(idxs)?;
        let obj = idxs.obj();
        let mut out = self.clone();
        out.e.rolling_select_by_vecusize_umax(idxs.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(idxs))]
    pub unsafe fn _rolling_select_by_vecusize_min(&self, idxs: &PyAny) -> PyResult<Self> {
        let idxs = parse_expr_nocopy(idxs)?;
        let obj = idxs.obj();
        let mut out = self.clone();
        out.e.rolling_select_by_vecusize_min(idxs.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg", feature = "map"))]
    #[pyo3(signature=(idxs))]
    pub unsafe fn _rolling_select_by_vecusize_umin(&self, idxs: &PyAny) -> PyResult<Self> {
        let idxs = parse_expr_nocopy(idxs)?;
        let obj = idxs.obj();
        let mut out = self.clone();
        out.e.rolling_select_by_vecusize_umin(idxs.e);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(all(feature = "rolling", feature = "agg"))]
    #[pyo3(signature=(idxs, q, method=QuantileMethod::Linear))]
    pub unsafe fn _rolling_select_by_vecusize_quantile(
        &self,
        idxs: &PyAny,
        q: f64,
        method: QuantileMethod,
    ) -> PyResult<Self> {
        let idxs = parse_expr_nocopy(idxs)?;
        let obj = idxs.obj();
        let mut out = self.clone();
        out.e.rolling_select_by_vecusize_quantile(idxs.e, q, method);
        Ok(out.add_obj_into(obj))
    }
}
