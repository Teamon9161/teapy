use super::export::*;
use super::pyfunc::{parse_expr, parse_expr_list, parse_expr_nocopy, where_};
use crate::arr::{
    DateTime, DropNaMethod, ExprOut, ExprOutView, Number, RefType, TimeDelta, WinsorizeMethod,
};
use crate::from_py::{NoDim0, PyValue};
use ndarray::SliceInfoElem;
use num::PrimInt;
use pyo3::{pyclass::CompareOp, PyTraverseError, PyVisit};
use std::iter::repeat;

macro_rules! impl_py_matmul {
    (cast_to_same $self: expr, $other: expr, $func: ident $(,$args: expr)*) => {
        {
            let (obj, obj2) = ($self.obj(), $other.obj());
            match (&$self.inner, &$other.inner) {
                (Exprs::F64(_), _) | (_, Exprs::F64(_)) => {
                    match_exprs!(($self.inner, e1, I32, I64, F32, F64, Usize), ($other.inner, e2, I32, I64, F32, F64, Usize), {
                        (e1.cast::<f64>().$func(e2.cast::<f64>() $(,$args)*)).to_py(obj).add_obj(obj2)
                    })
                },
                (Exprs::F32(_), _) | (_, Exprs::F32(_)) => {
                    match_exprs!(($self.inner, e1, I32, I64, F32, Usize), ($other.inner, e2, I32, I64, F32, Usize), {
                        (e1.cast::<f32>().$func(e2.cast::<f32>() $(,$args)*)).to_py(obj).add_obj(obj2)
                    })
                },
                (Exprs::I64(_), _) | (_, Exprs::I64(_)) => {
                    match_exprs!(($self.inner, e1, I64, I32, Bool, Usize), ($other.inner, e2, I64, I32, Bool, Usize), {
                        (e1.cast::<i64>().$func(e2.cast::<i64>() $(,$args)*)).to_py(obj).add_obj(obj2)
                    })
                },
                (Exprs::I32(_), _) | (_, Exprs::I32(_)) => {
                    match_exprs!(($self.inner, e1, I32, Bool, Usize), ($other.inner, e2, I32, Bool, Usize), {
                        (e1.cast::<i32>().$func(e2.cast::<i32>() $(,$args)*)).to_py(obj).add_obj(obj2)
                    })
                },
                _ => todo!()
            }
        }
    }
}

macro_rules! impl_py_cmp {
    (cast_to_same $self: expr, $other: expr, $func: ident, $py: expr $(,$dtype: ident)*) => {
        {
            let (obj, obj2) = ($self.obj(), $other.obj());
            use Exprs::*;
            match (&$self.inner, &$other.inner) {
                (F64(_), _) | (_, F64(_)) => {
                    match_exprs!(($self.inner, e1, I32, F64, Usize), ($other.inner, e2, I32, F64, Usize), {
                        (e1.cast::<f64>().$func(e2.cast::<f64>(), false)).to_py(obj).add_obj(obj2)
                    })
                },
                (F32(_), _) | (_, F32(_)) => {
                    match_exprs!(($self.inner, e1, I32, I64, F32, Usize), ($other.inner, e2, I32, I64, F32, Usize), {
                        (e1.cast::<f32>().$func(e2.cast::<f32>(), false)).to_py(obj).add_obj(obj2)
                    })
                },
                (I64(_), _) | (_, I64(_)) => {
                    match_exprs!(($self.inner, e1, I64, I32, Bool, Usize), ($other.inner, e2, I64, I32, Bool, Usize), {
                        (e1.cast::<i64>().$func(e2.cast::<i64>(), false)).to_py(obj).add_obj(obj2)
                    })
                },
                (I32(_), _) | (_, I32(_)) => {
                    match_exprs!(($self.inner, e1, I32, Bool, Usize), ($other.inner, e2, I32, Bool, Usize), {
                        (e1.cast::<i32>().$func(e2.cast::<i32>(), false)).to_py(obj).add_obj(obj2)
                    })
                },
                (Object(e1), String(e2)) => e1.clone().object_to_string($py).$func(e2.clone(), false).to_py(obj).add_obj(obj2),
                (String(e1), Object(e2)) => e1.clone().$func(e2.clone().object_to_string($py), false).to_py(obj).add_obj(obj2),
                $(
                    ($dtype(e1), $dtype(e2)) => e1.clone().$func(e2.clone(), false).to_py(obj).add_obj(obj2),
                )*
                _ => todo!()
            }
        }
    };

}

#[pymethods]
#[allow(clippy::missing_safety_doc)]
impl PyExpr {
    #[new]
    #[pyo3(signature=(expr, name=None, copy=false))]
    pub unsafe fn new(expr: &PyAny, name: Option<String>, copy: bool) -> PyResult<Self> {
        let mut out = parse_expr(expr, copy)?;
        if let Some(name) = name {
            out.set_name(name);
        }
        Ok(out)
    }

    #[getter]
    pub fn dtype(&self) -> &str {
        use Exprs::*;
        match self.inner {
            F32(_) => "Float32",
            F64(_) => "Float64",
            I32(_) => "Int32",
            I64(_) => "Int64",
            Usize(_) => "Usize",
            Bool(_) => "Bool",
            String(_) => "String",
            Str(_) => "Str",
            Object(_) => "Object",
            DateTime(_) => "DateTime",
            TimeDelta(_) => "TimeDelta",
            OpUsize(_) => "OptionUsize",
        }
    }

    #[getter]
    #[allow(unreachable_patterns)]
    pub fn get_base_type(&self) -> &'static str {
        match_exprs!(&self.inner, e, { e.get_base_type() })
    }

    #[getter]
    #[allow(unreachable_patterns)]
    pub fn get_base_strong_count(&self) -> PyResult<usize> {
        match_exprs!(&self.inner, e, {
            e.get_base_strong_count().map_err(PyValueError::new_err)
        })
    }

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

    #[allow(unreachable_patterns)]
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
            let ori_dim = self.ndim().cast_i32()?;
            let mut out = self.view_by_slice(slc_vec);
            for (idx, slc_obj) in zip(no_slice_idx_vec, no_slice_slc_obj_vec) {
                let slc = parse_expr_nocopy(slc_obj)?;
                let obj = slc.obj();
                let idx: Expr<'static, i32> = (idx as i32).into();
                let idx = idx - (ori_dim.clone() - out.ndim().cast_i32()?);
                out = if slc.is_bool() {
                    match_exprs!(&out.inner, expr, {
                        expr.clone()
                            .filter(slc.cast_bool()?, idx, false)
                            .to_py(out.obj())
                            .add_obj(obj)
                    })
                } else {
                    match_exprs!(&out.inner, expr, {
                        expr.clone()
                            .select_on_axis_by_i32_expr(slc.cast_i32()?, idx)
                            .to_py(out.obj())
                            .add_obj(obj)
                    })
                }
            }
            Ok(out)
        } else {
            let obj = PyTuple::new(py, vec![obj]);
            self.__getitem__(obj.into(), py)
        }
    }

    #[pyo3(name="eval", signature=(inplace=false))]
    #[allow(unreachable_patterns)]
    pub fn eval_py(&mut self, inplace: bool) -> Option<Self> {
        self.eval_inplace();
        if !inplace {
            Some(self.clone())
        } else {
            // self.eval_inplace();
            None
        }
    }

    #[getter]
    #[allow(unreachable_patterns)]
    pub fn get_view(self: PyRefMut<Self>, py: Python) -> PyResult<PyObject> {
        if self.is_string() || self.is_str() {
            let arr = self.clone().cast_object_eager(py)?.into_arr().to_owned().0;
            return PyArray::from_owned_array(py, arr).no_dim0(py);
        } else if let Exprs::DateTime(e) = &self.inner {
            let arr = e
                .clone()
                .eval()
                .view_arr()
                .map(|v| v.into_np_datetime::<numpy::datetime::units::Milliseconds>());
            return PyArray::from_owned_array(py, arr.0).no_dim0(py);
        }
        match_exprs!(
            &self.inner,
            expr,
            {
                unsafe {
                    let container = PyAny::from_borrowed_ptr(py, self.as_ptr());
                    let out_view = expr.try_view().map_err(PyValueError::new_err)?;
                    if matches!(out_view, ExprOutView::ArrVec(_)) {
                        let out = out_view
                            .as_arr_vec()
                            .iter()
                            .map(|arr| {
                                PyArray::borrow_from_array(arr, container)
                                    .no_dim0(py)
                                    .unwrap()
                            })
                            .collect_trusted();
                        Ok(out.into_py(py))
                    } else {
                        Ok(PyArray::borrow_from_array(out_view.as_arr(), container).no_dim0(py)?)
                    }
                }
            },
            I32,
            I64,
            F32,
            F64,
            Usize,
            Bool,
            Object
        )
    }

    #[allow(unreachable_patterns)]
    /// eval and view, used for fast test
    pub fn eview(mut self: PyRefMut<Self>, py: Python) -> PyResult<PyObject> {
        self.eval_inplace();
        self.get_view(py)
    }

    #[allow(unreachable_patterns)]
    pub fn value<'py>(&'py self, py: Python<'py>) -> PyResult<PyObject> {
        if self.is_string() || self.is_str() || self.is_timedelta() {
            let arr = self.clone().cast_object_eager(py)?.into_arr().to_owned().0;
            return PyArray::from_owned_array(py, arr).no_dim0(py);
        } else if let Exprs::DateTime(e) = &self.inner {
            let arr = e
                .clone()
                .eval()
                .view_arr()
                .map(|v| v.into_np_datetime::<numpy::datetime::units::Milliseconds>());
            return PyArray::from_owned_array(py, arr.0).no_dim0(py);
        }
        match_exprs!(
            &self.inner,
            expr,
            {
                let out = expr.clone().into_out();
                if matches!(out, ExprOut::ArrVec(_)) {
                    let out = out
                        .into_arr_vec()
                        .into_iter()
                        .map(|arr| {
                            PyArray::from_owned_array(py, arr.to_owned().0)
                                .no_dim0(py)
                                .unwrap()
                        })
                        .collect_trusted();
                    Ok(out.into_py(py))
                } else {
                    let out = out.into_arr().to_owned().0;
                    Ok(PyArray::from_owned_array(py, out).no_dim0(py)?)
                }
            },
            I32,
            I64,
            F32,
            F64,
            Usize,
            Bool,
            Object
        )
    }

    #[getter]
    #[allow(unreachable_patterns)]
    pub fn step(&self) -> usize {
        match_exprs!(&self.inner, expr, { expr.step_acc() })
    }

    // #[getter]
    // #[allow(unreachable_patterns)]
    // pub fn step_acc(&self) -> usize {
    //     match_exprs!(&self.inner, expr, { expr.step_acc() })
    // }

    #[getter]
    #[allow(unreachable_patterns)]
    pub fn get_name(&self) -> Option<String> {
        match_exprs!(&self.inner, expr, { expr.name() })
    }

    #[setter]
    #[allow(unreachable_patterns)]
    pub fn set_name(&mut self, name: String) {
        match_exprs!(&mut self.inner, expr, { expr.rename(name) })
    }

    #[pyo3(name = "copy")]
    #[allow(unreachable_patterns)]
    pub fn deep_copy(&self) -> Self {
        match_exprs!(&self.inner, expr, {
            expr.clone().deep_copy().to_py(self.obj())
        })
    }

    #[allow(unreachable_patterns)]
    pub(crate) fn is_owned(&self) -> Option<bool> {
        match_exprs!(&self.inner, expr, { expr.owned() })
    }

    #[allow(unreachable_patterns)]
    pub unsafe fn reshape(&self, shape: &PyAny) -> PyResult<Self> {
        let shape = parse_expr_nocopy(shape)?;
        let obj = shape.obj();
        let out = match_exprs!(&self.inner, expr, {
            expr.clone()
                .reshape(shape.cast_usize()?)
                .to_py(self.obj())
                .add_obj(obj)
        });
        Ok(out)
    }

    #[allow(unreachable_patterns)]
    pub fn strong_count(&mut self) -> usize {
        match_exprs!(&self.inner, expr, { expr.strong_count() })
    }

    #[allow(unreachable_patterns)]
    pub fn weak_count(&mut self) -> usize {
        match_exprs!(&self.inner, expr, { expr.weak_count() })
    }

    #[allow(unreachable_patterns)]
    pub fn ref_count(&mut self) -> usize {
        match_exprs!(&self.inner, expr, { expr.ref_count() })
    }

    #[allow(unreachable_patterns)]
    pub fn hint_arr_type(&mut self) -> Self {
        match_exprs!(&self.inner, expr, { expr.clone().hint_arr_type().into() })
    }

    unsafe fn __add__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        use Exprs::*;
        let out = match (&self.inner, &other.inner) {
            (F64(_), _) | (_, F64(_)) => (self.clone().cast_f64()? + other.cast_f64()?)
                .to_py(obj)
                .add_obj(obj2),
            (F32(_), _) | (_, F32(_)) => (self.clone().cast_f32()? + other.cast_f32()?)
                .to_py(obj)
                .add_obj(obj2),
            (I64(_), _) | (_, I64(_)) => (self.clone().cast_i64()? + other.cast_i64()?)
                .to_py(obj)
                .add_obj(obj2),
            (I32(_), _) | (_, I32(_)) => (self.clone().cast_i32()? + other.cast_i32()?)
                .to_py(obj)
                .add_obj(obj2),
            (Usize(_), _) | (_, Usize(_)) => (self.clone().cast_usize()? + other.cast_usize()?)
                .to_py(obj)
                .add_obj(obj2),
            (String(_), String(_)) => self
                .clone()
                .cast_string()?
                .add_string(other.cast_string()?)
                .to_py(obj)
                .add_obj(obj2),
            (Str(_), String(_)) => self
                .clone()
                .cast_string()?
                .add_string(other.cast_string()?)
                .to_py(obj)
                .add_obj(obj2),
            (String(_), Str(_)) => self
                .clone()
                .cast_string()?
                .add_str(other.cast_str()?)
                .to_py(obj)
                .add_obj(obj2),
            (Str(_), Str(_)) => self
                .clone()
                .cast_string()?
                .add_str(other.cast_str()?)
                .to_py(obj)
                .add_obj(obj2),
            (DateTime(_), TimeDelta(_)) => (self.clone().cast_datetime_default()?
                + other.cast_timedelta()?)
            .to_py(obj)
            .add_obj(obj2),
            (TimeDelta(_), DateTime(_)) => (other.cast_datetime_default()?
                + self.clone().cast_timedelta()?)
            .to_py(obj)
            .add_obj(obj2),
            (TimeDelta(_), TimeDelta(_)) => (self.clone().cast_timedelta()?
                + other.cast_timedelta()?)
            .to_py(obj)
            .add_obj(obj2),
            _ => todo!(),
        };
        Ok(out)
    }

    unsafe fn __radd__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        use Exprs::*;
        let out = match (&self.inner, &other.inner) {
            (F64(_), _) | (_, F64(_)) => (self.clone().cast_f64()? + other.cast_f64()?)
                .to_py(obj)
                .add_obj(obj2),
            (F32(_), _) | (_, F32(_)) => (self.clone().cast_f32()? + other.cast_f32()?)
                .to_py(obj)
                .add_obj(obj2),
            (I64(_), _) | (_, I64(_)) => (self.clone().cast_i64()? + other.cast_i64()?)
                .to_py(obj)
                .add_obj(obj2),
            (I32(_), _) | (_, I32(_)) => (self.clone().cast_i32()? + other.cast_i32()?)
                .to_py(obj)
                .add_obj(obj2),
            (Usize(_), _) | (_, Usize(_)) => (self.clone().cast_usize()? + other.cast_usize()?)
                .to_py(obj)
                .add_obj(obj2),
            // note that we should not swap the order of self and other when add string
            (String(_), String(_)) => other
                .cast_string()?
                .add_string(self.clone().cast_string()?)
                .to_py(obj)
                .add_obj(obj2),
            (String(_), Str(_)) => other
                .cast_string()?
                .add_string(self.clone().cast_string()?)
                .to_py(obj)
                .add_obj(obj2),
            (Str(_), String(_)) => other
                .cast_string()?
                .add_str(self.clone().cast_str()?)
                .to_py(obj)
                .add_obj(obj2),
            (Str(_), Str(_)) => other
                .cast_string()?
                .add_str(self.clone().cast_str()?)
                .to_py(obj)
                .add_obj(obj2),
            (DateTime(_), TimeDelta(_)) => (self.clone().cast_datetime_default()?
                + other.cast_timedelta()?)
            .to_py(obj)
            .add_obj(obj2),
            (TimeDelta(_), DateTime(_)) => (other.cast_datetime_default()?
                + self.clone().cast_timedelta()?)
            .to_py(obj)
            .add_obj(obj2),
            (TimeDelta(_), TimeDelta(_)) => (self.clone().cast_timedelta()?
                + other.cast_timedelta()?)
            .to_py(obj)
            .add_obj(obj2),
            _ => todo!(),
        };
        Ok(out)
    }

    unsafe fn __iadd__(&mut self, other: &PyAny) {
        *self = self.__add__(other).unwrap()
    }

    unsafe fn __sub__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        use Exprs::*;
        let out = match (&self.inner, &other.inner) {
            (F64(_), _) | (_, F64(_)) => (self.clone().cast_f64()? - other.cast_f64()?)
                .to_py(obj)
                .add_obj(obj2),
            (F32(_), _) | (_, F32(_)) => (self.clone().cast_f32()? - other.cast_f32()?)
                .to_py(obj)
                .add_obj(obj2),
            (I64(_), _) | (_, I64(_)) => (self.clone().cast_i64()? - other.cast_i64()?)
                .to_py(obj)
                .add_obj(obj2),
            (I32(_), _) | (_, I32(_)) => (self.clone().cast_i32()? - other.cast_i32()?)
                .to_py(obj)
                .add_obj(obj2),
            (Usize(_), _) | (_, Usize(_)) => (self.clone().cast_i64()? - other.cast_i64()?)
                .to_py(obj)
                .add_obj(obj2),
            (DateTime(_), TimeDelta(_)) => (self.clone().cast_datetime_default()?
                - other.cast_timedelta()?)
            .to_py(obj)
            .add_obj(obj2),
            (TimeDelta(_), TimeDelta(_)) => (self.clone().cast_timedelta()?
                - other.cast_timedelta()?)
            .to_py(obj)
            .add_obj(obj2),
            (DateTime(_), DateTime(_)) => (self
                .clone()
                .cast_datetime_default()?
                .sub_datetime(other.cast_datetime_default()?, false))
            .to_py(obj)
            .add_obj(obj2),
            _ => todo!(),
        };
        Ok(out)
    }

    unsafe fn __rsub__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        use Exprs::*;
        let out = match (&other.inner, &self.inner) {
            (F64(_), _) | (_, F64(_)) => (other.cast_f64()? - self.clone().cast_f64()?)
                .to_py(obj)
                .add_obj(obj2),
            (F32(_), _) | (_, F32(_)) => (other.cast_f32()? - self.clone().cast_f32()?)
                .to_py(obj)
                .add_obj(obj2),
            (I64(_), _) | (_, I64(_)) => (other.cast_i64()? - self.clone().cast_i64()?)
                .to_py(obj)
                .add_obj(obj2),
            (I32(_), _) | (_, I32(_)) => (other.cast_i32()? - self.clone().cast_i32()?)
                .to_py(obj)
                .add_obj(obj2),
            (Usize(_), _) | (_, Usize(_)) => (other.cast_i64()? - self.clone().cast_i64()?)
                .to_py(obj)
                .add_obj(obj2),
            (DateTime(_), TimeDelta(_)) => (other.cast_datetime_default()?
                - self.clone().cast_timedelta()?)
            .to_py(obj)
            .add_obj(obj2),
            (TimeDelta(_), TimeDelta(_)) => (other.cast_timedelta()?
                - self.clone().cast_timedelta()?)
            .to_py(obj)
            .add_obj(obj2),
            (DateTime(_), DateTime(_)) => (other
                .cast_datetime_default()?
                .sub_datetime(self.clone().cast_datetime_default()?, false))
            .to_py(obj)
            .add_obj(obj2),
            _ => todo!(),
        };
        Ok(out)
    }

    unsafe fn __isub__(&mut self, other: &PyAny) {
        *self = self.__sub__(other).unwrap()
    }

    unsafe fn __mul__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        use Exprs::*;
        let out = match (&self.inner, &other.inner) {
            (F64(_), _) | (_, F64(_)) => (self.clone().cast_f64()? * other.cast_f64()?)
                .to_py(obj)
                .add_obj(obj2),
            (F32(_), _) | (_, F32(_)) => (self.clone().cast_f32()? * other.cast_f32()?)
                .to_py(obj)
                .add_obj(obj2),
            (I64(_), _) | (_, I64(_)) => (self.clone().cast_i64()? * other.cast_i64()?)
                .to_py(obj)
                .add_obj(obj2),
            (I32(_), _) | (_, I32(_)) => (self.clone().cast_i32()? * other.cast_i32()?)
                .to_py(obj)
                .add_obj(obj2),
            (Usize(_), _) | (_, Usize(_)) => (self.clone().cast_usize()? * other.cast_usize()?)
                .to_py(obj)
                .add_obj(obj2),
            _ => todo!(),
        };
        Ok(out)
    }

    unsafe fn __rmul__(&self, other: &PyAny) -> PyResult<Self> {
        self.__mul__(other)
    }

    unsafe fn __imul__(&mut self, other: &PyAny) {
        *self = self.__mul__(other).unwrap()
    }

    unsafe fn __truediv__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        use Exprs::*;
        let out = match (&self.inner, &other.inner) {
            (F64(_), _) | (_, F64(_)) => (self.clone().cast_f64()? / other.cast_f64()?)
                .to_py(obj)
                .add_obj(obj2),
            (F32(_), _) | (_, F32(_)) => (self.clone().cast_f32()? / other.cast_f32()?)
                .to_py(obj)
                .add_obj(obj2),
            (I64(_), _)
            | (_, I64(_))
            | (I32(_), _)
            | (_, I32(_))
            | (Usize(_), _)
            | (_, Usize(_)) => (self.clone().cast_f64()? / other.cast_f64()?)
                .to_py(obj)
                .add_obj(obj2),
            _ => todo!(),
        };
        Ok(out)
    }

    unsafe fn __rtruediv__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        use Exprs::*;
        let out = match (&other.inner, &self.inner) {
            (F64(_), _) | (_, F64(_)) => (other.cast_f64()? / self.clone().cast_f64()?)
                .to_py(obj)
                .add_obj(obj2),
            (F32(_), _) | (_, F32(_)) => (other.cast_f32()? / self.clone().cast_f32()?)
                .to_py(obj)
                .add_obj(obj2),
            (I64(_), _)
            | (_, I64(_))
            | (I32(_), _)
            | (_, I32(_))
            | (Usize(_), _)
            | (_, Usize(_)) => (other.cast_f64()? / self.clone().cast_f64()?)
                .to_py(obj)
                .add_obj(obj2),
            _ => todo!(),
        };
        Ok(out)
    }

    unsafe fn __itruediv__(&mut self, other: &PyAny) {
        *self = self.__truediv__(other).unwrap()
    }

    #[allow(unreachable_patterns)]
    unsafe fn __and__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        match_exprs!(
            (&self.inner, e1, F32, F64, I32, I64, Bool),
            (other.inner, e2, F64, F32, I32, I64, Bool),
            {
                Ok((e1.clone().cast_bool() & e2.cast_bool())
                    .to_py(self.obj())
                    .add_obj(obj))
            }
        )
    }

    unsafe fn __rand__(&self, other: &PyAny) -> PyResult<Self> {
        self.__and__(other)
    }

    unsafe fn __iand__(&mut self, other: &PyAny) {
        *self = self.__and__(other).unwrap()
    }

    #[allow(unreachable_patterns)]
    unsafe fn __or__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        match_exprs!(
            (&self.inner, e1, F64, F32, I64, I32, Bool),
            (other.inner, e2, F64, F32, I64, I32, Bool),
            {
                Ok((e1.clone().cast_bool() | e2.cast_bool())
                    .to_py(self.obj())
                    .add_obj(obj))
            }
        )
    }

    unsafe fn __ror__(&self, other: &PyAny) -> PyResult<Self> {
        self.__or__(other)
    }

    unsafe fn __ior__(&mut self, other: &PyAny) {
        *self = self.__or__(other).unwrap()
    }

    unsafe fn __richcmp__(&self, other: &PyAny, op: CompareOp, py: Python) -> PyResult<PyExpr> {
        let other = parse_expr_nocopy(other)?;
        match op {
            CompareOp::Eq => Ok(
                impl_py_cmp!(cast_to_same self.clone(), other, eq, py, DateTime, String, TimeDelta),
            ),
            CompareOp::Lt => Ok(
                impl_py_cmp!(cast_to_same self.clone(), other, lt, py, DateTime, String, TimeDelta),
            ),
            CompareOp::Le => Ok(
                impl_py_cmp!(cast_to_same self.clone(), other, le, py, DateTime, String, TimeDelta),
            ),
            CompareOp::Ne => Ok(
                impl_py_cmp!(cast_to_same self.clone(), other, ne, py, DateTime, String, TimeDelta),
            ),
            CompareOp::Gt => Ok(
                impl_py_cmp!(cast_to_same self.clone(), other, gt, py, DateTime, String, TimeDelta),
            ),
            CompareOp::Ge => Ok(
                impl_py_cmp!(cast_to_same self.clone(), other, ge, py, DateTime, String, TimeDelta),
            ),
        }
    }

    unsafe fn __pow__(&self, other: &PyAny, _mod: &PyAny) -> PyResult<Self> {
        self.pow_py(other, false)
    }

    pub fn __round__(&self, precision: u32) -> PyResult<Self> {
        self.round(precision)
    }

    #[pyo3(name="pow", signature=(other, par=false))]
    unsafe fn pow_py(&self, other: &PyAny, par: bool) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        self.pow(other, par)
    }

    #[pyo3(signature=(par=false))]
    fn abs(&self, par: bool) -> PyResult<Self> {
        Ok(match_exprs!(
            &self.inner,
            e,
            { e.clone().abs(par).to_py(self.obj()) },
            F64,
            F32,
            I32,
            I64
        ))
    }

    fn __abs__(&self) -> PyResult<Self> {
        self.abs(false)
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

    fn __neg__(&self) -> PyResult<Self> {
        match_exprs!(
            &self.inner,
            e,
            { Ok((-e.clone()).to_py(self.obj())) },
            F64,
            F32,
            I64,
            I32
        )
    }

    fn __invert__(&self) -> PyResult<Self> {
        match_exprs!(&self.inner, e, { Ok((!e.clone()).to_py(self.obj())) }, Bool)
    }

    #[cfg(feature = "blas")]
    unsafe fn __matmul__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        Ok(impl_py_matmul!(cast_to_same self.clone(), other, dot))
    }

    #[cfg(feature = "blas")]
    unsafe fn __rmatmul__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        Ok(impl_py_matmul!(cast_to_same other, self.clone(), dot))
    }

    #[pyo3(signature=(name))]
    #[allow(unreachable_patterns)]
    pub fn alias(&mut self, name: String) -> Self {
        match_exprs!(&mut self.inner, expr, {
            let mut e = expr.clone();
            e.rename(name);
            e.to_py(self.obj())
        })
    }

    // pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
    //     match state.extract::<&pyo3::types::PyBytes>(py) {
    //         Ok(s) => {
    //             self.foo = bincode::deserialize(s.as_bytes()).unwrap();
    //             Ok(())
    //         }
    //         Err(e) => Err(e),
    //     }
    // }

    // pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
    //     Ok(PyBytes::new(py, &bincode::serialize(&self.foo).unwrap()).to_object(py))
    // }

    /// Returns the square root of a number.
    ///
    /// Returns NaN if self is a negative number other than -0.0.
    pub fn sqrt(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().sqrt().to_py(self.obj())
        })
    }

    /// Returns the cube root of each element.
    pub fn cbrt(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().exp2().to_py(self.obj())
        })
    }

    /// Returns the sign of each element.
    fn sign(&self) -> PyResult<Self> {
        Ok(match_exprs!(
            &self.inner,
            e,
            { e.clone().sign(false).to_py(self.obj()) },
            F64,
            F32,
            I32,
            I64
        ))
    }

    /// Returns the natural logarithm of each element.
    pub fn ln(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().ln().to_py(self.obj())
        })
    }

    /// Returns ln(1+n) (natural logarithm) more accurately than if the operations were performed separately.
    pub fn ln_1p(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().ln_1p().to_py(self.obj())
        })
    }

    /// Returns the base 2 logarithm of each element.
    pub fn log2(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().log2().to_py(self.obj())
        })
    }

    /// Returns the base 10 logarithm of each element.
    pub fn log10(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().log10().to_py(self.obj())
        })
    }

    /// Returns e^(self) of each element, (the exponential function).
    pub fn exp(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().exp().to_py(self.obj())
        })
    }

    /// Returns 2^(self) of each element.
    pub fn exp2(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().exp2().to_py(self.obj())
        })
    }

    /// Returns e^(self) - 1 in a way that is accurate even if the number is close to zero.
    pub fn exp_m1(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().exp_m1().to_py(self.obj())
        })
    }

    /// Computes the arccosine of each element. Return value is in radians in the range 0,
    /// pi or NaN if the number is outside the range -1, 1.
    pub fn acos(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().acos().to_py(self.obj())
        })
    }

    /// Computes the arcsine of each element. Return value is in radians in the range -pi/2,
    /// pi/2 or NaN if the number is outside the range -1, 1.
    pub fn asin(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().asin().to_py(self.obj())
        })
    }

    /// Computes the arctangent of each element. Return value is in radians in the range -pi/2, pi/2;
    pub fn atan(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().atan().to_py(self.obj())
        })
    }

    /// Computes the sine of each element (in radians).
    pub fn sin(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().sin().to_py(self.obj())
        })
    }

    /// Computes the cosine of each element (in radians).
    pub fn cos(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().cos().to_py(self.obj())
        })
    }

    /// Computes the tangent of each element (in radians).
    pub fn tan(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().tan().to_py(self.obj())
        })
    }

    /// Returns the smallest integer greater than or equal to `self`.
    pub fn ceil(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().ceil().to_py(self.obj())
        })
    }

    /// Returns the largest integer less than or equal to `self`.
    pub fn floor(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().floor().to_py(self.obj())
        })
    }

    /// Returns the fractional part of each element.
    pub fn fract(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().fract().to_py(self.obj())
        })
    }

    /// Returns the integer part of each element. This means that non-integer numbers are always truncated towards zero.
    pub fn trunc(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().trunc().to_py(self.obj())
        })
    }

    /// Returns true if this number is neither infinite nor NaN
    pub fn is_finite(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().is_finite().to_py(self.obj())
        })
    }

    /// Returns true if this value is positive infinity or negative infinity, and false otherwise.
    pub fn is_inf(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().is_inf().to_py(self.obj())
        })
    }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    ///
    /// The result might not be correctly rounded owing to implementation details;
    /// `self.log2()` can produce more accurate results for base 2,
    /// and `self.log10()` can produce more accurate results for base 10.
    pub fn log(&self, base: f64) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().log(base).to_py(self.obj())
        })
    }

    #[pyo3(signature=(axis=0, par=false))]
    #[allow(unreachable_patterns)]
    pub fn first(&self, axis: i32, par: bool) -> Self {
        match_exprs!(&self.inner, expr, {
            expr.clone().first(axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(axis=0, par=false))]
    #[allow(unreachable_patterns)]
    pub fn last(&self, axis: i32, par: bool) -> Self {
        match_exprs!(&self.inner, expr, {
            expr.clone().last(axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn valid_first(&self, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().valid_first(axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn valid_last(&self, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().valid_last(axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(n=1, fill=None, axis=0, par=false))]
    pub fn shift(
        &self,
        n: i32,
        fill: Option<&PyAny>,
        axis: i32,
        par: bool,
        py: Python,
    ) -> PyResult<Self> {
        let fill = if let Some(fill) = fill {
            if fill.is_none() {
                None
            } else {
                unsafe { Some(parse_expr_nocopy(fill)?) }
            }
        } else {
            None
        };
        let obj = fill.clone().map(|e| e.obj());
        let out = match &self.inner {
            Exprs::Bool(e) => e
                .clone()
                .shift(
                    n,
                    fill.expect("A fill value should be passed when shift a bool expression")
                        .cast_bool()?,
                    axis,
                    par,
                )
                .to_py(self.obj()),
            Exprs::F64(e) => e
                .clone()
                .shift(
                    n,
                    fill.unwrap_or_else(|| f64::NAN.into()).cast_f64()?,
                    axis,
                    par,
                )
                .to_py(self.obj()),
            Exprs::F32(e) => e
                .clone()
                .shift(
                    n,
                    fill.unwrap_or_else(|| f64::NAN.into()).cast_f32()?,
                    axis,
                    par,
                )
                .to_py(self.obj()),
            Exprs::I64(e) => {
                if let Some(fill) = fill {
                    e.clone()
                        .shift(n, fill.cast_i64()?, axis, par)
                        .to_py(self.obj())
                } else {
                    // default fill value is NaN
                    e.clone()
                        .cast::<f64>()
                        .shift(n, f64::NAN.into(), axis, par)
                        .to_py(self.obj())
                }
            }
            Exprs::I32(e) => {
                if let Some(fill) = fill {
                    e.clone()
                        .shift(n, fill.cast_i32()?, axis, par)
                        .to_py(self.obj())
                } else {
                    // default fill value is NaN
                    e.clone()
                        .cast::<f64>()
                        .shift(n, f64::NAN.into(), axis, par)
                        .to_py(self.obj())
                }
            }
            Exprs::Usize(e) => {
                if let Some(fill) = fill {
                    e.clone()
                        .shift(n, fill.cast_usize()?, axis, par)
                        .to_py(self.obj())
                } else {
                    // default fill value is NaN
                    e.clone()
                        .cast::<f64>()
                        .shift(n, f64::NAN.into(), axis, par)
                        .to_py(self.obj())
                }
            }
            Exprs::OpUsize(_e) => todo!(),
            Exprs::String(e) => e
                .clone()
                .shift(
                    n,
                    fill.unwrap_or_else(|| "".to_owned().into()).cast_string()?,
                    axis,
                    par,
                )
                .to_py(self.obj()),
            Exprs::Str(_) => {
                unimplemented!("shift is not supported for str expression, cast to string first")
            }
            Exprs::TimeDelta(e) => e
                .clone()
                .shift(
                    n,
                    fill.unwrap_or_else(|| "".to_owned().into())
                        .cast_timedelta()?,
                    axis,
                    par,
                )
                .to_py(self.obj()),
            Exprs::DateTime(e) => e
                .clone()
                .shift(
                    n,
                    fill.unwrap_or_else(|| DateTime(Default::default()).into())
                        .cast_datetime(None)?,
                    axis,
                    par,
                )
                .to_py(self.obj()),
            Exprs::Object(e) => e
                .clone()
                .shift(
                    n,
                    fill.unwrap_or_else(|| PyValue(py.None()).into())
                        .cast_object_eager(py)?,
                    axis,
                    par,
                )
                .to_py(self.obj()),
        };
        if let Some(obj) = obj {
            Ok(out.add_obj(obj))
        } else {
            Ok(out)
        }
    }

    #[pyo3(signature=(n=1, axis=0, par=false))]
    pub fn diff(&self, n: i32, axis: i32, par: bool) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            { expr.clone().diff(n, axis, par).to_py(self.obj()) },
            F32,
            I64,
            F64,
            I32
        )
    }

    #[pyo3(signature=(n=1, axis=0, par=false))]
    pub fn pct_change(&self, n: i32, axis: i32, par: bool) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            { expr.clone().pct_change(n, axis, par).to_py(self.obj()) },
            F32,
            I64,
            F64,
            I32
        )
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn count_nan(&self, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().count_nan(axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn count_notnan(&self, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().count_notnan(axis, par).to_py(self.obj())
        })
    }

    // #[args(self, value, axis="0", par=false)]
    // pub fn count_v(&self, value, axis: i32, par: bool) -> Self {
    //     match_exprs!(&self.inner, expr, {expr.clone().count_v(value, axis, par).to_py(self.obj())}, F64, I32)
    // }

    #[allow(unreachable_patterns)]
    #[pyo3(signature=(mask, axis=None, par=false))]
    pub unsafe fn filter(&self, mask: &PyAny, axis: Option<&PyAny>, par: bool) -> PyResult<Self> {
        let mask = parse_expr_nocopy(mask)?;
        let axis = if let Some(axis) = axis {
            parse_expr_nocopy(axis)?
        } else {
            0.into()
        };
        let obj = mask.obj();
        let obj2 = axis.obj();
        Ok(match_exprs!(&self.inner, expr, {
            expr.clone()
                .filter(mask.cast_bool()?, axis.cast_i32()?, par)
                .to_py(self.obj())
                .add_obj(obj)
                .add_obj(obj2)
        }))
    }

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
            0.into()
        };
        let obj = axis.obj();
        Ok(match_exprs!(
            &self.inner,
            expr,
            {
                expr.clone()
                    .dropna(axis.cast_i32()?, how, par)
                    .to_py(self.obj())
                    .add_obj(obj)
            },
            F32,
            F64,
            I32,
            I64,
            Usize
        ))
    }

    pub fn is_nan(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().is_nan().to_py(self.obj())
        })
    }

    pub fn not_nan(&self) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().not_nan().to_py(self.obj())
        })
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn median(&self, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().median(axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn all(&self, axis: i32, par: bool) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            { expr.clone().all(axis, par).to_py(self.obj()) },
            Bool
        )
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn any(&self, axis: i32, par: bool) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            { expr.clone().any(axis, par).to_py(self.obj()) },
            Bool
        )
    }

    #[allow(unreachable_patterns)]
    /// Return a view of the diagonal elements of the array.
    ///
    /// The diagonal is simply the sequence indexed by (0, 0, .., 0), (1, 1, ..., 1) etc as long as all axes have elements.
    ///
    /// Safety
    /// the data for the array view should exist
    pub unsafe fn diag(&self) -> Self {
        match_exprs!(&self.inner, expr, { expr.clone().diag().to_py(self.obj()) })
    }

    #[allow(unreachable_patterns)]
    /// Insert new array axis at axis and return the result.
    pub unsafe fn insert_axis(&self, axis: i32) -> Self {
        match_exprs!(&self.inner, expr, {
            expr.clone().insert_axis(axis).to_py(self.obj())
        })
    }

    #[allow(unreachable_patterns)]
    /// Remove new array axis at axis and return the result.
    pub unsafe fn remove_axis(&self, axis: i32) -> Self {
        match_exprs!(&self.inner, expr, {
            expr.clone().remove_axis(axis).to_py(self.obj())
        })
    }

    #[allow(unreachable_patterns)]
    /// Return a transposed view of the array.
    pub unsafe fn t(&self) -> Self {
        match_exprs!(&self.inner, expr, { expr.clone().t().to_py(self.obj()) })
    }

    #[allow(unreachable_patterns)]
    /// Swap axes ax and bx.
    ///
    /// This does not move any data, it just adjusts the array’s dimensions and strides.
    pub unsafe fn swap_axes(&self, ax: i32, bx: i32) -> Self {
        match_exprs!(&self.inner, expr, {
            expr.clone().swap_axes(ax, bx).to_py(self.obj())
        })
    }

    #[allow(unreachable_patterns)]
    /// Permute the axes.
    ///
    /// This does not move any data, it just adjusts the array’s dimensions and strides.
    ///
    ///i in the j-th place in the axes sequence means self's i-th axis becomes self.permuted_axes()'s j-th axis
    pub unsafe fn permuted_axes(&self, axes: &PyAny) -> PyResult<Self> {
        let axes = parse_expr_nocopy(axes)?;
        let obj = axes.obj();
        let out = match_exprs!(&self.inner, expr, {
            expr.clone()
                .permuted_axes(axes.cast_usize()?)
                .to_py(self.obj())
                .add_obj(obj)
        });
        Ok(out)
    }

    #[allow(unreachable_patterns)]
    /// Return the number of dimensions (axes) in the array
    pub fn ndim(&self) -> Self {
        match_exprs!(&self.inner, expr, { expr.clone().ndim().to_py(self.obj()) })
    }

    #[allow(unreachable_patterns)]
    #[getter]
    /// Return the shape of the array as a usize Expr.
    pub fn shape(&self) -> Self {
        match_exprs!(&self.inner, expr, {
            expr.clone().shape().to_py(self.obj())
        })
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn max(&self, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().max(axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn min(&self, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().min(axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn sum(&self, stable: bool, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().sum(stable, axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn cumsum(&self, stable: bool, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().cumsum(stable, axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn prod(&self, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().prod(axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn cumprod(&self, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().cumprod(axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn mean(&self, stable: bool, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().mean(stable, axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(stable=false, axis=0, par=false, warning=true))]
    pub fn zscore(
        &self,
        stable: bool,
        axis: i32,
        par: bool,
        warning: bool,
        py: Python,
    ) -> PyResult<Self> {
        if warning && !self.is_float() {
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                ("The dtype of input is not Float, so note that the result is not float too",),
            )?;
        }
        let out = match_exprs!(
            &self.inner,
            expr,
            { expr.clone().zscore(stable, axis, par).to_py(self.obj()) },
            F64,
            I32,
            F32,
            I64
        );
        Ok(out)
    }

    #[pyo3(signature=(method=WinsorizeMethod::Quantile, method_params=0.01, stable=false, axis=0, par=false, warning=true))]
    #[allow(clippy::too_many_arguments)]
    pub fn winsorize(
        &self,
        method: WinsorizeMethod,
        method_params: Option<f64>,
        stable: bool,
        axis: i32,
        par: bool,
        warning: bool,
        py: Python,
    ) -> PyResult<Self> {
        if warning && !self.is_float() {
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                ("The dtype of input is not Float, so note that the result is not float too",),
            )?;
        }
        let out = match_exprs!(
            &self.inner,
            expr,
            {
                expr.clone()
                    .winsorize(method, method_params, stable, axis, par)
                    .to_py(self.obj())
            },
            F64,
            I32,
            F32,
            I64
        );
        Ok(out)
    }

    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn var(&self, stable: bool, axis: i32, par: bool) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            { expr.clone().var(stable, axis, par).to_py(self.obj()) },
            F64,
            I32,
            F32,
            I64
        )
    }

    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn std(&self, stable: bool, axis: i32, par: bool) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            { expr.clone().std(stable, axis, par).to_py(self.obj()) },
            F64,
            I32,
            F32,
            I64
        )
    }

    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn skew(&self, stable: bool, axis: i32, par: bool) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            { expr.clone().skew(stable, axis, par).to_py(self.obj()) },
            F64,
            I32,
            F32,
            I64
        )
    }

    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn kurt(&self, stable: bool, axis: i32, par: bool) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            { expr.clone().kurt(stable, axis, par).to_py(self.obj()) },
            F64,
            I32,
            F32,
            I64
        )
    }

    #[pyo3(signature=(pct=false, rev=false, axis=0, par=false))]
    pub fn rank(&self, pct: bool, rev: bool, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().rank(pct, rev, axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(q, method=QuantileMethod::Linear, axis=0, par=false))]
    pub fn quantile(&self, q: f64, method: QuantileMethod, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone()
                .quantile(q, method, axis, par)
                .to_py(self.obj())
        })
    }

    #[pyo3(signature=(rev=false, axis=0, par=false))]
    pub fn argsort(&self, rev: bool, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().argsort(rev, axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(kth, sort=true, rev=false, axis=0, par=false))]
    pub fn arg_partition(&self, kth: usize, sort: bool, rev: bool, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone()
                .arg_partition(kth, sort, rev, axis, par)
                .to_py(self.obj())
        })
    }

    #[pyo3(signature=(group, rev=false, axis=0, par=false))]
    pub fn split_group(&self, group: usize, rev: bool, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone()
                .split_group(group, rev, axis, par)
                .to_py(self.obj())
        })
    }

    #[pyo3(signature=(other, stable=false, axis=0, par=false))]
    pub unsafe fn cov(&self, other: &PyAny, stable: bool, axis: i32, par: bool) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let res = match_exprs!(
            (&self.inner, e1, F64, F32, I64, I32),
            (&other.inner, e2, F64, F32, I64, I32),
            {
                e1.clone()
                    .cov(e2.clone(), stable, axis, par)
                    .to_py(self.obj())
                    .add_obj(obj)
            }
        );
        Ok(res)
    }

    #[pyo3(signature=(other, method=CorrMethod::Pearson, stable=false, axis=0, par=false))]
    pub unsafe fn corr(
        &self,
        other: &PyAny,
        method: CorrMethod,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let res = match_exprs!(
            (&self.inner, e1, F64, F32, I64, I32),
            (&other.inner, e2, F64, F32, I64, I32),
            {
                e1.clone()
                    .corr(e2.clone(), method, stable, axis, par)
                    .to_py(self.obj())
                    .add_obj(obj)
            }
        );
        Ok(res)
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_argmin(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone()
                .ts_argmin(window, min_periods, axis, par)
                .to_py(self.obj())
        })
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_argmax(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone()
                .ts_argmax(window, min_periods, axis, par)
                .to_py(self.obj())
        })
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_min(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone()
                .ts_min(window, min_periods, axis, par)
                .to_py(self.obj())
        })
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_max(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone()
                .ts_max(window, min_periods, axis, par)
                .to_py(self.obj())
        })
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, pct=false, axis=0, par=false))]
    pub fn ts_rank(
        &self,
        window: usize,
        min_periods: usize,
        pct: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        if !pct {
            match_exprs!(numeric & self.inner, expr, {
                expr.clone()
                    .ts_rank(window, min_periods, axis, par)
                    .to_py(self.obj())
            })
        } else {
            match_exprs!(numeric & self.inner, expr, {
                expr.clone()
                    .ts_rank_pct(window, min_periods, axis, par)
                    .to_py(self.obj())
            })
        }
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_prod(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone()
                .ts_prod(window, min_periods, axis, par)
                .to_py(self.obj())
        })
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_prod_mean(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone()
                .ts_prod_mean(window, min_periods, axis, par)
                .to_py(self.obj())
        })
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_minmaxnorm(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            {
                expr.clone()
                    .ts_minmaxnorm(window, min_periods, axis, par)
                    .to_py(self.obj())
            },
            F64,
            I32,
            F32,
            I64
        )
    }

    #[cfg(feature = "window_func")]
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
        let res = match_exprs!(
            (&self.inner, e1, F64, F32, I64, I32),
            (&other.inner, e2, F64, F32, I64, I32),
            {
                e1.clone()
                    .ts_cov(e2.clone(), window, min_periods, stable, axis, par)
                    .to_py(self.obj())
                    .add_obj(obj)
            }
        );
        Ok(res)
    }

    #[cfg(feature = "window_func")]
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
        let res = match_exprs!(
            (&self.inner, e1, F64, F32, I64, I32),
            (&other.inner, e2, F64, F32, I64, I32),
            {
                e1.clone()
                    .ts_corr(e2.clone(), window, min_periods, stable, axis, par)
                    .to_py(self.obj())
                    .add_obj(obj)
            }
        );
        Ok(res)
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_sum(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone()
                .ts_sum(window, min_periods, stable, axis, par)
                .to_py(self.obj())
        })
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_sma(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone()
                .ts_sma(window, min_periods, stable, axis, par)
                .to_py(self.obj())
        })
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_ewm(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone()
                .ts_ewm(window, min_periods, stable, axis, par)
                .to_py(self.obj())
        })
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_wma(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone()
                .ts_wma(window, min_periods, stable, axis, par)
                .to_py(self.obj())
        })
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_std(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            {
                expr.clone()
                    .ts_std(window, min_periods, stable, axis, par)
                    .to_py(self.obj())
            },
            F64,
            F32,
            I64,
            I32
        )
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_var(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            {
                expr.clone()
                    .ts_var(window, min_periods, stable, axis, par)
                    .to_py(self.obj())
            },
            F64,
            I32,
            I64,
            F32
        )
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_skew(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            {
                expr.clone()
                    .ts_skew(window, min_periods, stable, axis, par)
                    .to_py(self.obj())
            },
            F64,
            F32,
            I64,
            I32
        )
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_kurt(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            {
                expr.clone()
                    .ts_kurt(window, min_periods, stable, axis, par)
                    .to_py(self.obj())
            },
            F64,
            F32,
            I64,
            I32
        )
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_stable(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            {
                expr.clone()
                    .ts_stable(window, min_periods, stable, axis, par)
                    .to_py(self.obj())
            },
            F64,
            F32,
            I64,
            I32
        )
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_meanstdnorm(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            {
                expr.clone()
                    .ts_meanstdnorm(window, min_periods, stable, axis, par)
                    .to_py(self.obj())
            },
            F64,
            F32,
            I64,
            I32
        )
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_reg(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            {
                expr.clone()
                    .ts_reg(window, min_periods, stable, axis, par)
                    .to_py(self.obj())
            },
            F64,
            I32,
            F32,
            I64
        )
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_tsf(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            {
                expr.clone()
                    .ts_tsf(window, min_periods, stable, axis, par)
                    .to_py(self.obj())
            },
            F64,
            I32,
            F32,
            I64
        )
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_reg_slope(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            {
                expr.clone()
                    .ts_reg_slope(window, min_periods, stable, axis, par)
                    .to_py(self.obj())
            },
            F64,
            I32,
            F32,
            I64
        )
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, stable=false, axis=0, par=false))]
    pub fn ts_reg_intercept(
        &self,
        window: usize,
        min_periods: usize,
        stable: bool,
        axis: i32,
        par: bool,
    ) -> Self {
        match_exprs!(
            &self.inner,
            expr,
            {
                expr.clone()
                    .ts_reg_intercept(window, min_periods, stable, axis, par)
                    .to_py(self.obj())
            },
            F64,
            I32,
            F32,
            I64
        )
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(method=FillMethod::Ffill, value=None, axis=0, par=false))]
    pub fn fillna(&self, method: FillMethod, value: Option<f64>, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone()
                .fillna(method, value, axis, par)
                .to_py(self.obj())
        })
    }

    #[pyo3(signature=(min, max, axis=0, par=false))]
    pub fn clip(&self, min: f64, max: f64, axis: i32, par: bool) -> Self {
        match_exprs!(numeric & self.inner, expr, {
            expr.clone().clip(min, max, axis, par).to_py(self.obj())
        })
    }

    #[pyo3(signature=(slc, axis=0))]
    pub unsafe fn select(&self, slc: &PyAny, axis: i32) -> PyResult<Self> {
        let slc = parse_expr_nocopy(slc)?;
        self._select_on_axis_by_expr(slc, axis.into())
    }

    #[pyo3(signature=(slc, axis=0, par=false))]
    pub unsafe fn take(&self, slc: &PyAny, axis: i32, par: bool) -> PyResult<Self> {
        let slc = parse_expr_nocopy(slc)?;
        self._take_on_axis_by_expr(slc, axis, par)
    }

    #[cfg(feature = "stat")]
    #[pyo3(signature=(df, loc=None, scale=None))]
    pub unsafe fn t_cdf(&self, df: &PyAny, loc: Option<f64>, scale: Option<f64>) -> PyResult<Self> {
        let df = parse_expr_nocopy(df)?;
        let obj = df.obj();
        let out = match_exprs!(numeric & self.inner, e, {
            e.clone()
                .t_cdf(df.cast_f64()?, loc, scale)
                .to_py(self.obj())
                .add_obj(obj)
        });
        Ok(out)
    }

    #[cfg(feature = "stat")]
    #[pyo3(signature=(mean=None, std=None))]
    pub unsafe fn norm_cdf(&self, mean: Option<f64>, std: Option<f64>) -> PyResult<Self> {
        let out = match_exprs!(numeric & self.inner, e, {
            e.clone().norm_cdf(mean, std).to_py(self.obj())
        });
        Ok(out)
    }

    #[cfg(feature = "stat")]
    #[pyo3(signature=(df1, df2))]
    pub unsafe fn f_cdf(&self, df1: &PyAny, df2: &PyAny) -> PyResult<Self> {
        let df1 = parse_expr_nocopy(df1)?;
        let df2 = parse_expr_nocopy(df2)?;
        let (obj1, obj2) = (df1.obj(), df2.obj());
        let out = match_exprs!(numeric & self.inner, e, {
            e.clone()
                .f_cdf(df1.cast_f64()?, df2.cast_f64()?)
                .to_py(self.obj())
                .add_obj(obj1)
                .add_obj(obj2)
        });
        Ok(out)
    }

    #[pyo3(signature=(by, rev=false, return_idx=false))]
    pub fn sort_by(&self, by: Vec<Self>, rev: bool, return_idx: bool) -> Self {
        if !return_idx {
            self.sort_by_expr(by, rev)
        } else {
            self.sort_by_expr_idx(by, rev)
        }
    }

    pub fn cast(&self, ty: &PyAny, py: Python) -> PyResult<Self> {
        if let Ok(ty_name) = ty.extract::<&str>() {
            self.cast_by_str(ty_name, py)
        } else if let Ok(py_type) = ty.extract::<&pyo3::types::PyType>() {
            self.cast_by_str(py_type.name().unwrap(), py)
        } else {
            unimplemented!("Incorrect type for casting")
        }
    }

    #[pyo3(signature=(mask, value, axis=0, par=false))]
    pub unsafe fn put_mask(
        self: PyRef<Self>,
        mask: &PyAny,
        value: &PyAny,
        axis: i32,
        par: bool,
        py: Python,
    ) -> PyResult<Self> {
        let (mask, value) = (parse_expr_nocopy(mask)?, parse_expr_nocopy(value)?);
        let (obj1, obj2) = (mask.obj(), value.obj());
        let mask = mask.cast_bool().expect("Can not cast mask to bool.");
        let rtn = match self.inner() {
            Exprs::F64(expr) => expr
                .clone()
                .put_mask(mask, value.cast_f64()?, axis, par)
                .to_py_ref(self, py),
            Exprs::F32(expr) => expr
                .clone()
                .put_mask(mask, value.cast_f32()?, axis, par)
                .to_py_ref(self, py),
            Exprs::I64(expr) => expr
                .clone()
                .put_mask(mask, value.cast_i64()?, axis, par)
                .to_py_ref(self, py),
            Exprs::I32(expr) => expr
                .clone()
                .put_mask(mask, value.cast_i32()?, axis, par)
                .to_py_ref(self, py),
            Exprs::Bool(expr) => expr
                .clone()
                .put_mask(mask, value.cast_bool()?, axis, par)
                .to_py_ref(self, py),
            Exprs::Usize(expr) => expr
                .clone()
                .put_mask(mask, value.cast_usize()?, axis, par)
                .to_py_ref(self, py),
            Exprs::Str(expr) => expr
                .clone()
                .put_mask(mask, value.cast_str()?, axis, par)
                .to_py_ref(self, py),
            Exprs::String(expr) => expr
                .clone()
                .put_mask(mask, value.cast_string()?, axis, par)
                .to_py_ref(self, py),
            Exprs::DateTime(expr) => expr
                .clone()
                .put_mask(mask, value.cast_datetime(None)?, axis, par)
                .to_py_ref(self, py),
            Exprs::TimeDelta(expr) => expr
                .clone()
                .put_mask(mask, value.cast_timedelta()?, axis, par)
                .to_py_ref(self, py),
            Exprs::Object(expr) => expr
                .clone()
                .put_mask(mask, value.cast_object_eager(py)?, axis, par)
                .to_py_ref(self, py),
            _ => unimplemented!("put_mask is not implemented for this type."),
        };
        Ok(rtn.add_obj(obj1).add_obj(obj2))
    }

    #[pyo3(signature=(mask, value, par=false))]
    pub unsafe fn where_(&self, mask: &PyAny, value: &PyAny, par: bool) -> PyResult<PyExpr> {
        let mask = parse_expr_nocopy(mask)?;
        let value = parse_expr_nocopy(value)?;
        where_(self.clone(), mask, value, par)
    }

    #[pyo3(signature=(con, then))]
    pub unsafe fn if_then(&self, con: &PyAny, then: &PyAny, py: Python) -> PyResult<PyExpr> {
        let con = parse_expr_nocopy(con)?;
        let then = parse_expr_nocopy(then)?;
        let con = con.cast_bool()?;
        let out: PyExpr = match self.inner() {
            Exprs::F64(e) => e.clone().if_then(con, then.cast_f64()?).to_py(self.obj()),
            Exprs::F32(e) => e.clone().if_then(con, then.cast_f32()?).to_py(self.obj()),
            Exprs::I64(e) => e.clone().if_then(con, then.cast_i64()?).to_py(self.obj()),
            Exprs::I32(e) => e.clone().if_then(con, then.cast_i32()?).to_py(self.obj()),
            Exprs::Usize(e) => e.clone().if_then(con, then.cast_usize()?).to_py(self.obj()),
            Exprs::Bool(e) => e.clone().if_then(con, then.cast_bool()?).to_py(self.obj()),
            Exprs::Str(e) => e.clone().if_then(con, then.cast_str()?).to_py(self.obj()),
            Exprs::String(e) => e
                .clone()
                .if_then(con, then.cast_string()?)
                .to_py(self.obj()),
            Exprs::DateTime(e) => e
                .clone()
                .if_then(con, then.cast_datetime(None)?)
                .to_py(self.obj()),
            Exprs::TimeDelta(e) => e
                .clone()
                .if_then(con, then.cast_timedelta()?)
                .to_py(self.obj()),
            Exprs::Object(e) => e
                .clone()
                .if_then(con, then.cast_object_eager(py)?)
                .to_py(self.obj()),
            _ => unimplemented!("if_then is not implemented for this type."),
        };
        Ok(out)
    }

    #[cfg(feature = "blas")]
    pub unsafe fn lstsq(&self, y: &PyAny) -> PyResult<Self> {
        let y = parse_expr_nocopy(y)?;
        let obj = y.obj();
        match_exprs!(
            (&self.inner, x, F64, F32, I64, I32, Usize),
            (y.inner, y, F64, F32, I64, I32, Usize),
            { Ok(x.clone().lstsq(y).to_py(self.obj()).add_obj(obj)) }
        )
    }

    #[cfg(feature = "blas")]
    #[allow(unreachable_patterns)]
    pub fn params(&self) -> PyResult<Self> {
        match_exprs!(&self.inner, e, { Ok(e.clone().params().to_py(self.obj())) })
    }

    #[cfg(feature = "blas")]
    #[allow(unreachable_patterns)]
    pub fn singular_values(&self) -> PyResult<Self> {
        match_exprs!(&self.inner, e, {
            Ok(e.clone().singular_values().to_py(self.obj()))
        })
    }

    #[cfg(feature = "blas")]
    #[allow(unreachable_patterns)]
    pub fn ols_rank(&self) -> PyResult<Self> {
        match_exprs!(&self.inner, e, {
            Ok(e.clone().ols_rank().to_py(self.obj()))
        })
    }

    #[cfg(feature = "blas")]
    #[allow(unreachable_patterns)]
    pub fn sse(&self) -> PyResult<Self> {
        match_exprs!(&self.inner, e, { Ok(e.clone().sse().to_py(self.obj())) })
    }

    #[cfg(feature = "blas")]
    #[allow(unreachable_patterns)]
    pub fn fitted_values(&self) -> PyResult<Self> {
        match_exprs!(&self.inner, e, {
            Ok(e.clone().fitted_values().to_py(self.obj()))
        })
    }

    #[cfg(feature = "blas")]
    #[allow(unreachable_patterns)]
    #[pyo3(signature=(full=true, calc_uvt=true, split=true))]
    pub fn svd(&self, full: bool, calc_uvt: bool, split: bool, py: Python) -> PyResult<PyObject> {
        if split {
            match_exprs!(numeric & self.inner, e, {
                let out = e.clone().svd(full, calc_uvt);
                let mut out = out
                    .split_vec_base(1 + 2 * calc_uvt as usize)
                    .into_iter()
                    .map(|e| e.to_py(self.obj()))
                    .collect_trusted();
                if out.len() == 1 {
                    Ok(out.pop().into_py(py))
                } else {
                    Ok(out.into_py(py))
                }
            })
        } else {
            Ok(match_exprs!(numeric & self.inner, e, {
                e.clone().svd(full, calc_uvt).to_py(self.obj())
            })
            .into_py(py))
        }
    }

    #[cfg(feature = "blas")]
    #[allow(unreachable_patterns)]
    #[pyo3(signature=(return_s=false, r_cond=None, split=true))]
    pub fn pinv(
        &self,
        return_s: bool,
        r_cond: Option<f64>,
        split: bool,
        py: Python,
    ) -> PyResult<PyObject> {
        if split {
            match_exprs!(numeric & self.inner, e, {
                let out = e.clone().pinv(r_cond, return_s);
                let mut out = out
                    .split_vec_base(1 + return_s as usize)
                    .into_iter()
                    .map(|e| e.to_py(self.obj()))
                    .collect_trusted();
                if out.len() == 1 {
                    Ok(out.pop().into_py(py))
                } else {
                    Ok(out.into_py(py))
                }
            })
        } else {
            Ok(match_exprs!(numeric & self.inner, e, {
                e.clone().pinv(r_cond, return_s).to_py(self.obj())
            })
            .into_py(py))
        }
    }

    #[allow(unreachable_patterns)]
    pub unsafe fn broadcast(&self, shape: &PyAny) -> PyResult<Self> {
        let shape = parse_expr_nocopy(shape)?;
        let obj = shape.obj();
        match_exprs!(self.inner(), e, {
            Ok(e.clone()
                .broadcast(shape.cast_usize()?)
                .to_py(self.obj())
                .add_obj(obj))
        })
    }

    #[allow(unreachable_patterns)]
    pub unsafe fn broadcast_with(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let shape = other.shape();
        let obj = shape.obj();
        match_exprs!(self.inner(), e, {
            Ok(e.clone()
                .broadcast(shape.cast_usize()?)
                .to_py(self.obj())
                .add_obj(obj))
        })
    }

    #[pyo3(name = "concat")]
    #[pyo3(signature=(other, axis=0))]
    pub unsafe fn concat_py(&self, other: &PyAny, axis: i32) -> PyResult<Self> {
        let other = parse_expr_list(other, false)?;
        self.concat(other, axis)
    }

    #[pyo3(name = "stack")]
    #[pyo3(signature=(other, axis=0))]
    pub unsafe fn stack_py(&self, other: &PyAny, axis: i32) -> PyResult<Self> {
        let other = parse_expr_list(other, false)?;
        self.stack(other, axis)
    }

    pub fn offset_by(&self, delta: &str) -> PyResult<Self> {
        let delta: Expr<'static, TimeDelta> = TimeDelta::parse(delta).into();
        Ok((self.clone().cast_datetime(None)? + delta).to_py(self.obj()))
    }

    pub fn strptime(&self, fmt: String) -> PyResult<Self> {
        Ok((self.clone().cast_string()?.strptime(fmt)).to_py(self.obj()))
    }

    pub fn strftime(&self, fmt: Option<String>) -> PyResult<Self> {
        Ok((self.clone().cast_datetime(None)?.strftime(fmt)).to_py(self.obj()))
    }

    pub fn round(&self, precision: u32) -> PyResult<Self> {
        let out = match_exprs!(
            &self.inner,
            e,
            {
                e.clone()
                    .chain_view_f::<_, f64>(
                        move |arr| {
                            arr.mapv(|v| {
                                let scale = 10.pow(precision) as f64;
                                (v.f64() * scale).round() / scale
                            })
                            .into()
                        },
                        RefType::False,
                    )
                    .to_py(self.obj())
            },
            F64,
            I32,
            F32,
            I64,
            Usize
        );
        Ok(out)
    }

    pub fn round_string(&self, precision: usize) -> PyResult<Self> {
        let out = match_exprs!(
            &self.inner,
            e,
            {
                e.clone()
                    .chain_view_f::<_, String>(
                        move |arr| arr.map(|v| format!("{v:.precision$}")).into(),
                        RefType::False,
                    )
                    .to_py(self.obj())
            },
            F64,
            I32,
            F32,
            I64,
            Usize
        );
        Ok(out)
    }

    #[pyo3(signature=(index, duration, func, axis=0, **py_kwargs))]
    pub unsafe fn rolling_apply_by_time(
        &self,
        index: &PyAny,
        duration: &str,
        func: &PyAny,
        axis: i32,
        py_kwargs: Option<&PyDict>,
        py: Python,
    ) -> PyResult<PyObject> {
        let index_expr = parse_expr_nocopy(index)?;
        let mut rolling_idx = index_expr.cast_datetime(None)?.time_rolling(duration);
        rolling_idx.eval_inplace();
        let mut column_num = 0;
        let mut output = rolling_idx
            .view_arr()
            .to_dim1()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(end, start)| {
                let pye = self.take_by_slice(Some(axis), start, end, None);
                let res = func
                    .call((pye,), py_kwargs)
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
        // we don't need to add object for `out_data` because we no longer need a reference of `index`.
        let mut out_data: Vec<PyExpr> = (0..column_num)
            .into_par_iter()
            .map(|i| {
                let group_vec = output
                    .iter()
                    .map(|single_output_exprs| single_output_exprs.get(i).unwrap().no_dim0())
                    .collect_trusted();
                concat_expr(group_vec, axis).expect("concat expr error")
            })
            .collect();
        if out_data.len() == 1 {
            Ok(out_data.pop().unwrap().into_py(py))
        } else {
            Ok(out_data.into_py(py))
        }
    }

    #[allow(unreachable_patterns)]
    #[pyo3(signature=(window, func, axis=0, **py_kwargs))]
    pub unsafe fn rolling_apply(
        &mut self,
        window: usize,
        func: &PyAny,
        axis: i32,
        py_kwargs: Option<&PyDict>,
        py: Python,
    ) -> PyResult<PyObject> {
        if window == 0 {
            return Err(PyValueError::new_err("Window should be greater than 0"));
        }
        let mut column_num = 0;
        self.eval_inplace();
        let axis_n = match_exprs!(&self.inner, expr, { expr.view_arr().norm_axis(axis) });
        let length = match_exprs!(&self.inner, expr, {
            expr.view_arr().shape()[axis_n.index()]
        });
        let mut output = zip(repeat(0).take(window - 1), 0..window - 1)
            .chain((window - 1..length).enumerate())
            .map(|(end, start)| {
                let pye = self.take_by_slice(Some(axis), start, end, None);
                let res = func
                    .call((pye,), py_kwargs)
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
        // we don't need to add object for `out_data` because we no longer need a reference of `index`.
        let mut out_data: Vec<PyExpr> = (0..column_num)
            .into_par_iter()
            .map(|i| {
                let group_vec = output
                    .iter()
                    .map(|single_output_exprs| single_output_exprs.get(i).unwrap().no_dim0())
                    .collect_trusted();
                concat_expr(group_vec, axis).expect("Concat expr error")
            })
            .collect();
        if out_data.len() == 1 {
            Ok(out_data.pop().unwrap().into_py(py))
        } else {
            Ok(out_data.into_py(py))
        }
    }
}
