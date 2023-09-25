use super::export::*;
use super::pyfunc::{parse_expr, parse_expr_list, parse_expr_nocopy, where_};
use crate::from_py::{NoDim0, PyContext};
use ahash::{HashMap, HashMapExt};
use ndarray::SliceInfoElem;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use pyo3::{exceptions::PyAttributeError, pyclass::CompareOp, PyTraverseError, PyVisit};

#[cfg(feature = "window_func")]
use tears::lazy::RollingTimeStartBy;

use tears::{
    match_all, match_arrok, ArrOk, Data, DropNaMethod, StrError, TimeDelta, WinsorizeMethod,
};

// #[cfg(feature = "option_dtype")]
// use tears::{OptF32, OptF64, OptI32, OptI64};

static PYEXPR_ATTRIBUTE: Lazy<Mutex<HashMap<String, PyObject>>> =
    Lazy::new(|| Mutex::new(HashMap::<String, PyObject>::with_capacity(10)));

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
    #[pyo3(signature=(expr, name=None, copy=false))]
    pub unsafe fn new(expr: &PyAny, name: Option<String>, copy: bool) -> PyResult<Self> {
        let mut out = parse_expr(expr, copy)?;
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
    #[allow(unreachable_patterns)]
    pub fn get_base_type(&self) -> &'static str {
        // match_exprs!(&self.inner, e, { e.get_base_type() })
        self.e.base_type()
    }

    // #[getter]
    // #[allow(unreachable_patterns)]
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
    pub fn view_in(
        self: PyRefMut<Self>,
        context: Option<&PyAny>,
        py: Python,
    ) -> PyResult<PyObject> {
        let ct: PyContext<'static> = if let Some(context) = context {
            unsafe { std::mem::transmute(context.extract::<PyContext>()?) }
        } else {
            Default::default()
        };
        let (ct_rs, _obj_map) = (ct.ct, ct.obj_map);
        let data = self.e.view_data(ct_rs.as_ref()).map_err(StrError::to_py)?;
        let container = unsafe { PyAny::from_borrowed_ptr(py, self.as_ptr()) };
        if matches!(&data, Data::ArrVec(_)) {
            if let Data::ArrVec(arr_vec) = data {
                let out = arr_vec
                    .iter()
                    .map(|arr| {
                        match_arrok!(pyelement arr, a, {
                            unsafe{
                                PyArray::borrow_from_array(&a.view().0, container)
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
            let arr = match_arrok!(arr, a, { a.view().to_object(py) }, Str, String, TimeDelta);
            return PyArray::from_owned_array(py, arr.0).no_dim0(py);
        } else if let ArrOk::DateTime(arr) = arr {
            let arr = arr
                .view()
                .map(|v| v.into_np_datetime::<numpy::datetime::units::Microseconds>());
            return PyArray::from_owned_array(py, arr.0).no_dim0(py);
        }
        match_arrok!(
            pyelement arr,
            a,
            {
                unsafe {
                    Ok(PyArray::borrow_from_array(
                        &a.view().0,
                        container,
                    )
                    .no_dim0(py)?)
                }
            }
        )
    }

    #[getter]
    #[allow(unreachable_patterns)]
    pub fn get_view(self: PyRefMut<Self>, py: Python) -> PyResult<PyObject> {
        self.view_in(None, py)
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
    pub fn value<'py>(
        &'py mut self,
        unit: Option<&'py str>,
        context: Option<&'py PyAny>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let ct: PyContext<'static> = if let Some(context) = context {
            unsafe { std::mem::transmute(context.extract::<PyContext>()?) }
        } else {
            Default::default()
        };
        let (ct_rs, _obj_map) = (ct.ct, ct.obj_map);
        // let mut expr = self.clone();
        self.e.eval_inplace(ct_rs.clone())?;
        let data = self.e.view_data(ct_rs.as_ref()).map_err(StrError::to_py)?;
        // let mut expr = self.e.clone();
        if matches!(&data, Data::ArrVec(_)) {
            if let Data::ArrVec(_) = data {
                let arr_vec = data.view_arr_vec(ct_rs.as_ref()).map_err(StrError::to_py)?;
                let out = arr_vec
                    .into_iter()
                    .map(|arr| {
                        match_arrok!(pyelement arr, a, {
                            PyArray::from_owned_array(py, a.view().to_owned().0)
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
            let arr = match_arrok!(arr, a, { a.view().to_object(py) }, Str, String, TimeDelta);
            return PyArray::from_owned_array(py, arr.0).no_dim0(py);
        } else if let ArrOk::DateTime(arr) = &arr {
            match unit.unwrap_or("us").to_lowercase().as_str() {
                "ms" => {
                    let arr = arr
                        .view()
                        .map(|v| v.into_np_datetime::<numpy::datetime::units::Milliseconds>());
                    return PyArray::from_owned_array(py, arr.0).no_dim0(py);
                }
                "us" => {
                    let arr = arr
                        .view()
                        .map(|v| v.into_np_datetime::<numpy::datetime::units::Microseconds>());
                    return PyArray::from_owned_array(py, arr.0).no_dim0(py);
                }
                "ns" => {
                    let arr = arr
                        .view()
                        .map(|v| v.into_np_datetime::<numpy::datetime::units::Nanoseconds>());
                    return PyArray::from_owned_array(py, arr.0).no_dim0(py);
                }
                _ => unimplemented!("not support datetime unit"),
            }
        }
        match_arrok!(
            pyelement arr,
            a,
            {
                Ok(PyArray::from_owned_array(
                    py,
                    a.view().to_owned().0
                )
                .no_dim0(py)?)
            }
        )
    }

    #[getter]
    #[allow(unreachable_patterns)]
    pub fn step(&self) -> usize {
        self.e.step_acc()
    }

    // #[getter]
    // #[allow(unreachable_patterns)]
    // pub fn step_acc(&self) -> usize {
    //     match_exprs!(&self.inner, expr, { expr.step_acc() })
    // }

    #[getter]
    #[allow(unreachable_patterns)]
    pub fn get_name(&self) -> Option<String> {
        self.e.name()
        // match_exprs!(&self.inner, expr, { expr.name() })
    }

    #[setter]
    #[allow(unreachable_patterns)]
    pub fn set_name(&mut self, name: String) {
        self.e.rename(name)
        // match_exprs!(&mut self.inner, expr, { expr.rename(name) })
    }

    #[pyo3(name = "copy")]
    #[allow(unreachable_patterns)]
    pub fn deep_copy(&self) -> Self {
        let mut out = self.clone();
        out.e.deep_copy();
        out
    }

    #[allow(unreachable_patterns)]
    pub(crate) fn is_owned(&self) -> bool {
        self.e.is_owned()
    }

    #[allow(unreachable_patterns)]
    pub unsafe fn reshape(&self, shape: &PyAny) -> PyResult<Self> {
        let shape = parse_expr_nocopy(shape)?;
        let obj = shape.obj();
        let mut out = self.clone();
        out.e.reshape(shape.e);
        Ok(out.add_obj_into(obj))
    }

    // #[allow(unreachable_patterns)]
    // pub fn strong_count(&mut self) -> usize {
    //     match_exprs!(&self.inner, expr, { expr.strong_count() })
    // }

    // #[allow(unreachable_patterns)]
    // pub fn weak_count(&mut self) -> usize {
    //     match_exprs!(&self.inner, expr, { expr.weak_count() })
    // }

    // #[allow(unreachable_patterns)]
    // pub fn ref_count(&mut self) -> usize {
    //     match_exprs!(&self.inner, expr, { expr.ref_count() })
    // }

    // #[allow(unreachable_patterns)]
    // pub fn hint_arr_type(&mut self) -> Self {
    //     match_exprs!(&self.inner, expr, { expr.clone().hint_arr_type().into() })
    // }

    pub fn __getattr__<'py>(
        self: PyRef<'py, Self>,
        attr: &'py str,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let attr_dict = PYEXPR_ATTRIBUTE.lock();
        let res = attr_dict.get(attr);
        if let Some(res) = res {
            let func = res.clone();
            let functools = py.import("functools")?;
            let partial = functools.getattr("partial")?;
            partial.call1((func, self))
        } else {
            Err(PyAttributeError::new_err(format!(
                "'PyExpr' object has no attribute {attr}"
            )))
        }
    }

    fn __add__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((self.clone().e + other.e).to_py(obj).add_obj_into(obj2))
    }

    fn __radd__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((other.e + self.clone().e).to_py(obj).add_obj_into(obj2))
    }

    unsafe fn __iadd__(&mut self, other: &PyAny) {
        *self = self.__add__(other).unwrap()
    }

    fn __sub__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((self.clone().e - other.e).to_py(obj).add_obj_into(obj2))
    }

    fn __rsub__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((other.e - self.clone().e).to_py(obj).add_obj_into(obj2))
    }

    unsafe fn __isub__(&mut self, other: &PyAny) {
        *self = self.__sub__(other).unwrap()
    }

    fn __mul__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((self.clone().e * other.e).to_py(obj).add_obj_into(obj2))
    }

    unsafe fn __rmul__(&self, other: &PyAny) -> PyResult<Self> {
        self.__mul__(other)
    }

    unsafe fn __imul__(&mut self, other: &PyAny) {
        *self = self.__mul__(other).unwrap()
    }

    fn __truediv__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((self.clone().e / other.e).to_py(obj).add_obj_into(obj2))
    }

    fn __rtruediv__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((other.e / self.clone().e).to_py(obj).add_obj_into(obj2))
    }

    unsafe fn __itruediv__(&mut self, other: &PyAny) {
        *self = self.__truediv__(other).unwrap()
    }

    fn __and__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((self.clone().e & other.e).to_py(obj).add_obj_into(obj2))
    }

    fn __or__(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let (obj, obj2) = (self.obj(), other.obj());
        Ok((self.clone().e | other.e).to_py(obj).add_obj_into(obj2))
    }

    unsafe fn __rand__(&self, other: &PyAny) -> PyResult<Self> {
        self.__and__(other)
    }

    unsafe fn __iand__(&mut self, other: &PyAny) {
        *self = self.__and__(other).unwrap()
    }

    unsafe fn __ror__(&self, other: &PyAny) -> PyResult<Self> {
        self.__or__(other)
    }

    unsafe fn __ior__(&mut self, other: &PyAny) {
        *self = self.__or__(other).unwrap()
    }

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

    unsafe fn __pow__(&self, other: &PyAny, _mod: &PyAny) -> PyResult<Self> {
        self.pow_py(other, false)
    }

    pub fn __round__(&self, precision: u32) -> PyResult<Self> {
        self.round(precision)
    }

    #[pyo3(name="pow", signature=(other, par=false))]
    unsafe fn pow_py(&self, other: &PyAny, par: bool) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let mut out = self.clone();
        out.e.pow(other.e, par);
        Ok(out.add_obj_into(obj))
    }

    fn abs(&self) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.abs();
        Ok(out)
    }

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

    fn __neg__(&self) -> PyResult<Self> {
        let e = self.clone();
        Ok((-e.e).to_py(self.obj()))
    }

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
    #[allow(unreachable_patterns)]
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
        let mut e = self.clone();
        e.e.sqrt();
        e
    }

    /// Returns the cube root of each element.
    pub fn cbrt(&self) -> Self {
        let mut e = self.clone();
        e.e.cbrt();
        e
    }

    /// Returns the sign of each element.
    fn sign(&self) -> Self {
        let mut e = self.clone();
        e.e.sign();
        e
    }

    /// Returns the natural logarithm of each element.
    pub fn ln(&self) -> Self {
        let mut e = self.clone();
        e.e.ln();
        e
    }

    /// Returns ln(1+n) (natural logarithm) more accurately than if the operations were performed separately.
    pub fn ln_1p(&self) -> Self {
        let mut e = self.clone();
        e.e.ln_1p();
        e
    }

    /// Returns the base 2 logarithm of each element.
    pub fn log2(&self) -> Self {
        let mut e = self.clone();
        e.e.log2();
        e
    }

    /// Returns the base 10 logarithm of each element.
    pub fn log10(&self) -> Self {
        let mut e = self.clone();
        e.e.log10();
        e
    }

    /// Returns e^(self) of each element, (the exponential function).
    pub fn exp(&self) -> Self {
        let mut e = self.clone();
        e.e.exp();
        e
    }

    /// Returns 2^(self) of each element.
    pub fn exp2(&self) -> Self {
        let mut e = self.clone();
        e.e.exp2();
        e
    }

    /// Returns e^(self) - 1 in a way that is accurate even if the number is close to zero.
    pub fn exp_m1(&self) -> Self {
        let mut e = self.clone();
        e.e.exp_m1();
        e
    }

    /// Computes the arccosine of each element. Return value is in radians in the range 0,
    /// pi or NaN if the number is outside the range -1, 1.
    pub fn acos(&self) -> Self {
        let mut e = self.clone();
        e.e.acos();
        e
    }

    /// Computes the arcsine of each element. Return value is in radians in the range -pi/2,
    /// pi/2 or NaN if the number is outside the range -1, 1.
    pub fn asin(&self) -> Self {
        let mut e = self.clone();
        e.e.asin();
        e
    }

    /// Computes the arctangent of each element. Return value is in radians in the range -pi/2, pi/2;
    pub fn atan(&self) -> Self {
        let mut e = self.clone();
        e.e.atan();
        e
    }

    /// Computes the sine of each element (in radians).
    pub fn sin(&self) -> Self {
        let mut e = self.clone();
        e.e.sin();
        e
    }

    /// Computes the cosine of each element (in radians).
    pub fn cos(&self) -> Self {
        let mut e = self.clone();
        e.e.cos();
        e
    }

    /// Computes the tangent of each element (in radians).
    pub fn tan(&self) -> Self {
        let mut e = self.clone();
        e.e.tan();
        e
    }

    /// Returns the smallest integer greater than or equal to `self`.
    pub fn ceil(&self) -> Self {
        let mut e = self.clone();
        e.e.ceil();
        e
    }

    /// Returns the largest integer less than or equal to `self`.
    pub fn floor(&self) -> Self {
        let mut e = self.clone();
        e.e.floor();
        e
    }

    /// Returns the fractional part of each element.
    pub fn fract(&self) -> Self {
        let mut e = self.clone();
        e.e.fract();
        e
    }

    /// Returns the integer part of each element. This means that non-integer numbers are always truncated towards zero.
    pub fn trunc(&self) -> Self {
        let mut e = self.clone();
        e.e.trunc();
        e
    }

    /// Returns true if this number is neither infinite nor NaN
    pub fn is_finite(&self) -> Self {
        let mut e = self.clone();
        e.e.is_finite();
        e
    }

    /// Returns true if this value is positive infinity or negative infinity, and false otherwise.
    pub fn is_inf(&self) -> Self {
        let mut e = self.clone();
        e.e.is_infinite();
        e
    }

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

    #[allow(unreachable_patterns)]
    pub fn first(&self) -> Self {
        let mut e = self.clone();
        e.e.first();
        e
    }

    #[allow(unreachable_patterns)]
    pub fn last(&self) -> Self {
        let mut e = self.clone();
        e.e.last();
        e
    }

    // #[pyo3(signature=(axis=0))]
    // #[allow(unreachable_patterns)]
    // pub fn first(&self, axis: i32) -> Self {
    //     let mut e = self.clone();
    //     e.e.first(axis);
    //     e
    // }

    // #[pyo3(signature=(axis=0))]
    // #[allow(unreachable_patterns)]
    // pub fn last(&self, axis: i32) -> Self {
    //     let mut e = self.clone();
    //     e.e.last(axis);
    //     e
    // }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn valid_first(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.valid_first(axis, par);
        e
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn valid_last(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.valid_last(axis, par);
        e
    }

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
        let obj = fill.clone().map(|e| e.obj());
        let mut out = self.clone();
        out.e.shift(n.into(), fill.map(|f| f.e), axis, par);
        if let Some(obj) = obj {
            Ok(out.add_obj_into(obj))
        } else {
            Ok(out)
        }
    }

    #[pyo3(signature=(n=1, axis=0, par=false))]
    pub fn diff(&self, n: i32, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.diff(n, axis, par);
        e
    }

    #[pyo3(signature=(n=1, axis=0, par=false))]
    pub fn pct_change(&self, n: i32, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.pct_change(n, axis, par);
        e
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn count_nan(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.count_nan(axis, par);
        e
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn count_notnan(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.count_notnan(axis, par);
        e
    }

    #[pyo3(signature=(value, axis=0, par=false))]
    pub fn count_value(&self, value: &PyAny, axis: i32, par: bool) -> PyResult<Self> {
        let value = parse_expr_nocopy(value)?;
        let obj = value.obj();
        let mut e = self.clone();
        e.e.count_value(value.e, axis, par);
        Ok(e.add_obj_into(obj))
    }

    #[allow(unreachable_patterns)]
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

    pub fn is_nan(&self) -> Self {
        let mut out = self.clone();
        out.e.is_nan();
        out
    }

    pub fn not_nan(&self) -> Self {
        let mut out = self.clone();
        out.e.not_nan();
        out
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn median(&self, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.median(axis, par);
        out
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn all(&self, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.all(axis, par);
        out
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn any(&self, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.any(axis, par);
        out
    }

    #[allow(unreachable_patterns)]
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

    #[allow(unreachable_patterns)]
    /// Insert new array axis at axis and return the result.
    pub unsafe fn insert_axis(&self, axis: i32) -> Self {
        let mut out = self.clone();
        out.e.insert_axis(axis);
        out
    }

    #[allow(unreachable_patterns)]
    /// Remove new array axis at axis and return the result.
    pub unsafe fn remove_axis(&self, axis: i32) -> Self {
        let mut out = self.clone();
        out.e.remove_axis(axis);
        out
    }

    #[allow(unreachable_patterns)]
    /// Return a transposed view of the array.
    pub unsafe fn t(&self) -> Self {
        let mut out = self.clone();
        out.e.t();
        out
    }

    #[allow(unreachable_patterns)]
    /// Swap axes ax and bx.
    ///
    /// This does not move any data, it just adjusts the array’s dimensions and strides.
    pub unsafe fn swap_axes(&self, ax: i32, bx: i32) -> Self {
        let mut out = self.clone();
        out.e.swap_axes(ax, bx);
        out
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
        let mut out = self.clone();
        out.e.permuted_axes(axes.e);
        Ok(out.add_obj_into(obj))
    }

    #[allow(unreachable_patterns)]
    /// Return the number of dimensions (axes) in the array
    pub fn ndim(&self) -> Self {
        let mut out = self.clone();
        out.e.ndim();
        out
    }

    #[allow(unreachable_patterns)]
    #[getter]
    /// Return the shape of the array as a usize Expr.
    pub fn shape(&self) -> Self {
        let mut out = self.clone();
        out.e.shape();
        out
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn max(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.max(axis, par);
        e
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn min(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.min(axis, par);
        e
    }

    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn sum(&self, stable: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.sum(stable, axis, par);
        e
    }

    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn cumsum(&self, stable: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.cumsum(stable, axis, par);
        e
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn prod(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.prod(axis, par);
        e
    }

    #[pyo3(signature=(axis=0, par=false))]
    pub fn cumprod(&self, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.cumprod(axis, par);
        e
    }

    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn mean(&self, stable: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.mean(stable, axis, par);
        e
    }

    #[pyo3(signature=(stable=false, axis=0, par=false, _warning=true))]
    pub fn zscore(
        &self,
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
        e.e.zscore_inplace(stable, axis, par);
        Ok(e)
    }

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
        e.e.winsorize_inplace(method, method_params, stable, axis, par);
        Ok(e)
    }

    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn var(&self, stable: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.var(stable, axis, par);
        e
    }

    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn std(&self, stable: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.std(stable, axis, par);
        e
    }

    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn skew(&self, stable: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.skew(stable, axis, par);
        e
    }

    #[pyo3(signature=(stable=false, axis=0, par=false))]
    pub fn kurt(&self, stable: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.kurt(stable, axis, par);
        e
    }

    #[pyo3(signature=(pct=false, rev=false, axis=0, par=false))]
    pub fn rank(&self, pct: bool, rev: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.rank(pct, rev, axis, par);
        e
    }

    #[pyo3(signature=(q, method=QuantileMethod::Linear, axis=0, par=false))]
    pub fn quantile(&self, q: f64, method: QuantileMethod, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.quantile(q, method, axis, par);
        e
    }

    #[pyo3(signature=(rev=false, axis=0, par=false))]
    pub fn argsort(&self, rev: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.argsort(rev, axis, par);
        e
    }

    #[pyo3(signature=(kth, sort=true, rev=false, axis=0, par=false))]
    pub fn arg_partition(&self, kth: usize, sort: bool, rev: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.arg_partition(kth, sort, rev, axis, par);
        e
    }

    #[pyo3(signature=(kth, sort=true, rev=false, axis=0, par=false))]
    pub fn partition(&self, kth: usize, sort: bool, rev: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.partition(kth, sort, rev, axis, par);
        e
    }

    #[pyo3(signature=(group, rev=false, axis=0, par=false))]
    pub fn split_group(&self, group: usize, rev: bool, axis: i32, par: bool) -> Self {
        let mut e = self.clone();
        e.e.split_group(group, rev, axis, par);
        e
    }

    #[pyo3(signature=(other, stable=false, axis=0, par=false))]
    pub unsafe fn cov(&self, other: &PyAny, stable: bool, axis: i32, par: bool) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let obj = other.obj();
        let mut e = self.clone();
        e.e.cov(other.e, stable, axis, par);
        Ok(e.add_obj_into(obj))
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
        let mut e = self.clone();
        e.e.corr(other.e, method, stable, axis, par);
        Ok(e.add_obj_into(obj))
    }

    #[pyo3(signature=(keep="first".to_string()))]
    pub fn _get_sorted_unique_idx(&self, keep: String) -> Self {
        let mut out = self.clone();
        out.e.get_sorted_unique_idx(keep);
        out
        // match_exprs!(&self.inner, expr, {
        //     expr.clone().get_sorted_unique_idx(keep).to_py(self.obj())
        // })
    }

    pub fn sorted_unique(&self) -> Self {
        let mut out = self.clone();
        out.e.sorted_unique();
        out
    }

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

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_argmin(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.ts_argmin(window, min_periods, axis, par);
        out
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_argmax(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.ts_argmax(window, min_periods, axis, par);
        out
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_min(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.ts_min(window, min_periods, axis, par);
        out
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_max(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.ts_max(window, min_periods, axis, par);
        out
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
        let mut out = self.clone();
        if !pct {
            out.e.ts_rank(window, min_periods, axis, par);
        } else {
            out.e.ts_rank_pct(window, min_periods, axis, par);
        }
        out
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_prod(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.ts_prod(window, min_periods, axis, par);
        out
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_prod_mean(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.ts_prod_mean(window, min_periods, axis, par);
        out
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(window, min_periods=1, axis=0, par=false))]
    pub fn ts_minmaxnorm(&self, window: usize, min_periods: usize, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.ts_minmaxnorm(window, min_periods, axis, par);
        out
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
        let mut out = self.clone();
        out.e
            .ts_cov(other.e, window, min_periods, stable, axis, par);
        Ok(out.add_obj_into(obj))
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
        let mut out = self.clone();
        out.e
            .ts_corr(other.e, window, min_periods, stable, axis, par);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(feature = "window_func")]
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

    #[cfg(feature = "window_func")]
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

    #[cfg(feature = "window_func")]
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

    #[cfg(feature = "window_func")]
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
        let mut out = self.clone();
        out.e.ts_sum(window, min_periods, stable, axis, par);
        out
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
        let mut out = self.clone();
        out.e.ts_sma(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "window_func")]
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
        let mut out = self.clone();
        out.e.ts_ewm(window, min_periods, stable, axis, par);
        out
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
        let mut out = self.clone();
        out.e.ts_wma(window, min_periods, stable, axis, par);
        out
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
        let mut out = self.clone();
        out.e.ts_std(window, min_periods, stable, axis, par);
        out
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
        let mut out = self.clone();
        out.e.ts_var(window, min_periods, stable, axis, par);
        out
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
        let mut out = self.clone();
        out.e.ts_skew(window, min_periods, stable, axis, par);
        out
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
        let mut out = self.clone();
        out.e.ts_kurt(window, min_periods, stable, axis, par);
        out
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
        let mut out = self.clone();
        out.e.ts_stable(window, min_periods, stable, axis, par);
        out
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
        let mut out = self.clone();
        out.e.ts_meanstdnorm(window, min_periods, stable, axis, par);
        out
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
        let mut out = self.clone();
        out.e.ts_reg(window, min_periods, stable, axis, par);
        out
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
        let mut out = self.clone();
        out.e.ts_tsf(window, min_periods, stable, axis, par);
        out
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
        let mut out = self.clone();
        out.e.ts_reg_slope(window, min_periods, stable, axis, par);
        out
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
        let mut out = self.clone();
        out.e
            .ts_reg_intercept(window, min_periods, stable, axis, par);
        out
    }

    #[cfg(feature = "window_func")]
    #[pyo3(signature=(method=FillMethod::Ffill, value=None, axis=0, par=false))]
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

    #[pyo3(signature=(min, max, axis=0, par=false))]
    pub fn clip(&self, min: f64, max: f64, axis: i32, par: bool) -> Self {
        let mut out = self.clone();
        out.e.clip(min.into(), max.into(), axis, par);
        out
    }

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

    #[cfg(feature = "stat")]
    #[pyo3(signature=(df, loc=None, scale=None))]
    pub unsafe fn t_cdf(&self, df: &PyAny, loc: Option<f64>, scale: Option<f64>) -> PyResult<Self> {
        let df = parse_expr_nocopy(df)?;
        let obj = df.obj();
        let mut out = self.clone();
        out.e.t_cdf(df.e, loc, scale);
        Ok(out.add_obj_into(obj))
    }

    #[cfg(feature = "stat")]
    #[pyo3(signature=(mean=None, std=None))]
    pub unsafe fn norm_cdf(&self, mean: Option<f64>, std: Option<f64>) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.norm_cdf(mean, std);
        Ok(out)
    }

    #[cfg(feature = "stat")]
    #[pyo3(signature=(df1, df2))]
    pub unsafe fn f_cdf(&self, df1: &PyAny, df2: &PyAny) -> PyResult<Self> {
        let df1 = parse_expr_nocopy(df1)?;
        let df2 = parse_expr_nocopy(df2)?;
        let (obj1, obj2) = (df1.obj(), df2.obj());
        let mut out = self.clone();
        out.e.f_cdf(df1.e, df2.e);
        Ok(out.add_obj_into(obj1).add_obj_into(obj2))
    }

    #[pyo3(signature=(by, rev=false, return_idx=false))]
    pub fn sort(&self, by: &PyAny, rev: bool, return_idx: bool) -> PyResult<Self> {
        let by = unsafe { parse_expr_list(by, false) }?;
        let obj_vec = by.iter().map(|e| e.obj()).collect_trusted();
        let by = by.into_iter().map(|e| e.e).collect_trusted();
        let mut out = self.clone();
        if !return_idx {
            out.e.sort(by, rev);
        } else {
            out.e.get_sort_idx(by, rev);
        }
        Ok(out.add_obj_vec_into(obj_vec))
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

    #[pyo3(signature=(mask, value, par=false))]
    pub unsafe fn where_(&self, mask: &PyAny, value: &PyAny, par: bool) -> PyResult<PyExpr> {
        let mask = parse_expr_nocopy(mask)?;
        let value = parse_expr_nocopy(value)?;
        where_(self.clone(), mask, value, par)
    }

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
    #[allow(unreachable_patterns)]
    pub fn params(&self) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.params();
        Ok(out)
    }

    #[cfg(feature = "blas")]
    #[allow(unreachable_patterns)]
    pub fn singular_values(&self) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.singular_values();
        Ok(out)
    }

    #[cfg(feature = "blas")]
    #[allow(unreachable_patterns)]
    pub fn ols_rank(&self) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.ols_rank();
        Ok(out)
    }

    #[cfg(feature = "blas")]
    #[allow(unreachable_patterns)]
    pub fn sse(&self) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.sse();
        Ok(out)
    }

    #[cfg(feature = "blas")]
    #[allow(unreachable_patterns)]
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

    #[allow(unreachable_patterns)]
    pub unsafe fn broadcast(&self, shape: &PyAny) -> PyResult<Self> {
        let shape = parse_expr_nocopy(shape)?;
        let obj = shape.obj();
        let mut out = self.clone();
        out.e.broadcast(shape.e);
        Ok(out.add_obj_into(obj))
    }

    #[allow(unreachable_patterns)]
    pub unsafe fn broadcast_with(&self, other: &PyAny) -> PyResult<Self> {
        let other = parse_expr_nocopy(other)?;
        let shape = other.shape();
        let obj = shape.obj();
        let mut out = self.clone();
        out.e.broadcast_with(other.e);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(name = "concat")]
    #[pyo3(signature=(other, axis=0))]
    pub unsafe fn concat_py(&self, other: &PyAny, axis: i32) -> PyResult<Self> {
        let other = parse_expr_list(other, false)?;
        let obj_vec = other.iter().map(|e| e.obj()).collect_trusted();
        let other = other.into_iter().map(|e| e.e).collect_trusted();
        let mut out = self.clone();
        out.e.concat(other, axis);
        Ok(out.add_obj_vec_into(obj_vec))
    }

    #[pyo3(name = "stack")]
    #[pyo3(signature=(other, axis=0))]
    pub unsafe fn stack_py(&self, other: &PyAny, axis: i32) -> PyResult<Self> {
        let other = parse_expr_list(other, false)?;
        let obj_vec = other.iter().map(|e| e.obj()).collect_trusted();
        let other = other.into_iter().map(|e| e.e).collect_trusted();
        let mut out = self.clone();
        out.e.stack(other, axis);
        Ok(out.add_obj_vec_into(obj_vec))
    }

    pub fn offset_by(&self, delta: &str) -> PyResult<Self> {
        let delta: Expr<'static> = TimeDelta::parse(delta).into();
        let out = self.clone();
        Ok((out.e + delta).to_py(self.obj()))
    }

    pub fn strptime(&self, fmt: String) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.strptime(fmt);
        Ok(out)
    }

    pub fn strftime(&self, fmt: Option<String>) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.strftime(fmt);
        Ok(out)
    }

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

    #[cfg(feature = "window_func")]
    pub unsafe fn _get_fix_window_rolling_idx(&self, window: &PyAny) -> PyResult<Self> {
        let mut length = self.clone();
        length.e.len();
        let mut window = parse_expr(window, true)?;
        Expr::get_fix_window_rolling_idx(&mut window.e, length.e);
        Ok(window.add_obj_into(self.obj()))
    }

    #[pyo3(signature=(duration, start_by=RollingTimeStartBy::Full))]
    #[cfg(feature = "window_func")]
    pub unsafe fn _get_time_rolling_idx(
        &self,
        duration: &str,
        start_by: RollingTimeStartBy,
    ) -> PyResult<Self> {
        let mut out = self.clone();
        out.e.get_time_rolling_idx(duration, start_by);
        Ok(out)
    }

    #[pyo3(signature=(agg_expr, roll_start, others=None))]
    #[cfg(feature = "window_func")]
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
        out.e
            .rolling_apply_with_start(agg_expr.e, roll_start.e, others, false);
        out.add_obj(obj1).add_obj(obj2).add_obj_vec(others_obj_vec);
        Ok(out)
    }

    #[pyo3(signature=(duration, closed="right".to_owned(), split=true))]
    pub unsafe fn _get_time_groupby_info(
        &self,
        duration: &str,
        closed: String,
        split: bool,
        py: Python,
    ) -> PyResult<PyObject> {
        let mut out = self.clone();
        out.e.get_group_by_time_info(duration, closed);
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

    #[pyo3(signature=(agg_expr, group_info, others=None))]
    pub unsafe fn group_by_time(
        &self,
        agg_expr: &PyAny,
        group_info: &PyAny,
        others: Option<&PyAny>,
    ) -> PyResult<Self> {
        let agg_expr = parse_expr_nocopy(agg_expr)?;
        let group_info = parse_expr_nocopy(group_info)?;
        let others = if let Some(others) = others {
            Some(parse_expr_list(others, false)?)
        } else {
            None
        };
        let obj1 = agg_expr.obj();
        let obj2 = group_info.obj();
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
        out.e.group_by_time(agg_expr.e, group_info.e, others);
        out.add_obj(obj1).add_obj(obj2).add_obj_vec(others_obj_vec);
        Ok(out)
    }

    #[pyo3(signature=(groupby_info, stable=false))]
    pub unsafe fn _group_by_time_mean(&self, groupby_info: &PyAny, stable: bool) -> PyResult<Self> {
        let groupby_info = parse_expr_nocopy(groupby_info)?;
        let obj = groupby_info.obj();
        let mut out = self.clone();
        out.e.group_by_time_mean(groupby_info.e, stable);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(groupby_info, stable=false))]
    pub unsafe fn _group_by_time_sum(&self, groupby_info: &PyAny, stable: bool) -> PyResult<Self> {
        let groupby_info = parse_expr_nocopy(groupby_info)?;
        let obj = groupby_info.obj();
        let mut out = self.clone();
        out.e.group_by_time_sum(groupby_info.e, stable);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(groupby_info, stable=false))]
    pub unsafe fn _group_by_time_std(&self, groupby_info: &PyAny, stable: bool) -> PyResult<Self> {
        let groupby_info = parse_expr_nocopy(groupby_info)?;
        let obj = groupby_info.obj();
        let mut out = self.clone();
        out.e.group_by_time_std(groupby_info.e, stable);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(groupby_info, stable=false))]
    pub unsafe fn _group_by_time_var(&self, groupby_info: &PyAny, stable: bool) -> PyResult<Self> {
        let groupby_info = parse_expr_nocopy(groupby_info)?;
        let obj = groupby_info.obj();
        let mut out = self.clone();
        out.e.group_by_time_var(groupby_info.e, stable);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(groupby_info))]
    pub unsafe fn _group_by_time_min(&self, groupby_info: &PyAny) -> PyResult<Self> {
        let groupby_info = parse_expr_nocopy(groupby_info)?;
        let obj = groupby_info.obj();
        let mut out = self.clone();
        out.e.group_by_time_min(groupby_info.e);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(groupby_info))]
    pub unsafe fn _group_by_time_max(&self, groupby_info: &PyAny) -> PyResult<Self> {
        let groupby_info = parse_expr_nocopy(groupby_info)?;
        let obj = groupby_info.obj();
        let mut out = self.clone();
        out.e.group_by_time_max(groupby_info.e);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(groupby_info))]
    pub unsafe fn _group_by_time_first(&self, groupby_info: &PyAny) -> PyResult<Self> {
        let groupby_info = parse_expr_nocopy(groupby_info)?;
        let obj = groupby_info.obj();
        let mut out = self.clone();
        out.e.group_by_time_first(groupby_info.e);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(groupby_info))]
    pub unsafe fn _group_by_time_last(&self, groupby_info: &PyAny) -> PyResult<Self> {
        let groupby_info = parse_expr_nocopy(groupby_info)?;
        let obj = groupby_info.obj();
        let mut out = self.clone();
        out.e.group_by_time_last(groupby_info.e);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(groupby_info))]
    pub unsafe fn _group_by_time_valid_first(&self, groupby_info: &PyAny) -> PyResult<Self> {
        let groupby_info = parse_expr_nocopy(groupby_info)?;
        let obj = groupby_info.obj();
        let mut out = self.clone();
        out.e.group_by_time_valid_first(groupby_info.e);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(groupby_info))]
    pub unsafe fn _group_by_time_valid_last(&self, groupby_info: &PyAny) -> PyResult<Self> {
        let groupby_info = parse_expr_nocopy(groupby_info)?;
        let obj = groupby_info.obj();
        let mut out = self.clone();
        out.e.group_by_time_valid_last(groupby_info.e);
        Ok(out.add_obj_into(obj))
    }
    #[pyo3(signature=(groupby_info, other, method=CorrMethod::Pearson, stable=false))]
    pub unsafe fn _group_by_time_corr(
        &self,
        groupby_info: &PyAny,
        other: &PyAny,
        method: CorrMethod,
        stable: bool,
    ) -> PyResult<Self> {
        let groupby_info = parse_expr_nocopy(groupby_info)?;
        let other = parse_expr_nocopy(other)?;
        let obj = groupby_info.obj();
        let obj2 = other.obj();
        let mut out = self.clone();
        out.e
            .group_by_time_corr(other.e, groupby_info.e, method, stable);
        out.add_obj(obj).add_obj(obj2);
        Ok(out)
    }

    #[pyo3(signature=(window, offset))]
    #[cfg(feature = "window_func")]
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

    #[pyo3(signature=(roll_start, stable=false))]
    pub unsafe fn _rolling_select_mean(&self, roll_start: &PyAny, stable: bool) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_mean(roll_start.e, stable);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(roll_start, stable=false))]
    pub unsafe fn _rolling_select_sum(&self, roll_start: &PyAny, stable: bool) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_sum(roll_start.e, stable);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(roll_start, stable=false))]
    pub unsafe fn _rolling_select_std(&self, roll_start: &PyAny, stable: bool) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_std(roll_start.e, stable);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(roll_start, stable=false))]
    pub unsafe fn _rolling_select_var(&self, roll_start: &PyAny, stable: bool) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_var(roll_start.e, stable);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(roll_start))]
    pub unsafe fn _rolling_select_min(&self, roll_start: &PyAny) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_min(roll_start.e);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(roll_start))]
    pub unsafe fn _rolling_select_max(&self, roll_start: &PyAny) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_max(roll_start.e);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(roll_start))]
    pub unsafe fn _rolling_select_umin(&self, roll_start: &PyAny) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_umin(roll_start.e);
        Ok(out.add_obj_into(obj))
    }

    #[pyo3(signature=(roll_start))]
    pub unsafe fn _rolling_select_umax(&self, roll_start: &PyAny) -> PyResult<Self> {
        let roll_start = parse_expr_nocopy(roll_start)?;
        let obj = roll_start.obj();
        let mut out = self.clone();
        out.e.rolling_select_umax(roll_start.e);
        Ok(out.add_obj_into(obj))
    }

    // #[pyo3(signature=(idxs))]
    // pub unsafe fn _rolling_select_by_vecusize_sum(&self, idxs: &PyAny) -> PyResult<Self> {
    //     let idxs = parse_expr_nocopy(idxs)?;
    //     let obj = idxs.obj();
    //     let out = match_exprs!(numeric & self.inner, e, {
    //         e.clone()
    //             .rolling_select_by_vecusize_sum(idxs.cast_vecusize()?)
    //             .to_py(self.obj())
    //             .add_obj_into(obj)
    //     });
    //     Ok(out)
    // }

    // #[pyo3(signature=(idxs))]
    // pub unsafe fn _rolling_select_by_vecusize_mean(&self, idxs: &PyAny) -> PyResult<Self> {
    //     let idxs = parse_expr_nocopy(idxs)?;
    //     let obj = idxs.obj();
    //     let out = match_exprs!(numeric & self.inner, e, {
    //         e.clone()
    //             .rolling_select_by_vecusize_mean(idxs.cast_vecusize()?)
    //             .to_py(self.obj())
    //             .add_obj_into(obj)
    //     });
    //     Ok(out)
    // }

    // #[pyo3(signature=(idxs))]
    // pub unsafe fn _rolling_select_by_vecusize_max(&self, idxs: &PyAny) -> PyResult<Self> {
    //     let idxs = parse_expr_nocopy(idxs)?;
    //     let obj = idxs.obj();
    //     let out = match_exprs!(numeric & self.inner, e, {
    //         e.clone()
    //             .rolling_select_by_vecusize_max(idxs.cast_vecusize()?)
    //             .to_py(self.obj())
    //             .add_obj_into(obj)
    //     });
    //     Ok(out)
    // }

    // #[pyo3(signature=(idxs))]
    // pub unsafe fn _rolling_select_by_vecusize_min(&self, idxs: &PyAny) -> PyResult<Self> {
    //     let idxs = parse_expr_nocopy(idxs)?;
    //     let obj = idxs.obj();
    //     let out = match_exprs!(numeric & self.inner, e, {
    //         e.clone()
    //             .rolling_select_by_vecusize_min(idxs.cast_vecusize()?)
    //             .to_py(self.obj())
    //             .add_obj_into(obj)
    //     });
    //     Ok(out)
    // }

    // #[pyo3(signature=(index, duration, func, axis=0, **py_kwargs))]
    // pub unsafe fn rolling_apply_by_time(
    //     &self,
    //     index: &PyAny,
    //     duration: &str,
    //     func: &PyAny,
    //     axis: i32,
    //     py_kwargs: Option<&PyDict>,
    //     py: Python,
    // ) -> PyResult<PyObject> {
    //     let index_expr = parse_expr_nocopy(index)?;
    //     let mut rolling_idx = index_expr
    //         .cast_datetime(None)?
    //         .get_time_rolling_idx(duration, RollingTimeStartBy::Full);
    //     rolling_idx = rolling_idx.eval(None)?.0;
    //     let mut column_num = 0;
    //     let mut output = rolling_idx
    //         .view_arr()
    //         .to_dim1()
    //         .unwrap()
    //         .into_iter()
    //         .enumerate()
    //         .map(|(end, start)| {
    //             let pye = self.select_by_slice_eager(Some(axis), start, end, None);
    //             let res = func
    //                 .call((pye,), py_kwargs)
    //                 .expect("Call python function error!");
    //             let res = unsafe {
    //                 parse_expr_list(res, false).expect("Can not parse fucntion return as Expr list")
    //             };
    //             column_num = res.len();
    //             res
    //         })
    //         .collect_trusted();
    //     let eval_res: Vec<_> = output
    //         .par_iter_mut()
    //         .flatten()
    //         .map(|e| e.eval_inplace(None))
    //         .collect();
    //     if eval_res.iter().any(|e| e.is_err()) {
    //         return Err(PyRuntimeError::new_err(
    //             "Some of the expressions can't be evaluated",
    //         ));
    //     }
    //     // we don't need to add object for `out_data` because we no longer need a reference of `index`.
    //     let mut out_data: Vec<PyExpr> = (0..column_num)
    //         .into_par_iter()
    //         .map(|i| {
    //             let group_vec = output
    //                 .iter()
    //                 .map(|single_output_exprs| single_output_exprs.get(i).unwrap().no_dim0())
    //                 .collect_trusted();
    //             concat_expr(group_vec, axis).expect("concat expr error")
    //         })
    //         .collect();
    //     if out_data.len() == 1 {
    //         Ok(out_data.pop().unwrap().into_py(py))
    //     } else {
    //         Ok(out_data.into_py(py))
    //     }
    // }

    // #[allow(unreachable_patterns)]
    // #[pyo3(signature=(window, func, axis=0, **py_kwargs))]
    // pub unsafe fn rolling_apply(
    //     &mut self,
    //     window: usize,
    //     func: &PyAny,
    //     axis: i32,
    //     py_kwargs: Option<&PyDict>,
    //     py: Python,
    // ) -> PyResult<PyObject> {
    //     if window == 0 {
    //         return Err(PyValueError::new_err("Window should be greater than 0"));
    //     }
    //     let mut column_num = 0;
    //     self.eval_inplace(None)?;
    //     let axis_n = match_exprs!(&self.inner, expr, { expr.view_arr().norm_axis(axis) });
    //     let length = match_exprs!(&self.inner, expr, {
    //         expr.view_arr().shape()[axis_n.index()]
    //     });
    //     let mut output = zip(repeat(0).take(window - 1), 0..window - 1)
    //         .chain((window - 1..length).enumerate())
    //         .map(|(end, start)| {
    //             let pye = self.select_by_slice_eager(Some(axis), start, end, None);
    //             let res = func
    //                 .call((pye,), py_kwargs)
    //                 .expect("Call python function error!");
    //             let res = unsafe {
    //                 parse_expr_list(res, false).expect("Can not parse fucntion return as Expr list")
    //             };
    //             column_num = res.len();
    //             res
    //         })
    //         .collect_trusted();
    //     let eval_res: Vec<_> = output
    //         .par_iter_mut()
    //         .flatten()
    //         .map(|e| e.eval_inplace(None))
    //         .collect();
    //     if eval_res.iter().any(|e| e.is_err()) {
    //         return Err(PyRuntimeError::new_err(
    //             "Some of the expressions can't be evaluated",
    //         ));
    //     }
    //     // we don't need to add object for `out_data` because we no longer need a reference of `index`.
    //     let mut out_data: Vec<PyExpr> = (0..column_num)
    //         .into_par_iter()
    //         .map(|i| {
    //             let group_vec = output
    //                 .iter()
    //                 .map(|single_output_exprs| single_output_exprs.get(i).unwrap().no_dim0())
    //                 .collect_trusted();
    //             concat_expr(group_vec, axis).expect("Concat expr error")
    //         })
    //         .collect();
    //     if out_data.len() == 1 {
    //         Ok(out_data.pop().unwrap().into_py(py))
    //     } else {
    //         Ok(out_data.into_py(py))
    //     }
    // }
}
