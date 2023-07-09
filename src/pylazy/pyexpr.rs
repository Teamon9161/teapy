use std::cmp::Ordering;
use std::ops::Deref;

use ndarray::{Slice, SliceInfoElem};

use super::export::*;
use crate::{
    arr::{ArbArray, DateTime, ExprElement, Number, RefType, TimeDelta, TimeUnit},
    from_py::PyValue,
};

#[pyclass(subclass)]
#[derive(Clone, Default)]
pub struct PyExpr {
    pub inner: Exprs<'static>,
    pub obj: Option<Vec<PyObject>>,
}

impl From<Exprs<'static>> for PyExpr {
    fn from(e: Exprs<'static>) -> Self {
        PyExpr {
            inner: e,
            obj: None,
        }
    }
}

impl<T: ExprElement + 'static> From<T> for PyExpr {
    fn from(v: T) -> Self {
        let e: Expr<'static, T> = Expr::new(v.into(), None);
        e.into()
    }
}

impl Deref for PyExpr {
    type Target = Exprs<'static>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: ExprElement> From<Expr<'static, T>> for PyExpr
where
    Exprs<'static>: From<Expr<'static, T>>,
{
    fn from(e: Expr<'static, T>) -> Self {
        let e: Exprs<'static> = e.into();
        e.into()
    }
}

impl<T: ExprElement + 'static> Expr<'static, T> {
    pub fn to_py(self, obj: Option<Vec<PyObject>>) -> PyExpr {
        let exprs: Exprs<'static> = self.into();
        PyExpr { inner: exprs, obj }
    }

    pub fn to_py_ref(self, obj: PyRef<PyExpr>, py: Python) -> PyExpr {
        let exprs: Exprs<'static> = self.into();
        PyExpr {
            inner: exprs,
            obj: Some(vec![obj.into_py(py)]),
        }
    }
}

impl Exprs<'static> {
    pub fn to_py(self, obj: Option<Vec<PyObject>>) -> PyExpr {
        PyExpr { inner: self, obj }
    }
}

impl PyExpr {
    #[inline]
    pub fn obj(&self) -> Option<Vec<PyObject>> {
        self.obj.clone()
    }

    #[inline]
    pub fn inner(&self) -> &Exprs<'static> {
        &self.inner
    }

    pub fn add_obj(mut self, mut another: Option<Vec<PyObject>>) -> Self {
        if let Some(obj) = &mut self.obj {
            if let Some(another) = &mut another {
                obj.append(another);
            }
        } else if let Some(obj) = another {
            self.obj = Some(obj);
        }
        self
    }

    pub fn add_obj_vec(mut self, another: Vec<Option<Vec<PyObject>>>) -> Self {
        for obj in another {
            self = self.add_obj(obj)
        }
        self
    }

    // Cast the output of the expression to f64 ndarray
    pub fn cast_f64(self) -> PyResult<Expr<'static, f64>> {
        if let Ok(v) = self.inner.cast_f64() {
            Ok(v)
        } else {
            Err(PyValueError::new_err("Can not cast the output to f64"))
        }
    }

    // Cast the output of the expression to f32 ndarray
    pub fn cast_f32(self) -> PyResult<Expr<'static, f32>> {
        if let Ok(v) = self.inner.cast_f32() {
            Ok(v)
        } else {
            Err(PyValueError::new_err("Can not cast the output to f32"))
        }
    }

    // Cast the output of the expression to i64 ndarray
    pub fn cast_i64(self) -> PyResult<Expr<'static, i64>> {
        if let Ok(v) = self.inner.cast_i64() {
            Ok(v)
        } else {
            Err(PyValueError::new_err("Can not cast the output to i64"))
        }
    }

    // Cast the output of the expression to i32 ndarray
    pub fn cast_i32(self) -> PyResult<Expr<'static, i32>> {
        if let Ok(v) = self.inner.cast_i32() {
            Ok(v)
        } else {
            Err(PyValueError::new_err("Can not cast the output to i32"))
        }
    }

    // Cast the output of the expression to usize ndarray
    pub fn cast_usize(self) -> PyResult<Expr<'static, usize>> {
        if let Ok(v) = self.inner.cast_usize() {
            Ok(v)
        } else {
            Err(PyValueError::new_err("Can not cast the output to usize"))
        }
    }

    // Cast the output of the expression to object ndarray lazily
    pub fn cast_object(self) -> PyResult<Expr<'static, PyValue>> {
        if let Ok(v) = self.inner.cast_object() {
            Ok(v)
        } else {
            Err(PyValueError::new_err(
                "Can not cast the output to object lazily",
            ))
        }
    }

    // Cast the output of the expression to object ndarray
    pub fn cast_object_eager(self, py: Python) -> PyResult<Expr<'static, PyValue>> {
        if let Ok(v) = self.inner.cast_object_eager(py) {
            Ok(v)
        } else {
            Err(PyValueError::new_err("Can not cast the output to object"))
        }
    }

    // Cast the output of the expression to datetime ndarray
    pub fn cast_datetime(self, unit: Option<TimeUnit>) -> PyResult<Expr<'static, DateTime>> {
        if let Ok(v) = self.inner.cast_datetime(unit) {
            Ok(v)
        } else {
            Err(PyValueError::new_err("Can not cast the output to datetime"))
        }
    }

    // Cast the output of the expression to timedelta ndarray
    pub fn cast_timedelta(self) -> PyResult<Expr<'static, TimeDelta>> {
        if let Ok(v) = self.inner.cast_timedelta() {
            Ok(v)
        } else {
            Err(PyValueError::new_err(
                "Can not cast the output to timedelta",
            ))
        }
    }

    // Cast the output of the expression to datetime ndarray
    pub fn cast_datetime_default(self) -> PyResult<Expr<'static, DateTime>> {
        self.cast_datetime(Default::default())
    }

    // Cast the output of the expression to str ndarray
    pub fn cast_str(self) -> PyResult<Expr<'static, &'static str>> {
        if let Ok(v) = self.inner.cast_str() {
            Ok(v)
        } else {
            Err(PyValueError::new_err("Can not cast the output to str"))
        }
    }

    // Cast the output of the expression to string ndarray
    pub fn cast_string(self) -> PyResult<Expr<'static, String>> {
        if let Ok(v) = self.inner.cast_string() {
            Ok(v)
        } else {
            Err(PyValueError::new_err("Can not cast the output to string"))
        }
    }

    // Cast the output of the expression to bool ndarray
    pub fn cast_bool(self) -> PyResult<Expr<'static, bool>> {
        if let Ok(v) = self.inner.cast_bool() {
            Ok(v)
        } else {
            Err(PyValueError::new_err("Can not cast the output to bool"))
        }
    }

    #[allow(unreachable_patterns, dead_code)]
    pub fn eval(mut self) -> Self {
        self.eval_inplace();
        self
        // match_exprs!(self.inner, expr, {
        //     let expr = expr.eval();
        //     if let Some(owned) = expr.owned() {
        //         if owned {
        //             // we don't need to reference
        //             expr.into()
        //         } else {
        //             expr.to_py(self.obj)
        //         }
        //     } else {
        //         expr.to_py(self.obj)
        //     }
        // })
    }

    #[allow(unreachable_patterns)]
    pub fn eval_inplace(&mut self) {
        match_exprs!(&mut self.inner, expr, {
            expr.eval_inplace();
            if let Some(owned) = expr.owned() {
                if owned {
                    self.obj = None
                } // we don't need to reference
            }
        })
    }

    #[allow(unreachable_patterns)]
    pub fn _select_on_axis_by_expr(&self, slc: Self, axis: Self) -> PyResult<Self> {
        let obj = slc.obj();
        match_exprs!(&self.inner, expr, {
            Ok(expr
                .clone()
                .select_on_axis_by_expr(slc.cast_usize()?, axis.cast_i32()?)
                .to_py(self.obj())
                .add_obj(obj))
        })
    }

    pub fn _select_on_axis(&self, slc: Vec<usize>, axis: i32) -> Self {
        let slc_expr: Expr<usize> = slc.into();
        self._select_on_axis_by_expr(slc_expr.into(), axis.into())
            .unwrap()
    }

    /// take values using slice
    ///
    /// # Safety
    ///
    /// the data for the array view should exist
    #[allow(unreachable_patterns)]
    pub unsafe fn view_by_slice(&self, slc: Vec<SliceInfoElem>) -> Self {
        // match_exprs!(&self.inner, expr, {
        //     expr.clone().view_by_slice(slc).to_py_ref(self, py)
        // })
        match_exprs!(&self.inner, expr, {
            expr.clone().view_by_slice(slc).to_py(self.obj())
        })
    }

    #[allow(unreachable_patterns)]
    pub fn _take_on_axis_by_expr(&self, slc: Self, axis: i32, par: bool) -> PyResult<Self> {
        let obj = slc.obj();
        match_exprs!(&self.inner, expr, {
            Ok(expr
                .clone()
                .take_on_axis_by_expr(slc.cast_usize()?, axis, par)
                .to_py(self.obj())
                .add_obj(obj))
        })
    }

    #[allow(unreachable_patterns)]
    /// # Safety
    ///
    /// The slice must be valid
    pub unsafe fn _take_on_axis_by_expr_unchecked(
        &self,
        slc: Self,
        axis: i32,
        par: bool,
    ) -> PyResult<Self> {
        let obj = slc.obj();
        match_exprs!(&self.inner, expr, {
            Ok(expr
                .clone()
                .take_on_axis_by_expr_unchecked(slc.cast_usize()?, axis, par)
                .to_py(self.obj())
                .add_obj(obj))
        })
    }

    pub fn _take_on_axis(&self, slc: Vec<usize>, axis: i32, par: bool) -> Self {
        let slc_expr: Expr<usize> = slc.into();
        self._take_on_axis_by_expr(slc_expr.into(), axis, par)
            .unwrap()
    }

    #[allow(unreachable_patterns)]
    pub fn sort_by_expr(&self, by: Vec<PyExpr>, rev: bool) -> Self {
        let obj_vec: Vec<Option<Vec<PyObject>>> = by.iter().map(|e| e.obj()).collect();
        let mut out = match_exprs!(&self.inner, expr, {
            expr.clone().sort_by_expr(by, rev).to_py(self.obj())
        });
        for obj in obj_vec {
            out = out.add_obj(obj);
        }
        out
    }

    #[allow(unreachable_patterns)]
    pub fn sort_by_expr_idx(&self, by: Vec<PyExpr>, rev: bool) -> Self {
        let obj_vec: Vec<Option<Vec<PyObject>>> = by.iter().map(|e| e.obj()).collect();
        let mut out = match_exprs!(&self.inner, expr, {
            expr.clone().sort_by_expr_idx(by, rev).to_py(self.obj())
        });
        for obj in obj_vec {
            out = out.add_obj(obj);
        }
        out
    }

    #[allow(unreachable_patterns)]
    pub fn no_dim0(&self) -> Self {
        match_exprs!(&self.inner, expr, {
            expr.clone().no_dim0().to_py(self.obj())
        })
    }

    pub fn pow(&self, other: PyExpr, par: bool) -> PyResult<Self> {
        let obj = other.obj();
        let out: PyExpr = if other.inner.is_int() {
            match_exprs!(
                &self.inner,
                e1,
                {
                    e1.clone()
                        .cast::<f64>()
                        .powi(other.cast_i32()?, par)
                        .to_py(self.obj())
                        .add_obj(obj)
                },
                F64,
                I32,
                F32,
                I64
            )
        } else {
            match_exprs!(
                (&self.inner, e1, F64, F32, I32, I64),
                (other.inner, e2, F64, F32),
                {
                    e1.clone()
                        .cast::<f64>()
                        .powf(e2.cast::<f64>(), par)
                        .to_py(self.obj())
                        .add_obj(obj)
                }
            )
        };
        Ok(out)
    }

    pub(crate) fn cast_by_str(&self, ty_name: &str, py: Python) -> PyResult<Self> {
        match ty_name.to_lowercase().as_str() {
            "float" | "f64" => Ok(self.clone().cast_f64()?.to_py(self.obj())),
            "f32" => Ok(self.clone().cast_f32()?.to_py(self.obj())),
            "int" | "i32" => Ok(self.clone().cast_i32()?.to_py(self.obj())),
            "i64" => Ok(self.clone().cast_i64()?.to_py(self.obj())),
            "usize" | "uint" => Ok(self.clone().cast_usize()?.to_py(self.obj())),
            "bool" => Ok(self.clone().cast_bool()?.to_py(self.obj())),
            "object" => Ok(self.clone().cast_object_eager(py)?.to_py(self.obj())),
            "str" => {
                if self.is_object() {
                    Ok(self
                        .clone()
                        .cast_object()?
                        .object_to_string(py)
                        .to_py(self.obj()))
                } else {
                    Ok(self.clone().cast_string()?.to_py(self.obj()))
                }
            }
            "datetime" => Ok(self.clone().cast_datetime_default()?.to_py(self.obj())),
            "timedelta" => Ok(self.clone().cast_timedelta()?.to_py(self.obj())),
            _ => Err(PyValueError::new_err(format!(
                "cast to type: {ty_name} is not implemented"
            ))),
        }
    }

    #[allow(unreachable_patterns)]
    pub fn concat(&self, other: Vec<PyExpr>, axis: i32) -> PyResult<PyExpr> {
        let obj_vec = other.iter().map(|e| e.obj()).collect_trusted();
        macro_rules! concat_macro {
            ($({$arm: ident => $cast_func: ident $(($arg: expr))?, $name: expr}),*) => {
                match &self.inner {
                    $(Exprs::$arm(expr) => {
                        let other = other.into_iter().map(|e| e.$cast_func($($arg)?).expect(&format!("can not cast to {:?}", $name))).collect_trusted();
                        expr.clone().concat(other, axis).to_py(self.obj())
                    }),*
                    _ => unimplemented!("concat for this dtype is not implemented")
                }
            };
        }
        let rtn = concat_macro!(
            {F64 => cast_f64, "f64"}, {I32 => cast_i32, "i32"}, {F32 => cast_f32, "f32"}, {I64 => cast_i64, "i64"},
            {Bool => cast_bool, "bool"}, {Usize => cast_usize, "usize"}, {Str => cast_str, "str"}, {String => cast_string, "string"},
            {Object => cast_object, "object"}, {DateTime => cast_datetime(None), "DateTime"}, {TimeDelta => cast_timedelta, "TimeDelta"}
        );
        Ok(rtn.add_obj_vec(obj_vec))
    }

    #[allow(unreachable_patterns)]
    pub fn stack(&self, other: Vec<PyExpr>, axis: i32) -> PyResult<PyExpr> {
        let obj_vec = other.iter().map(|e| e.obj()).collect_trusted();
        macro_rules! stack_macro {
            ($({$arm: ident => $cast_func: ident $(($arg: expr))?, $name: expr}),*) => {
                match &self.inner {
                    $(Exprs::$arm(expr) => {
                        let other = other.into_iter().map(|e| e.$cast_func($($arg)?).expect(&format!("can not cast to {:?}", $name))).collect_trusted();
                        expr.clone().stack(other, axis).to_py(self.obj())
                    }),*
                    _ => unimplemented!("stack for this dtype is not implemented")
                }
            };
        }
        let rtn = stack_macro!(
            {F64 => cast_f64, "f64"}, {I32 => cast_i32, "i32"}, {F32 => cast_f32, "f32"}, {I64 => cast_i64, "i64"},
            {Bool => cast_bool, "bool"}, {Usize => cast_usize, "usize"}, {Str => cast_str, "str"}, {String => cast_string, "string"},
            {Object => cast_object, "object"}, {DateTime => cast_datetime(None), "DateTime"}, {TimeDelta => cast_timedelta, "TimeDelta"}
        );
        Ok(rtn.add_obj_vec(obj_vec))
    }

    /// # Safety
    ///
    /// Data of the base expression must exist.
    #[allow(unreachable_patterns)]
    pub unsafe fn take_by_slice(
        &self,
        axis: Option<i32>,
        start: usize,
        end: usize,
        step: Option<usize>,
    ) -> Self {
        match_exprs!(&self.inner, e, {
            let axis = axis.unwrap_or(0);
            let step = step.unwrap_or(1);
            let name = e.name();
            let e_view = e.view_arr();
            let axis = e_view.norm_axis(axis);
            let e = Expr::new(
                ArbArray::View(
                    e_view
                        .slice_axis(axis, Slice::from(start..=end).step_by(step as isize))
                        .wrap(),
                ),
                name,
            );
            let e: Exprs<'_> = e.into();
            let e: Exprs<'static> = std::mem::transmute(e);
            e.into()
        })
    }
}

impl<T: ExprElement + 'static> Expr<'static, T> {
    pub fn sort_by_expr_idx(self, mut by: Vec<PyExpr>, rev: bool) -> Expr<'static, usize> {
        self.chain_view_f(
            move |arr| {
                assert_eq!(arr.ndim(), 1, "Currently only 1 dim Expr can be sorted");
                let arr = arr.to_dim1().unwrap();
                let len = arr.len();
                // evaluate the key expressions first
                by.par_iter_mut().for_each(|e| e.eval_inplace());
                let mut idx = Vec::from_iter(0..len);
                idx.sort_by(move |a, b| {
                    let mut order = Ordering::Equal;
                    for e in by.iter() {
                        let rtn = match &e.inner {
                            Exprs::F64(e) => {
                                let key_view = e.view();
                                let key_arr = key_view
                                    .into_arr()
                                    .to_dim1()
                                    .expect("Currently only 1 dim Expr can be sort key");
                                let (va, vb) = unsafe { (*key_arr.uget(*a), *key_arr.uget(*b)) };
                                if !rev {
                                    va.nan_sort_cmp_stable(&vb)
                                } else {
                                    va.nan_sort_cmp_rev_stable(&vb)
                                }
                            }
                            Exprs::I32(e) => {
                                let key_view = e.view();
                                let key_arr = key_view
                                    .into_arr()
                                    .to_dim1()
                                    .expect("Currently only 1 dim Expr can be sort key");
                                let (va, vb) = unsafe { (*key_arr.uget(*a), *key_arr.uget(*b)) };
                                if !rev {
                                    va.nan_sort_cmp_stable(&vb)
                                } else {
                                    va.nan_sort_cmp_rev_stable(&vb)
                                }
                            }
                            Exprs::String(e) => {
                                let key_view = e.view();
                                let key_arr = key_view
                                    .into_arr()
                                    .to_dim1()
                                    .expect("Currently only 1 dim Expr can be sort key");
                                let (va, vb) = unsafe { (key_arr.uget(*a), key_arr.uget(*b)) };
                                if !rev {
                                    va.cmp(vb)
                                } else {
                                    va.cmp(vb).reverse()
                                }
                            }
                            Exprs::DateTime(e) => {
                                let key_view = e.view();
                                let key_arr = key_view
                                    .into_arr()
                                    .to_dim1()
                                    .expect("Currently only 1 dim Expr can be sort key");
                                let (va, vb) = unsafe { (key_arr.uget(*a), key_arr.uget(*b)) };
                                if !rev {
                                    va.0.cmp(&vb.0)
                                } else {
                                    va.0.cmp(&vb.0).reverse()
                                }
                            }
                            _ => unimplemented!("sort by Expr of this dtype is not implemented"),
                        };
                        if rtn != Ordering::Equal {
                            order = rtn;
                            break;
                        }
                    }
                    order
                });
                Arr1::from_vec(idx).to_dimd().unwrap().into()
            },
            RefType::False,
        )
    }

    pub fn sort_by_expr(self, by: Vec<PyExpr>, rev: bool) -> Self
    where
        T: Clone,
    {
        let idx = self.clone().sort_by_expr_idx(by, rev);
        // safety: the idx is valid
        unsafe { self.take_on_axis_by_expr_unchecked(idx, 0, false) }
    }
}
