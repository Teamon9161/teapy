use std::fmt::Debug;
use std::ops::Deref;

use ndarray::{Slice, SliceInfoElem};

// use crate::from_py::PyContext;

use super::export::*;
use tears::{ArbArray, DateTime, ExprElement, OptUsize, PyValue, StrError, TimeDelta, TimeUnit};
#[cfg(feature = "option_dtype")]
use tears::{OptF32, OptF64, OptI32, OptI64};

pub type RefObj = Option<Vec<PyObject>>;

#[pyclass(subclass)]
#[derive(Clone, Default)]
pub struct PyExpr {
    pub inner: Exprs<'static>,
    pub obj: RefObj,
}

impl From<Exprs<'static>> for PyExpr {
    fn from(e: Exprs<'static>) -> Self {
        PyExpr {
            inner: e,
            obj: None,
        }
    }
}

impl Debug for PyExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match_exprs!(&self.inner, expr, { write!(f, "{:#?}", expr) })
    }
}

pub trait IntoPyExpr {
    fn into_pyexpr(self) -> PyExpr;
}

impl<T: ExprElement + 'static> IntoPyExpr for T {
    fn into_pyexpr(self) -> PyExpr {
        let e: Expr<'static, T> = self.into();
        e.into()
    }
}
// impl<T: ExprElement + 'static> From<T> for PyExpr {
//     fn from(v: T) -> Self {
//         // let e: Expr<'static, T> = Expr::new(v.into(), None);
//         let e: Expr<'static, T> = v.into();
//         e.into()
//     }
// }

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

pub trait ExprToPy {
    fn to_py(self, obj: RefObj) -> PyExpr;
    // fn to_py_ref(self, obj: PyRef<PyExpr>, py: Python) -> PyExpr;
}

impl<T: ExprElement + 'static> ExprToPy for Expr<'static, T> {
    fn to_py(self, obj: RefObj) -> PyExpr {
        let exprs: Exprs<'static> = self.into();
        PyExpr { inner: exprs, obj }
    }

    // fn to_py_ref(self, obj: PyRef<PyExpr>, py: Python) -> PyExpr {
    //     let exprs: Exprs<'static> = self.into();
    //     PyExpr {
    //         inner: exprs,
    //         obj: Some(vec![obj.into_py(py)]),
    //     }
    // }
}

impl ExprToPy for Exprs<'static> {
    fn to_py(self, obj: RefObj) -> PyExpr {
        PyExpr { inner: self, obj }
    }

    // fn to_py_ref(self, obj: PyRef<PyExpr>, py: Python) -> PyExpr {
    //     PyExpr {
    //         inner: self,
    //         obj: Some(vec![obj.into_py(py)]),
    //     }
    // }
}

impl PyExpr {
    #[inline]
    pub fn obj(&self) -> RefObj {
        self.obj.clone()
    }

    #[inline]
    pub fn inner(&self) -> &Exprs<'static> {
        &self.inner
    }

    // pub fn cast_by_context(self, context: Option<Context<'static>>) -> PyResult<Self> {
    //     if context.is_none() {
    //         return Ok(self);
    //     }
    //     let obj = self.obj();
    //     let out = match_exprs!(self.inner, expr, {
    //         expr.cast_by_context(context)
    //             .map_err(StrError::to_py)?
    //             .to_py(obj)
    //     });
    //     Ok(out)
    // }

    pub fn add_obj(&mut self, mut another: RefObj) {
        if let Some(obj) = &mut self.obj {
            if let Some(another) = &mut another {
                obj.append(another);
            }
        } else if let Some(obj) = another {
            self.obj = Some(obj);
        }
    }

    #[inline]
    pub fn add_obj_into(mut self, another: RefObj) -> Self {
        self.add_obj(another);
        self
    }

    #[inline]
    pub fn add_obj_vec(&mut self, another: Vec<RefObj>) {
        for obj in another {
            self.add_obj(obj);
        }
    }

    #[inline]
    pub fn add_obj_vec_into(mut self, another: Vec<RefObj>) -> Self {
        self.add_obj_vec(another);
        self
    }

    // Cast the output of the expression to f64 ndarray
    pub fn cast_f64(self) -> PyResult<Expr<'static, f64>> {
        self.inner.cast_f64().map_err(StrError::to_py)
    }

    #[cfg(feature = "option_dtype")]
    // Cast the output of the expression to Option<f32> ndarray
    pub fn cast_optf32(self) -> PyResult<Expr<'static, OptF32>> {
        self.inner.cast_optf32().map_err(StrError::to_py)
    }

    #[cfg(feature = "option_dtype")]
    // Cast the output of the expression to Option<f64> ndarray
    pub fn cast_optf64(self) -> PyResult<Expr<'static, OptF64>> {
        self.inner.cast_optf64().map_err(StrError::to_py)
    }

    #[cfg(feature = "option_dtype")]
    // Cast the output of the expression to Option<i32> ndarray
    pub fn cast_opti32(self) -> PyResult<Expr<'static, OptI32>> {
        self.inner.cast_opti32().map_err(StrError::to_py)
    }

    #[cfg(feature = "option_dtype")]
    // Cast the output of the expression to Option<i64> ndarray
    pub fn cast_opti64(self) -> PyResult<Expr<'static, OptI64>> {
        self.inner.cast_opti64().map_err(StrError::to_py)
    }

    // Cast the output of the expression to Option<usize> ndarray
    pub fn cast_optusize(self) -> PyResult<Expr<'static, OptUsize>> {
        self.inner.cast_optusize().map_err(StrError::to_py)
    }

    // Cast the output of the expression to f32 ndarray
    pub fn cast_f32(self) -> PyResult<Expr<'static, f32>> {
        self.inner.cast_f32().map_err(StrError::to_py)
    }

    // Cast the output of the expression to i64 ndarray
    pub fn cast_i64(self) -> PyResult<Expr<'static, i64>> {
        self.inner.cast_i64().map_err(StrError::to_py)
    }

    // Cast the output of the expression to i32 ndarray
    pub fn cast_i32(self) -> PyResult<Expr<'static, i32>> {
        self.inner.cast_i32().map_err(StrError::to_py)
    }

    // Cast the output of the expression to usize ndarray
    pub fn cast_usize(self) -> PyResult<Expr<'static, usize>> {
        self.inner.cast_usize().map_err(StrError::to_py)
    }

    // Cast the output of the expression to usize ndarray
    pub fn cast_vecusize(self) -> PyResult<Expr<'static, Vec<usize>>> {
        self.inner.cast_vecusize().map_err(StrError::to_py)
    }

    // Cast the output of the expression to object ndarray lazily
    pub fn cast_object(self) -> PyResult<Expr<'static, PyValue>> {
        self.inner.cast_object().map_err(StrError::to_py)
    }

    // Cast the output of the expression to object ndarray
    pub fn cast_object_eager(self, py: Python) -> PyResult<Expr<'static, PyValue>> {
        self.inner.cast_object_eager(py).map_err(StrError::to_py)
    }

    // Cast the output of the expression to datetime ndarray
    pub fn cast_datetime(self, unit: Option<TimeUnit>) -> PyResult<Expr<'static, DateTime>> {
        self.inner.cast_datetime(unit).map_err(StrError::to_py)
    }

    // Cast the output of the expression to timedelta ndarray
    pub fn cast_timedelta(self) -> PyResult<Expr<'static, TimeDelta>> {
        self.inner.cast_timedelta().map_err(StrError::to_py)
    }

    // Cast the output of the expression to datetime ndarray
    pub fn cast_datetime_default(self) -> PyResult<Expr<'static, DateTime>> {
        self.cast_datetime(Default::default())
    }

    // Cast the output of the expression to str ndarray
    pub fn cast_str(self) -> PyResult<Expr<'static, &'static str>> {
        self.inner.cast_str().map_err(StrError::to_py)
    }

    // Cast the output of the expression to string ndarray
    pub fn cast_string(self) -> PyResult<Expr<'static, String>> {
        self.inner.cast_string().map_err(StrError::to_py)
    }

    // Cast the output of the expression to bool ndarray
    pub fn cast_bool(self) -> PyResult<Expr<'static, bool>> {
        self.inner.cast_bool().map_err(StrError::to_py)
    }

    #[allow(unreachable_patterns, dead_code)]
    pub fn eval(mut self, context: Option<&PyAny>) -> PyResult<Self> {
        self.eval_inplace(context)?;
        Ok(self)
    }

    #[allow(unreachable_patterns)]
    pub fn eval_inplace(&mut self, _context: Option<&PyAny>) -> PyResult<()> {
        // let ct: PyContext<'static> = if let Some(context) = context {
        //     unsafe { std::mem::transmute(context.extract::<PyContext>()?) }
        // } else {
        //     Default::default()
        // };
        // let (ct_rs, obj_map) = (ct.ct, ct.obj_map);
        // if ct_rs.is_some() {
        //     self.inner
        //     .cast_by_context(ct_rs.clone())
        //     .map_err(StrError::to_py)?;
        //     for obj in obj_map.into_values() {
        //         self.add_obj(obj)
        //     }
        // }
        match_exprs!(&mut self.inner, expr, {
            expr.eval_inplace(None).map_err(StrError::to_py)?;
            if let Some(owned) = expr.is_owned() {
                if owned {
                    self.obj = None
                } // we don't need to reference
            }
        });
        Ok(())
    }

    #[allow(unreachable_patterns)]
    pub fn _select_by_expr(&self, slc: Self, axis: Self) -> PyResult<Self> {
        let obj = slc.obj();
        match_exprs!(&self.inner, expr, {
            Ok(expr
                .clone()
                .select_by_expr(slc.cast_usize()?, axis.cast_i32()?)
                .to_py(self.obj())
                .add_obj_into(obj))
        })
    }

    #[allow(unreachable_patterns)]
    /// # Safety
    ///
    /// The data for the array view should exist
    pub unsafe fn _select_by_expr_unchecked(&self, slc: Self, axis: Self) -> PyResult<Self> {
        let obj = slc.obj();
        if matches!(&slc.inner, &Exprs::OptUsize(_)) {
            if matches!(
                &self.inner,
                &Exprs::I32(_) | &Exprs::I64(_) | &Exprs::Usize(_)
            ) {
                match_exprs!(
                    &self.inner,
                    expr,
                    {
                        Ok(expr
                            .clone()
                            .cast::<f64>()
                            .take_option_on_axis_by_expr_unchecked(
                                slc.cast_optusize()?,
                                axis.cast_i32()?,
                                false,
                            )
                            .to_py(self.obj())
                            .add_obj_into(obj))
                    },
                    I32,
                    I64,
                    Usize
                )
            } else {
                match_exprs!(
                    &self.inner,
                    expr,
                    {
                        Ok(expr
                            .clone()
                            .take_option_on_axis_by_expr_unchecked(
                                slc.cast_optusize()?,
                                axis.cast_i32()?,
                                false,
                            )
                            .to_py(self.obj())
                            .add_obj_into(obj))
                    },
                    F32,
                    F64,
                    Str,
                    String,
                    DateTime,
                    Object,
                    TimeDelta,
                    #[cfg(feature = "option_dtype")]
                    OptF32,
                    #[cfg(feature = "option_dtype")]
                    OptF64,
                    #[cfg(feature = "option_dtype")]
                    OptI32,
                    #[cfg(feature = "option_dtype")]
                    OptI64
                )
            }
        } else {
            match_exprs!(&self.inner, expr, {
                Ok(expr
                    .clone()
                    .select_by_expr_unchecked(slc.cast_usize()?, axis.cast_i32()?)
                    .to_py(self.obj())
                    .add_obj_into(obj))
            })
        }
    }

    #[allow(unreachable_patterns)]
    pub fn _select_by_i32_expr(&self, slc: Self, axis: Self) -> PyResult<Self> {
        let obj = slc.obj();
        match_exprs!(&self.inner, expr, {
            Ok(expr
                .clone()
                .select_by_i32_expr(slc.cast_i32()?, axis.cast_i32()?)
                .to_py(self.obj())
                .add_obj_into(obj))
        })
    }

    // pub fn _select(&self, slc: Vec<usize>, axis: i32) -> Self {
    //     let slc_expr: Expr<usize> = slc.into();
    //     self._select_by_expr(slc_expr.into(), axis.into_pyexpr())
    //         .unwrap()
    // }

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
    pub fn sort_by_expr(&self, by: Vec<Self>, rev: bool) -> Self {
        let obj_vec: Vec<RefObj> = by.iter().map(|e| e.obj()).collect();
        let by = by.into_iter().map(|e| e.inner).collect_trusted();
        let out = match_exprs!(&self.inner, expr, {
            expr.clone().sort_by_expr(by, rev).to_py(self.obj())
        });
        out.add_obj_vec_into(obj_vec)
    }

    #[allow(unreachable_patterns)]
    pub fn get_sort_idx(&self, by: Vec<Self>, rev: bool) -> Self {
        let obj_vec: Vec<RefObj> = by.iter().map(|e| e.obj()).collect();
        let by = by.into_iter().map(|e| e.inner).collect_trusted();
        let out = match_exprs!(&self.inner, expr, {
            expr.clone().get_sort_idx(by, rev).to_py(self.obj())
        });
        out.add_obj_vec_into(obj_vec)
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
                        .add_obj_into(obj)
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
                        .add_obj_into(obj)
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
                        .object_to_string(py)?
                        .to_py(self.obj()))
                } else {
                    Ok(self.clone().cast_string()?.to_py(self.obj()))
                }
            }
            "datetime" => Ok(self.clone().cast_datetime_default()?.to_py(self.obj())),
            "datetime(ns)" => Ok(self
                .clone()
                .cast_datetime(Some(TimeUnit::Nanosecond))?
                .to_py(self.obj())),
            "datetime(us)" => Ok(self
                .clone()
                .cast_datetime(Some(TimeUnit::Microsecond))?
                .to_py(self.obj())),
            "datetime(ms)" => Ok(self
                .clone()
                .cast_datetime(Some(TimeUnit::Millisecond))?
                .to_py(self.obj())),
            "datetime(s)" => Ok(self
                .clone()
                .cast_datetime(Some(TimeUnit::Second))?
                .to_py(self.obj())),
            "timedelta" => Ok(self.clone().cast_timedelta()?.to_py(self.obj())),
            #[cfg(feature = "option_dtype")]
            "option<f64>" => Ok(self.clone().cast_optf64()?.to_py(self.obj())),
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
        Ok(rtn.add_obj_vec_into(obj_vec))
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
        Ok(rtn.add_obj_vec_into(obj_vec))
    }

    /// # Safety
    ///
    /// Data of the base expression must exist.
    #[allow(unreachable_patterns)]
    pub unsafe fn select_by_slice_eager(
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
