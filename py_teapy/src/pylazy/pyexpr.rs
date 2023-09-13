use std::fmt::Debug;
// use std::ops::Deref;

// use ndarray::{Slice, SliceInfoElem};

use crate::from_py::PyContext;

use super::export::*;
use tears::{ExprElement, StrError, TimeUnit};
// #[cfg(feature = "option_dtype")]
// use tears::{OptF32, OptF64, OptI32, OptI64};

pub type RefObj = Option<Vec<PyObject>>;

#[pyclass(subclass)]
#[derive(Clone, Default)]
pub struct PyExpr {
    pub e: Expr<'static>,
    pub obj: RefObj,
}

impl From<Expr<'static>> for PyExpr {
    fn from(e: Expr<'static>) -> Self {
        PyExpr { e, obj: None }
    }
}

impl Debug for PyExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // match_exprs!(&self.inner, expr, { write!(f, "{:#?}", expr) })
        write!(f, "{:#?}", self.e)
    }
}

pub trait IntoPyExpr {
    fn into_pyexpr(self) -> PyExpr;
}

impl<T: ExprElement + 'static> IntoPyExpr for T {
    fn into_pyexpr(self) -> PyExpr {
        let e: Expr = self.into();
        e.into()
    }
}

pub trait ExprToPy {
    fn to_py(self, obj: RefObj) -> PyExpr;
}

impl ExprToPy for Expr<'static> {
    fn to_py(self, obj: RefObj) -> PyExpr {
        PyExpr { e: self, obj }
    }
}

impl PyExpr {
    #[inline]
    pub fn obj(&self) -> RefObj {
        self.obj.clone()
    }

    #[inline]
    pub fn inner(&self) -> &Expr<'static> {
        &self.e
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

    pub fn add_obj(&mut self, mut another: RefObj) -> &mut Self {
        if let Some(obj) = &mut self.obj {
            if let Some(another) = &mut another {
                obj.append(another);
            }
        } else if let Some(obj) = another {
            self.obj = Some(obj);
        }
        self
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

    // #[allow(unreachable_patterns, dead_code)]
    // pub fn eval(mut self, context: Option<&PyAny>, freeze: bool) -> PyResult<Self> {
    //     self.eval_inplace(context, freeze)?;
    //     Ok(self)
    // }

    #[allow(unreachable_patterns)]
    pub fn eval_inplace(&mut self, context: Option<&PyAny>, freeze: bool) -> PyResult<()> {
        let ct: PyContext<'static> = if let Some(context) = context {
            unsafe { std::mem::transmute(context.extract::<PyContext>()?) }
        } else {
            Default::default()
        };
        let (ct_rs, obj_map) = (ct.ct, ct.obj_map);
        for obj in obj_map.into_values() {
            self.add_obj(obj);
        }
        if freeze {
            self.e.eval_inplace_freeze(ct_rs).map_err(StrError::to_py)?;
        } else {
            self.e.eval_inplace(ct_rs).map_err(StrError::to_py)?;
        }
        if self.e.is_owned() {
            self.obj = None
        }
        Ok(())
    }

    pub(crate) fn cast_by_str(&self, ty_name: &str, py: Python) -> PyResult<Self> {
        let mut expr = self.clone();
        match ty_name.to_lowercase().as_str() {
            "float" | "f64" => expr.e.cast_f64(),
            "f32" => expr.e.cast_f32(),
            "int" | "i32" => expr.e.cast_i32(),
            "i64" => expr.e.cast_i64(),
            "usize" | "uint" => expr.e.cast_usize(),
            "bool" => expr.e.cast_bool(),
            "object" => expr.e.cast_object_eager(py).map_err(StrError::to_py)?,
            "str" => {
                expr.e.cast_string()
                // if self.is_object() {
                //     Ok(self
                //         .clone()
                //         .cast_object()?
                //         .object_to_string(py)?
                //         .to_py(self.obj()))
                // } else {
                // Ok(self.clone().cast_string()?.to_py(self.obj()))
                // }
            }
            "datetime" => expr.e.cast_datetime_default(),
            "datetime(ns)" => expr.e.cast_datetime(Some(TimeUnit::Nanosecond)),
            "datetime(us)" => expr.e.cast_datetime(Some(TimeUnit::Microsecond)),
            "datetime(ms)" => expr.e.cast_datetime(Some(TimeUnit::Millisecond)),
            "datetime(s)" => expr.e.cast_datetime(Some(TimeUnit::Second)),
            "timedelta" => expr.e.cast_timedelta(),
            #[cfg(feature = "option_dtype")]
            "option<f64>" => expr.e.cast_optf64(),
            _ => Err(PyValueError::new_err(
                "cast to type: {ty_name} is not implemented",
            ))?,
        };
        Ok(expr)
    }
}
