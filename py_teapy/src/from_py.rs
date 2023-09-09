use std::sync::Arc;

use crate::pylazy::RefObj;
use numpy::{
    datetime::{units, Datetime},
    PyArray, PyArrayDyn,
};
use pyo3::{exceptions::PyValueError, FromPyObject, PyAny, PyObject, PyResult, Python, ToPyObject};
use tears::{Context, PyValue};
#[cfg(feature = "option_dtype")]
use tears::{OptF64, OptI64};
// #[cfg(feature="datadict")]
use crate::pylazy::PyDataDict;
use ahash::HashMap;
// use super::export::*;

#[derive(FromPyObject)]
pub enum Scalar {
    Bool(bool),
    F64(f64),
    I32(i32),
    Usize(usize),
    String(String),
    Object(PyValue),
}

macro_rules! impl_scalar {
    ($($arm: ident),*) => {
        impl ToPyObject for Scalar {
            fn to_object(&self, py: Python<'_>) -> PyObject {
                match self {
                    $(Scalar::$arm(e) => e.to_object(py)),*
                }
            }
        }
    };
}

impl_scalar!(Bool, F64, I32, Usize, String, Object);

// 所有可接受的python array类型
#[derive(FromPyObject)]
pub enum PyArrayOk<'py> {
    Bool(&'py PyArrayDyn<bool>),
    F32(&'py PyArrayDyn<f32>),
    F64(&'py PyArrayDyn<f64>),
    I32(&'py PyArrayDyn<i32>),
    I64(&'py PyArrayDyn<i64>),
    Usize(&'py PyArrayDyn<usize>),
    Object(&'py PyArrayDyn<PyValue>),
    DateTimeMs(&'py PyArrayDyn<Datetime<units::Milliseconds>>),
    DateTimeUs(&'py PyArrayDyn<Datetime<units::Microseconds>>),
    DateTimeNs(&'py PyArrayDyn<Datetime<units::Nanoseconds>>),
}

/// match the enum `PyArrayOk` to get the discrete dtype of `PyArray` so that we can
/// call functions on a `PyArray` of which dtype is known;
macro_rules! match_pyarray {

    ($pyarr: expr, $e: ident, $body: tt $(,$arm: ident)*) => {
        match $pyarr {
            $(PyArrayOk::$arm($e) => $body,)*
            _ => unimplemented!("match pyarray of this dtype is not implemented")
        }
    };

    ($pyarr: expr, $e: ident, $body: tt) => {
        match_pyarray!($pyarr, $e, $body, Bool, F32, F64, I32, I64, Usize, Object, DatetimeMs, DatetimeNs, DatetimeUs)
    };

    (numeric $pyarr: expr, $e: ident, $body: tt) => {
        match_pyarray!($pyarr, $e, $body, F32, F64, I32, I64, Usize)
    };

    (datetime $pyarr: expr, $e: ident, $body: tt) => {
        match_pyarray!($pyarr, $e, $body, DatetimeMs, DatetimeNs, DatetimeUs)
    };
}

impl<'py> PyArrayOk<'py> {
    pub fn is_object(&self) -> bool {
        matches!(self, PyArrayOk::Object(_))
    }

    pub fn is_datetime(&self) -> bool {
        use PyArrayOk::*;
        matches!(self, DateTimeMs(_) | DateTimeNs(_) | DateTimeUs(_))
    }

    pub fn into_object(self) -> PyResult<&'py PyArrayDyn<PyValue>> {
        if let PyArrayOk::Object(obj_arr) = self {
            Ok(obj_arr)
        } else {
            Err(PyValueError::new_err("Dtype of the array is not object"))
        }
    }
}

// do not change the order of the variants
#[derive(FromPyObject)]
pub enum PyList {
    Bool(Vec<bool>),
    // #[cfg(feature = "option_dtype")]
    // Boll(Vec<OptBool>)
    I64(Vec<i64>),
    #[cfg(feature = "option_dtype")]
    OptI64(Vec<OptI64>),
    F64(Vec<f64>),
    #[cfg(feature = "option_dtype")]
    OptF64(Vec<OptF64>),
    String(Vec<String>),
    Object(Vec<PyValue>),
}

macro_rules! match_pylist {
    ($list: expr, $l: ident, $body: tt) => {
        match $list {
            PyList::Bool($l) => $body,
            PyList::I64($l) => $body,
            #[cfg(feature = "option_dtype")]
            PyList::OptI64($l) => $body,
            PyList::F64($l) => $body,
            #[cfg(feature = "option_dtype")]
            PyList::OptF64($l) => $body,
            PyList::String($l) => $body,
            PyList::Object($l) => $body,
        }
    };
}

pub trait NoDim0 {
    fn no_dim0(self, py: Python) -> PyResult<PyObject>;
}

impl<T, D> NoDim0 for &PyArray<T, D> {
    fn no_dim0(self, py: Python) -> PyResult<PyObject> {
        if self.ndim() == 0 {
            Ok(self.call_method0("item")?.to_object(py))
        } else {
            Ok(self.to_object(py))
        }
    }
}

#[derive(Clone, Default)]
pub struct PyContext<'py> {
    pub ct: Option<Context<'py>>,
    pub obj_map: HashMap<String, RefObj>,
}

impl<'py> From<Context<'py>> for PyContext<'py> {
    fn from(ct: Context<'py>) -> Self {
        Self {
            ct: Some(ct),
            obj_map: Default::default(),
        }
    }
}

impl<'py> FromPyObject<'py> for PyContext<'py> {
    fn extract(ob: &'py PyAny) -> PyResult<PyContext<'py>> {
        if ob.is_none() {
            Ok(Self {
                ct: None,
                obj_map: Default::default(),
            })
        } else if let Ok(dd) = ob.extract::<PyDataDict>() {
            Ok(Self {
                // safety: we can cast 'py to 'static as we are running in python
                ct: unsafe { Some(std::mem::transmute(Arc::new(dd.dd))) },
                obj_map: dd.obj_map,
            })
        } else if ob.hasattr("_dd")? && ob.get_type().name()? == "DataDict" {
            let dd = ob.getattr("_dd")?.extract::<PyDataDict>()?;
            Ok(Self {
                ct: unsafe { Some(std::mem::transmute(Arc::new(dd.dd))) },
                obj_map: dd.obj_map,
            })
        } else {
            Err(PyValueError::new_err(
                "Cannot extract a Context from the object",
            ))
        }
    }
}
