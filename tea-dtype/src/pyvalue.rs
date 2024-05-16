use std::fmt::Debug;

// use super::cast::Cast;
use super::Cast;
use crate::{DataType, GetDataType, IsNone};
use numpy::{Element, PyArrayDescr};
use pyo3::{Bound, FromPyObject, PyAny, PyObject, PyResult, Python, ToPyObject};
#[cfg(feature = "serde")]
use serde::{Serialize, Serializer};
// use std::string::ToString;

#[derive(Clone)]
#[repr(transparent)]
pub struct PyValue(pub PyObject);

// impl ToString for PyValue {
//     #[inline(always)]
//     fn to_string(&self) -> String {
//         self.0.to_string()
//     }
// }

impl std::fmt::Display for PyValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "{}", self.strftime(None))
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl Debug for PyValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.0)
    }
}

// impl GetNone for PyValue {
//     #[inline(always)]
//     fn none() -> Self {
//         PyValue(Python::with_gil(|py| py.None()))
//     }

//     #[inline(always)]
//     fn is_none(&self) -> bool {
//         Python::with_gil(|py| self.0.as_ref(py).is_none())
//     }
// }

impl IsNone for PyValue {
    type Inner = PyValue;
    type Cast<U: IsNone<Inner = U> + Clone> = U;

    #[inline(always)]
    fn none() -> Self {
        PyValue(Python::with_gil(|py| py.None()))
    }

    #[inline(always)]
    fn is_none(&self) -> bool {
        Python::with_gil(|py| self.0.as_ref(py).is_none())
    }
    
    #[inline]
    fn to_opt(self) -> Option<Self::Inner> {
        if self.is_none() {
            None
        } else {
            Some(self)
        }
    }

    #[inline(always)]
    fn from_inner(inner: Self::Inner) -> Self {
        inner
    }

    #[inline]
    fn inner_cast<U: IsNone<Inner = U> + Clone>(inner: U) -> Self::Cast<U>
    where
        Self::Inner: Cast<U::Inner>,
    {
        Cast::<U>::cast(inner)
    }
}

impl PartialEq for PyValue {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| self.0.as_ref(py).eq(other.0.as_ref(py))).unwrap()
    }
}

#[cfg(feature = "serde")]
impl Serialize for PyValue {
    fn serialize<S: Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        unimplemented!("can not serialize PyObject")
    }
}

impl ToPyObject for PyValue {
    #[inline(always)]
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.0.to_object(py)
    }
}

impl Default for PyValue {
    #[inline(always)]
    fn default() -> Self {
        PyValue(Python::with_gil(|py| py.None()))
    }
}

impl GetDataType for PyValue {
    // type Physical = PyObject;
    #[inline(always)]
    fn dtype() -> DataType {
        DataType::Object
    }
}


unsafe impl Element for PyValue {
    const IS_COPY: bool = false;
    #[inline(always)]
    fn get_dtype_bound(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        PyArrayDescr::object_bound(py)
    }
}

impl<'source> FromPyObject<'source> for PyValue {
    #[inline(always)]
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Ok(PyValue(ob.to_object(ob.py())))
    }
}

// impl Cast<String> for PyValue {
//     #[inline]
//     fn cast(self) -> String {
//         self.to_string()
//     }
// }

macro_rules! impl_pyvalue_cast {
    ($($T: ty),*) => {
        $(impl Cast<$T> for PyValue
        {
            #[inline]
            fn cast(self) -> $T {
                Python::with_gil(|py| self.0.extract::<$T>(py))
                    .expect(format!("Failed to cast pyvalue to {}", stringify!($T)).as_str())
            }
        })*
    };
}

impl_pyvalue_cast!(bool, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64, String);