use crate::prelude::*;
use derive_more::{Deref, Display};
use numpy::{Element, PyArrayDescr};
use pyo3::prelude::*;
use std::fmt::Debug;

#[derive(Clone, Debug, Display, Deref)]
#[repr(transparent)]
pub struct Object(pub PyObject);

impl IsNone for Object {
    type Inner = Object;
    type Cast<U: IsNone<Inner = U> + Clone> = U;

    #[inline(always)]
    fn none() -> Self {
        Object(Python::with_gil(|py| py.None()))
    }

    #[inline(always)]
    fn is_none(&self) -> bool {
        Python::with_gil(|py| self.0.bind(py).is_none())
    }

    #[inline]
    fn to_opt(self) -> Option<Self::Inner> {
        if self.is_none() {
            None
        } else {
            Some(self)
        }
    }

    #[inline]
    fn as_opt(&self) -> Option<&Self::Inner> {
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

impl PartialEq for Object {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Python::with_gil(|py| self.0.bind(py).eq(other.0.bind(py))).unwrap()
    }
}

impl ToPyObject for Object {
    #[inline(always)]
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.0.to_object(py)
    }
}

impl Default for Object {
    #[inline(always)]
    fn default() -> Self {
        Object(Python::with_gil(|py| py.None()))
    }
}

impl GetDataType for Object {
    #[inline(always)]
    fn dtype() -> DataType {
        DataType::Object
    }
}

unsafe impl Element for Object {
    const IS_COPY: bool = false;
    #[inline(always)]
    fn get_dtype_bound(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        PyArrayDescr::object_bound(py)
    }
}

impl<'source> FromPyObject<'source> for Object {
    #[inline(always)]
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        Ok(Object(ob.to_object(ob.py())))
    }
}

impl<'a> Cast<Object> for &'a str {
    #[inline]
    fn cast(self) -> Object {
        Python::with_gil(|py| Object(self.to_object(py)))
    }
}

macro_rules! impl_object_cast {
    ($($T: ty),*) => {
        $(
            impl Cast<$T> for Object
            {
                #[inline]
                fn cast(self) -> $T {
                    Python::with_gil(|py| self.0.extract::<$T>(py))
                        .expect(format!("Failed to cast Object to {}", stringify!($T)).as_str())
                }
            }

            impl Cast<Object> for $T
            {
                #[inline]
                fn cast(self) -> Object {
                    Python::with_gil(|py| Object(self.to_object(py)))
                }
            }

            impl Cast<Object> for Option<$T>
            {
                #[inline]
                fn cast(self) -> Object {
                    if let Some(v) = self {
                        v.cast()
                    } else {
                        Object::none()
                    }
                }
            }

            impl Cast<Option<$T>> for Object
            {
                #[inline]
                fn cast(self) -> Option<$T> {
                    if self.is_none() {
                        return None;
                    } else {
                        Python::with_gil(|py| self.0.extract::<$T>(py)).ok()
                    }
                }
            }
        )*
    };
}

impl_object_cast!(
    bool, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64, String
);
