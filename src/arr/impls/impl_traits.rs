// use crate::arr::ArbArray;

use crate::arr::TimeDelta;
use crate::from_py::PyValue;

use super::super::export::*;
use ndarray::{arr0, ArrayBase, Data, DataOwned, RawData};
use std::fmt;
use std::ops::{Deref, DerefMut};

impl<T, S, D> Default for ArrBase<S, D>
where
    S: DataOwned<Elem = T>,
    D: Dimension,
    T: Default,
{
    // NOTE: We can implement Default for non-zero dimensional array views by
    // using an empty slice, however we need a trait for nonzero Dimension.
    fn default() -> Self {
        ArrayBase::default(D::default()).wrap()
    }
}

impl<S, T, D> fmt::Debug for ArrBase<S, D>
where
    S: Data<Elem = T>,
    D: Dimension,
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<S: RawData, D> Deref for ArrBase<S, D> {
    type Target = ArrayBase<S, D>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: RawData, D> DerefMut for ArrBase<S, D> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> From<T> for ArrD<T> {
    fn from(v: T) -> Self {
        arr0(v).wrap().to_dimd().unwrap()
    }
}

impl<T> From<T> for ArbArray<'_, T> {
    fn from(v: T) -> Self {
        let arr = arr0(v).wrap().to_dimd().unwrap();
        arr.into()
    }
}

impl Default for ArrOk<'_> {
    fn default() -> Self {
        let out: ArrD<i32> = Default::default();
        out.into()
    }
}

macro_rules! impl_from {
    ($(($arm: ident, $ty: ty)),*) => {
        impl<'a, T: GetDataType> From<ArbArray<'a, T>> for ArrOk<'a> {
            #[allow(unreachable_patterns)]
            fn from(arr: ArbArray<'a, T>) -> Self {
                unsafe {
                    match T::dtype() {
                        $(DataType::$arm => ArrOk::$arm(arr.into_dtype::<$ty>()),)*
                        DataType::Str => ArrOk::Str(arr.into_dtype::<&'a str>()),
                        _ => unimplemented!("Create ArrOk from this type of ArbArray is not implemented")
                    }
                }
            }
        }

        $(
            impl From<ArrD<$ty>> for ArrOk<'_> {
                fn from(arr: ArrD<$ty>) -> Self {
                    ArrOk::$arm(arr.into())
                }
            }

            impl<'a> From<ArrViewD<'a, $ty>> for ArrOk<'a> {
                fn from(arr: ArrViewD<'a, $ty>) -> Self {
                    ArrOk::$arm(arr.into())
                }
            }

            impl<'a> From<ArrViewMutD<'a, $ty>> for ArrOk<'a> {
                fn from(arr: ArrViewMutD<'a, $ty>) -> Self {
                    ArrOk::$arm(arr.into())
                }
            }
        )*

    };
}

impl_from!(
    (Bool, bool),
    (F32, f32),
    (F64, f64),
    (I32, i32),
    (I64, i64),
    (Usize, usize),
    (Object, PyValue),
    (String, String),
    (DateTime, DateTime),
    (TimeDelta, TimeDelta), //, (Str, &str)
    (OpUsize, Option<usize>)
);

impl<'a> From<ArrD<&'a str>> for ArrOk<'a> {
    fn from(arr: ArrD<&'a str>) -> Self {
        ArrOk::Str(arr.into())
    }
}

impl<'a> From<ArrViewD<'a, &'a str>> for ArrOk<'a> {
    fn from(arr: ArrViewD<'a, &'a str>) -> Self {
        ArrOk::Str(arr.into())
    }
}

impl<'a> From<ArrViewMutD<'a, &'a str>> for ArrOk<'a> {
    fn from(arr: ArrViewMutD<'a, &'a str>) -> Self {
        ArrOk::Str(arr.into())
    }
}

use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl<A, D, S> Serialize for ArrBase<S, D>
where
    A: Serialize,
    D: Dimension + Serialize,
    S: Data<Elem = A>,
{
    fn serialize<Se>(&self, serializer: Se) -> Result<Se::Ok, Se::Error>
    where
        Se: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de, A, Di, S> Deserialize<'de> for ArrBase<S, Di>
where
    A: Deserialize<'de>,
    Di: Dimension + Deserialize<'de>,
    S: DataOwned<Elem = A>,
{
    fn deserialize<D>(deserializer: D) -> Result<ArrBase<S, Di>, D::Error>
    where
        D: Deserializer<'de>,
    {
        ArrayBase::<S, Di>::deserialize(deserializer).map(|a| a.wrap())
    }
}
