use crate::{Arr, ArrView, ArrViewMut};
use datatype::{OptBool, OptF32, OptF64, OptI32, OptI64};
use ndarray::Dimension;

pub trait ArrToOpt {
    type OutType;
    fn to_opt(&self) -> Self::OutType;
}

#[cfg(feature = "option_dtype")]
macro_rules! impl_arr_to_opt {
    ($typ: ident, $real: ty) => {
        impl<D: Dimension> ArrToOpt for Arr<$real, D> {
            type OutType = Arr<$typ, D>;
            #[inline]
            fn to_opt(&self) -> Self::OutType {
                self.map(|v| $typ(Some(*v)))
            }
        }

        impl<'a, D: Dimension> ArrToOpt for ArrView<'a, $real, D> {
            type OutType = Arr<$typ, D>;
            #[inline]
            fn to_opt(&self) -> Self::OutType {
                self.map(|v| $typ(Some(*v)))
            }
        }

        impl<'a, D: Dimension> ArrToOpt for ArrViewMut<'a, $real, D> {
            type OutType = Arr<$typ, D>;
            #[inline]
            fn to_opt(&self) -> Self::OutType {
                self.map(|v| $typ(Some(*v)))
            }
        }
    };
}

#[cfg(feature = "option_dtype")]
impl_arr_to_opt!(OptF64, f64);
#[cfg(feature = "option_dtype")]
impl_arr_to_opt!(OptF32, f32);
#[cfg(feature = "option_dtype")]
impl_arr_to_opt!(OptI64, i64);
#[cfg(feature = "option_dtype")]
impl_arr_to_opt!(OptI32, i32);
#[cfg(feature = "option_dtype")]
impl_arr_to_opt!(OptBool, bool);
