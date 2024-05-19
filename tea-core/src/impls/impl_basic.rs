// use crate::impl_reduce_nd;
use crate::prelude::{Arr1, ArrBase, ArrD, WrapNdarray};
use datatype::{BoolType, IsNone, Number};
use ndarray::{Data, Dimension, Ix1, Zip};
use num::Zero;
use tea_macros::arr_agg_ext;
use tevec::prelude::*;


impl<T, S: Data<Elem = T>> ArrBase<S, Ix1> {
    /// sum of the array on a given axis, return valid_num n and the sum of the array
    pub fn nsum_1d(&self, stable: bool) -> (usize, T)
    where
        T: Number,
        T::Inner: Number,
    {
        if !stable {
            if let Some(slc) = self.as_slice_memory_order() {
                let (n, sum) = utils::vec_nfold(slc, T::zero, T::n_add);
                return (n, sum);
            }
        }
        // fall back to normal calculation
        let (n, acc) = if !stable {
            self.n_fold_valid(T::Inner::zero(), |acc, v| acc + v)
        } else {
            let mut c = T::Inner::zero();
            self.n_fold_valid(T::Inner::zero(), |acc, v| acc.kh_sum(v, &mut c))
        };
        if n >= 1 {
            (n, T::from_inner(acc))
        } else {
            (0, T::none())
        }
    }
}

#[arr_agg_ext]
impl<T: Send + Sync, S: Data<Elem = T>, D: Dimension> BasicAggExt for ArrBase<S, D> {
    /// Max value of the array on a given axis
    pub fn max(&self) -> T
    where T: Number
    {
        if let Some(slc) = self.0.as_slice_memory_order() {
            let res = utils::vec_fold(slc, T::min_, T::max_with);
            return if res == T::min_() {
                T::none()
            } else {
                res
            };
        }
        let mut max = T::min_();
        self.as_dim1().apply(|v| {
            if max < *v {
                max = *v;
            }
        });
        // note: assume that not all of the elements are the max value of type T
        if max == T::min_() {
            T::none()
        } else {
            max
        }
    }

    /// Min value of the array on a given axis
    pub fn min(&self) -> T
    where T: Number
    {
        if let Some(slc) = self.as_slice_memory_order() {
            let res = utils::vec_fold(slc, T::max_, T::min_with);
            return if res == T::max_() {
                T::none()
            } else {
                res
            };
        }
        let mut min = T::max_();
        self.as_dim1().apply(|v| {
            if min > *v {
                min = *v;
            }
        });
        // note: assume that not all of the elements are the max value of type T
        if min == T::max_() {
            T::none()
        } else {
            min
        }
    }

    /// sum of the array on a given axis
    #[inline]
    pub fn sum(&self, stable: bool) -> T
    where T: Number, T::Inner: Number,
    {
        self.as_dim1().nsum_1d(stable).1
    }

    /// mean of the array on a given axis
    #[inline]
    pub fn mean(&self, min_periods: usize, stable: bool) -> f64
    where T: Number, T::Inner: Number,
    {
        let(n, sum) = self.as_dim1().nsum_1d(stable);
        if n >= min_periods {
            sum.f64() / n.f64()
        } else {
            f64::NAN
        }
    }

    /// count a value of an array on a given axis
    #[inline]
    pub fn count_v(&self, value: T) -> i32
    where 
        T: IsNone + Clone, 
        T::Inner: PartialEq
    {
        // self.count_by(|v| v == &value)
        self.0.iter().cloned().vcount_value(value) as i32
    }

    /// count a value of an array on a given axis
    #[inline]
    pub fn count_nan(&self) -> i32
    where T: IsNone + Clone
    {
        // self.count_by(|v| v == &value)
        self.0.iter().cloned().count_none() as i32
    }
    /// count a value of an array on a given axis
    #[inline]
    pub fn count_notnan(&self) -> i32
    where T: IsNone + Clone
    {
        // self.count_by(|v| v == &value)
        Vec1ViewAggValid::count(self.0.iter().cloned()) as i32
    }

    #[inline]
    pub fn any(&self) -> bool
    where T: BoolType
    {
        self.0.iter().cloned().any()
    }
    
    #[inline]
    pub fn all(&self) -> bool
    where T: BoolType + Copy
    {
        self.0.iter().cloned().all()
    }
}


// impl_reduce_nd!(
//     max,
//     /// Max value of the array on a given axis
//     pub fn max_1d(&self) -> T
//     where T: Number
//     // {T: Number,}
//     {
//         if let Some(slc) = self.0.as_slice_memory_order() {
//             let res = utils::vec_fold(slc, T::min_, T::max_with);
//             return if res == T::min_() {
//                 T::none()
//             } else {
//                 res
//             };
//         }
//         let mut max = T::min_();
//         self.apply(|v| {
//             if max < *v {
//                 max = *v;
//             }
//         });
//         // note: assume that not all of the elements are the max value of type T
//         if max == T::min_() {
//             T::none()
//         } else {
//             max
//         }
//     }
// );

// impl_reduce_nd!(
//     min,
//     /// Min value of the array on a given axis
//     pub fn min_1d(&self) -> T
//     {T: Number,}
//     {
//         if let Some(slc) = self.as_slice_memory_order() {
//             let res = utils::vec_fold(slc, T::max_, T::min_with);
//             return if res == T::max_() {
//                 T::none()
//             } else {
//                 res
//             };
//         }
//         let mut min = T::max_();
//         self.apply(|v| {
//             if min > *v {
//                 min = *v;
//             }
//         });
//         // note: assume that not all of the elements are the max value of type T
//         if min == T::max_() {
//             T::none()
//         } else {
//             min
//         }
//     }
// );

// impl_reduce_nd!(
//     sum,
//     /// sum of the array on a given axis
//     #[inline]
//     pub fn sum_1d(&self, stable: bool) -> T
//     {T: Number,}
//     {
//         self.nsum_1d(stable).1
//     }
// );

// impl_reduce_nd!(
//     mean,
//     /// mean of the array on a given axis
//     #[inline]
//     pub fn mean_1d(&self, min_periods: usize, stable: bool) -> f64
//     {T: Number,}
//     {
//         let(n, sum) = self.nsum_1d(stable);
//         if n >= min_periods {
//             sum.f64() / n.f64()
//         } else {
//             f64::NAN
//         }
//     }
// );

// impl_reduce_nd!(
//     count_v,
//     /// count a value of an array on a given axis
//     #[inline]
//     pub fn count_v_1d(&self, value: T) -> i32
//     {T: PartialEq; Send; Sync,}
//     {
//         self.count_by(|v| v == &value)
//     }
// );

// impl_reduce_nd!(
//     count_notnan,
//     /// count not NaN number of an array on a given axis
//     #[inline]
//     pub fn count_notnan_1d(&self) -> i32
//     {T: IsNone; Send; Sync,}
//     {
//         self.count_by(|v| !v.is_none())
//     }
// );

// impl_reduce_nd!(
//     count_nan,
//     /// count NaN number of an array on a given axis
//     #[inline]
//     pub fn count_nan_1d(&self) -> i32
//     {T: IsNone; Send; Sync,}
//     {
//         self.count_by(|v| v.is_none())
//     }
// );

// impl_reduce_nd!(
//     any,
//     /// sum of the array on a given axis
//     #[inline]
//     pub fn any_1d(&self) -> bool
//     {T: BoolType; Send ; Sync; Copy,}
//     {
//         for v in self {
//             if v.bool_() {
//                 return true;
//             }
//         }
//         false
//     }
// );

// impl_reduce_nd!(
//     all,
//     /// sum of the array on a given axis
//     #[inline]
//     pub fn all_1d(&self) -> bool
//     {T: BoolType; Send ; Sync; Copy,}
//     {
//         for v in self {
//             if !v.bool_() {
//                 return false;
//             }
//         }
//         true
//     }
// );
