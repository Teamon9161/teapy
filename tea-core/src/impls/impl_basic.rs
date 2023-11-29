use crate::impl_reduce_nd;
use crate::prelude::{Arr1, ArrBase, ArrD, WrapNdarray};
use datatype::{BoolType, GetNone, Number};
use ndarray::{Data, Dimension, Ix1, RemoveAxis, Zip};

impl<T, S: Data<Elem = T>> ArrBase<S, Ix1> {
    /// sum of the array on a given axis, return valid_num n and the sum of the array
    pub fn nsum_1d(&self, stable: bool) -> (usize, T)
    where
        T: Number,
    {
        if !stable {
            if let Some(slc) = self.as_slice_memory_order() {
                let (n, sum) = utils::vec_nfold(slc, T::zero, T::n_add);
                return (n, sum);
            }
        }
        // fall back to normal calculation
        let (n, acc) = if !stable {
            self.n_acc_valid(T::zero(), |v| *v)
        } else {
            self.stable_n_acc_valid(T::zero(), |v| *v)
        };
        if n >= 1 {
            (n, acc)
        } else {
            (0, T::nan())
        }
    }
}

impl_reduce_nd!(
    max,
    /// Max value of the array on a given axis
    pub fn max_1d(&self) -> T
    {T: Number,}
    {
        if let Some(slc) = self.0.as_slice_memory_order() {
            let res = utils::vec_fold(slc, T::min_, T::max_with);
            return if res == T::min_() {
                T::nan()
            } else {
                res
            };
        }
        let mut max = T::min_();
        self.apply(|v| {
            if max < *v {
                max = *v;
            }
        });
        // note: assume that not all of the elements are the max value of type T
        if max == T::min_() {
            T::nan()
        } else {
            max
        }
    }
);

impl_reduce_nd!(
    min,
    /// Min value of the array on a given axis
    pub fn min_1d(&self) -> T
    {T: Number,}
    {
        if let Some(slc) = self.as_slice_memory_order() {
            let res = utils::vec_fold(slc, T::max_, T::min_with);
            return if res == T::max_() {
                T::nan()
            } else {
                res
            };
        }
        let mut min = T::max_();
        self.apply(|v| {
            if min > *v {
                min = *v;
            }
        });
        // note: assume that not all of the elements are the max value of type T
        if min == T::max_() {
            T::nan()
        } else {
            min
        }
    }
);

impl_reduce_nd!(
    sum,
    /// sum of the array on a given axis
    #[inline]
    pub fn sum_1d(&self, stable: bool) -> T
    {T: Number,}
    {
        self.nsum_1d(stable).1
    }
);

impl_reduce_nd!(
    mean,
    /// mean of the array on a given axis
    #[inline]
    pub fn mean_1d(&self, min_periods: usize, stable: bool) -> f64
    {T: Number,}
    {
        let(n, sum) = self.nsum_1d(stable);
        if n >= min_periods {
            sum.f64() / n.f64()
        } else {
            f64::NAN
        }
    }
);

impl_reduce_nd!(
    count_v,
    /// count a value of an array on a given axis
    #[inline]
    pub fn count_v_1d(&self, value: T) -> i32
    {T: PartialEq; Send; Sync,}
    {
        self.count_by(|v| v == &value)
    }
);

impl_reduce_nd!(
    count_notnan,
    /// count not NaN number of an array on a given axis
    #[inline]
    pub fn count_notnan_1d(&self) -> i32
    {T: GetNone; Send; Sync,}
    {
        self.count_by(|v| !v.is_none())
    }
);

impl_reduce_nd!(
    count_nan,
    /// count NaN number of an array on a given axis
    #[inline]
    pub fn count_nan_1d(&self) -> i32
    {T: GetNone; Send; Sync,}
    {
        self.count_by(|v| v.is_none())
    }
);

impl_reduce_nd!(
    any,
    /// sum of the array on a given axis
    #[inline]
    pub fn any_1d(&self) -> bool
    {T: BoolType; Send ; Sync; Copy,}
    {
        for v in self {
            if v.bool_() {
                return true;
            }
        }
        false
    }
);

impl_reduce_nd!(
    all,
    /// sum of the array on a given axis
    #[inline]
    pub fn all_1d(&self) -> bool
    {T: BoolType; Send ; Sync; Copy,}
    {
        for v in self {
            if !v.bool_() {
                return false;
            }
        }
        true
    }
);
