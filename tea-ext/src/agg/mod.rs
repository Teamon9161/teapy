mod corr;
#[cfg(feature = "lazy")]
mod impl_lazy;

#[cfg(feature = "lazy")]
pub use corr::AutoExprAgg2Ext;
pub use corr::{Agg2Ext, CorrToolExt1d};
#[cfg(feature = "lazy")]
pub use impl_lazy::{corr, AutoExprAggExt, DataDictCorrExt, ExprAggExt};
#[cfg(feature = "lazy")]
use lazy::Expr;
use ndarray::{Data, Dimension, Ix1, Zip};
use tea_core::prelude::*;
// use tea_core::utils::{kh_sum, vec_fold, vec_nfold};

#[ext_trait]
impl<T: IsNone + Clone + Send + Sync, S: Data<Elem = T>> AggExt1d for ArrBase<S, Ix1> {
    /// production of the array on a given axis, return valid_num n and the production of the array
    fn nprod_1d(&self) -> (usize, T::Inner)
    where
        T::Inner: Number,
    {
        if let Some(slc) = self.0.try_as_slice() {
            slc.titer().vfold_n(T::Inner::one(), |acc, x| acc * x)
        } else {
            // fall back to normal calculation
            self.0.titer().vfold_n(T::Inner::one(), |acc, x| acc * x)
        }
    }

    /// mean and variance of the array on a given axis
    #[inline]
    pub fn meanvar_1d(&self, min_periods: usize) -> (f64, f64)
    where
        T::Inner: Number,
    {
        self.as_dim1().0.titer().vmean_var(min_periods)
    }

    #[cfg(feature = "map")]
    #[inline]
    fn umax_1d(&self) -> T
    where
        T: PartialEq,
        T::Inner: Number,
    {
        use crate::map::MapExt1d;
        self.sorted_unique_1d().max_1d()
    }

    #[cfg(feature = "map")]
    #[inline]
    fn umin_1d(&self) -> T
    where
        T: PartialEq,
        T::Inner: Number,
    {
        use crate::map::MapExt1d;
        self.sorted_unique_1d().min_1d()
    }

    pub fn cut_nsum_1d<S2>(&self, mask: &ArrBase<S2, Ix1>, min_periods: usize) -> (usize, T)
    where
        T: Number,
        S2: Data<Elem = bool>,
    {
        let mut n = 0_usize;
        let sum = self.fold_with(mask.view(), T::zero(), |acc, v, valid| {
            if (*valid) && v.not_none() {
                n += 1;
                acc + *v
            } else {
                acc
            }
        });
        if n >= min_periods {
            (n, sum)
        } else {
            (n, T::none())
        }
    }

    #[inline]
    pub fn cut_sum_1d<S2>(
        &self,
        mask: &ArrBase<S2, Ix1>,
        min_periods: usize,
        // stable: bool,
    ) -> T
    where
        T: Number,
        S2: Data<Elem = bool>,
    {
        self.cut_nsum_1d(mask, min_periods).1
    }

    #[inline]
    pub fn cut_mean_1d<S2>(&self, mask: &ArrBase<S2, Ix1>, min_periods: usize) -> f64
    where
        T: Number,
        S2: Data<Elem = bool>,
    {
        let (n, sum) = self.cut_nsum_1d(mask, min_periods);
        sum.f64() / n.f64()
    }
}

#[arr_agg_ext(lazy = "view", type = "Numeric")]
impl<S, D, T> AggExtNd<D, T> for ArrBase<S, D>
where
    S: Data<Elem = T>,
    D: Dimension,
    T: IsNone + Send + Sync,
{
    /// return -1 if all of the elements are NaN
    #[inline]
    fn argmax(&self) -> i32
    where
        T::Inner: PartialOrd,
    {
        self.as_dim1()
            .0
            .titer()
            .vargmax()
            .map(|v| v as i32)
            .unwrap_or(-1)
    }

    /// return -1 if all of the elements are NaN
    #[inline]
    fn argmin(&self) -> i32
    where
        T::Inner: PartialOrd,
    {
        self.as_dim1()
            .0
            .titer()
            .vargmin()
            .map(|v| v as i32)
            .unwrap_or(-1)
    }

    /// first valid value
    #[inline]
    fn first(&self) -> T
    where
        T: Clone,
    {
        if self.len() == 0 {
            unreachable!("first_1d should not be called on an empty array")
        } else {
            unsafe { self.as_dim1().uget(0) }.clone()
        }
    }

    /// last valid value
    #[inline]
    fn last(&self) -> T
    where
        T: Clone,
    {
        let len = self.len();
        if len == 0 {
            unreachable!("last_1d should not be called on an empty array")
        } else {
            unsafe { self.as_dim1().uget(len - 1) }.clone()
        }
    }

    /// first valid value
    #[inline]
    fn valid_first(&self) -> T
    where
        T: IsNone,
    {
        for v in self.as_dim1().iter() {
            if !v.is_none() {
                return v.clone();
            }
        }
        T::none()
    }

    /// last valid value
    #[inline]
    fn valid_last(&self) -> T
    where
        T: IsNone,
    {
        for v in self.as_dim1().iter().rev() {
            if !v.is_none() {
                return v.clone();
            }
        }
        T::none()
    }

    /// Calculate the quantile of the array on a given axis
    #[inline]
    fn quantile(&self, q: f64, method: QuantileMethod) -> f64
    where
        T: IsNone + Cast<f64>,
        T::Inner: Number,
    {
        self.as_dim1().0.vquantile(q, method).unwrap()
    }

    /// Calculate the median of the array on a given axis
    #[inline]
    fn median(&self) -> f64
    where
        T: IsNone + Cast<f64>,
        T::Inner: Number,
    {
        self.quantile_1d(0.5, QuantileMethod::Linear)
    }

    /// sum of the array on a given axis
    #[inline]
    fn prod(&self) -> T
    where
        T: IsNone,
        T::Inner: Number,
    {
        T::from_inner(self.as_dim1().nprod_1d().1)
    }

    /// variance of the array on a given axis
    #[inline]
    fn var(&self, min_periods: usize) -> f64
    where
        T: IsNone,
        T::Inner: Number,
    {
        self.as_dim1().meanvar_1d(min_periods).1
    }

    /// standard deviation of the array on a given axis
    #[inline]
    fn std(&self, min_periods: usize) -> f64
    where
        T: IsNone,
        T::Inner: Number,
    {
        self.var_1d(min_periods).sqrt()
    }

    /// skewness of the array on a given axis
    #[inline]
    fn skew(&self, min_periods: usize) -> f64
    where
        T: IsNone,
        T::Inner: Number,
    {
        self.as_dim1().0.titer().vskew(min_periods)
    }

    /// kurtosis of the array on a given axis
    #[inline]
    fn kurt(&self, min_periods: usize) -> f64
    where
        T: IsNone,
        T::Inner: Number,
    {
        self.as_dim1().0.titer().vkurt(min_periods)
    }

    /// count NaN number of an array on a given axis
    #[lazy_only]
    fn count_none(&self) {}

    /// count not NaN number of an array on a given axis
    #[lazy_only]
    fn count_valid(&self) {}
}

#[cfg(test)]
mod tests {
    use tea_core::prelude::*;
    #[test]
    fn test_arr0_agg() {
        let arr = arr0(1.);
        assert_eq!(arr.mean(1, 0, false).into_scalar().unwrap(), 1.)
    }
}
