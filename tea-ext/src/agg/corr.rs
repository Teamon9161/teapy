#[cfg(feature = "lazy")]
use lazy::Expr;
use ndarray::{Data, DimMax, Dimension, Ix1, Zip};
use std::iter::zip;
use tea_core::prelude::*;
use tea_core::utils::kh_sum;

#[derive(Copy, Clone)]
pub enum CorrMethod {
    Pearson,
    #[cfg(feature = "map")]
    Spearman,
}

#[ext_trait]
impl<T, S: Data<Elem = T>> CorrToolExt1d for ArrBase<S, Ix1> {
    /// Remove NaN values in two 1d arrays.
    #[inline]
    pub fn remove_nan2_1d<S2, T2>(&self, other: &ArrBase<S2, Ix1>) -> (Arr1<T>, Arr1<T2>)
    where
        T: Number,
        S2: Data<Elem = T2>,
        T2: Number,
    {
        let (out1, out2): (Vec<_>, Vec<_>) = zip(self, other)
            .filter(|(v1, v2)| v1.notnan() & v2.notnan())
            .unzip();
        (Arr1::from_vec(out1), Arr1::from_vec(out2))
    }

    pub fn weight_mean_1d<S2, T2>(
        &self,
        other: &ArrBase<S2, Ix1>,
        min_periods: usize,
        stable: bool,
    ) -> f64
    where
        T: Number,
        S2: Data<Elem = T2>,
        T2: Number,
    {
        let weight_sum = other.sum_1d(stable);
        debug_assert_eq!(self.len(), other.len());
        let len = self.len();
        let mut sum = 0.;
        let mut nan_num = 0;
        for i in 0..len {
            let v1 = unsafe { *self.uget(i) };
            let v2 = unsafe { *other.uget(i) };
            if v1.notnan() & v2.notnan() {
                sum += v1.f64() * v2.f64();
            } else {
                nan_num += 1;
            }
        }
        if len - nan_num >= min_periods {
            sum / weight_sum.f64()
        } else {
            f64::NAN
        }
    }
}

#[arr_agg2_ext(lazy = "view2", type = "numeric", type2 = "numeric")]
impl<T, D: Dimension, S: Data<Elem = T>> Agg2Ext for ArrBase<S, D> {
    /// covariance of 2 array
    fn cov<S2, D2, T2>(&self, other: &ArrBase<S2, D2>, min_periods: usize, stable: bool) -> f64
    where
        S2: Data<Elem = T2>,
        D2: Dimension,
        D: DimMax<D2>,
        T: Number,
        T2: Number,
    {
        assert_eq!(
            self.len(),
            other.len(),
            "Both arrays must be the same length when calculating covariance."
        );
        let (mut sum_a, mut sum_b, mut sum_ab) = (0., 0., 0.);
        let arr = self.as_dim1();
        let other_arr = other.as_dim1();
        let n = if !stable {
            arr.n_apply_valid_with(&other_arr, |va, vb| {
                let (va, vb) = (va.f64(), vb.f64());
                sum_a += va;
                sum_b += vb;
                sum_ab += va * vb;
            })
        } else {
            // Kahan summation, see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            let (mut c_a, mut c_b, mut c_ab) = (0., 0., 0.);
            let (mean_a, mean_b) = (
                arr.mean_1d(min_periods, true),
                other_arr.mean_1d(min_periods, true),
            );
            arr.n_apply_valid_with(&other_arr, |va, vb| {
                let (va, vb) = (va.f64() - mean_a, vb.f64() - mean_b);
                sum_a = kh_sum(sum_a, va, &mut c_a);
                sum_b = kh_sum(sum_b, vb, &mut c_b);
                sum_ab = kh_sum(sum_ab, va * vb, &mut c_ab);
            })
        };
        if n < min_periods {
            return f64::NAN;
        }
        if n >= 2 {
            (sum_ab - (sum_a * sum_b) / n.f64()) / (n - 1).f64()
        } else {
            f64::NAN
        }
    }

    #[lazy_exclude]
    fn corr_pearson<S2, D2, T2>(
        &self,
        other: &ArrBase<S2, D2>,
        min_periods: usize,
        stable: bool,
    ) -> f64
    where
        S2: Data<Elem = T2>,
        D2: Dimension,
        D: DimMax<D2>,
        T: Number,
        T2: Number,
    {
        assert_eq!(
            self.len(),
            other.len(),
            "Both arrays must be the same length when calculating correlation."
        );
        let (mut sum_a, mut sum2_a, mut sum_b, mut sum2_b, mut sum_ab) = (0., 0., 0., 0., 0.);
        let arr = self.as_dim1();
        let other_arr = other.as_dim1();
        let n = if !stable {
            arr.n_apply_valid_with(&other_arr, |va, vb| {
                let (va, vb) = (va.f64(), vb.f64());
                sum_a += va;
                sum2_a += va.powi(2);
                sum_b += vb;
                sum2_b += vb.powi(2);
                sum_ab += va * vb;
            })
        } else {
            // Kahan summation, see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            let (mut c_a, mut c_a2, mut c_b, mut c_b2, mut c_ab) = (0., 0., 0., 0., 0.);
            let (mean_a, mean_b) = (
                arr.mean_1d(min_periods, true),
                other_arr.mean_1d(min_periods, true),
            );
            arr.n_apply_valid_with(&other_arr, |va, vb| {
                let (va, vb) = (va.f64() - mean_a, vb.f64() - mean_b);
                sum_a = kh_sum(sum_a, va, &mut c_a);
                sum2_a = kh_sum(sum2_a, va * va, &mut c_a2);
                sum_b = kh_sum(sum_b, vb, &mut c_b);
                sum2_b = kh_sum(sum2_b, vb * vb, &mut c_b2);
                sum_ab = kh_sum(sum_ab, va * vb, &mut c_ab);
            })
        };
        if n < min_periods {
            return f64::NAN;
        }
        if n >= 2 {
            let n = n.f64();
            let mean_a = sum_a / n;
            let mut var_a = sum2_a / n;
            let mean_b = sum_b / n;
            let mut var_b = sum2_b / n;
            var_a -= mean_a.powi(2);
            var_b -= mean_b.powi(2);
            if (var_a > 1e-14) & (var_b > 1e-14) {
                let exy = sum_ab / n;
                let exey = sum_a * sum_b / n.powi(2);
                (exy - exey) / (var_a * var_b).sqrt()
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        }
    }

    #[cfg(feature = "map")]
    #[lazy_exclude]
    fn corr_spearman<S2, D2, T2>(
        &self,
        other: &ArrBase<S2, D2>,
        min_periods: usize,
        stable: bool,
    ) -> f64
    where
        S2: Data<Elem = T2>,
        D2: Dimension,
        D: DimMax<D2>,
        T: Number,
        T2: Number,
    {
        use crate::map::*;
        assert_eq!(
            self.len(),
            other.len(),
            "Both arrays must be the same length when calculating correlation."
        );
        let (arr1, arr2) = self.as_dim1().remove_nan2_1d(&other.as_dim1());
        let mut rank1 = Arr1::<f64>::uninit(arr1.raw_dim());
        let mut rank2 = Arr1::<f64>::uninit(arr2.raw_dim());
        arr1.rank_1d(&mut rank1.view_mut(), false, false);
        arr2.rank_1d(&mut rank2.view_mut(), false, false);
        unsafe {
            let rank1 = rank1.assume_init();
            let rank2 = rank2.assume_init();
            rank1.corr_pearson_1d(&rank2, min_periods, stable)
        }
    }

    fn corr<S2, D2, T2>(
        &self,
        other: &ArrBase<S2, D2>,
        method: CorrMethod,
        min_periods: usize,
        stable: bool,
    ) -> f64
    where
        S2: Data<Elem = T2>,
        D2: Dimension,
        D: DimMax<D2>,
        T: Number,
        T2: Number,
    {
        match method {
            CorrMethod::Pearson => self.corr_pearson_1d(other, min_periods, stable),
            #[cfg(feature = "map")]
            CorrMethod::Spearman => self.corr_spearman_1d(other, min_periods, stable),
            // _ => panic!("Not supported method: {} in correlation", method),
        }
    }
}
