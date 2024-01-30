mod corr;
#[cfg(feature = "lazy")]
mod impl_lazy;

#[cfg(feature = "lazy")]
pub use corr::AutoExprAgg2Ext;
pub use corr::{Agg2Ext, CorrMethod, CorrToolExt1d};
#[cfg(feature = "lazy")]
pub use impl_lazy::{corr, AutoExprAggExt, DataDictCorrExt, ExprAggExt};
#[cfg(feature = "lazy")]
use lazy::Expr;
use ndarray::{Data, Dimension, Ix1, Zip};
use tea_core::prelude::*;
use tea_core::utils::{kh_sum, vec_fold, vec_nfold};

#[ext_trait]
impl<T, S: Data<Elem = T>> AggExt1d for ArrBase<S, Ix1> {
    /// sum of the array on a given axis, return valid_num n and the sum of the array
    fn nsum_1d(&self, stable: bool) -> (usize, T)
    where
        T: Number,
    {
        if !stable {
            if let Some(slc) = self.as_slice_memory_order() {
                let (n, sum) = vec_nfold(slc, T::zero, T::n_add);
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

    /// production of the array on a given axis, return valid_num n and the production of the array
    fn nprod_1d(&self) -> (usize, T)
    where
        T: Number,
    {
        if let Some(slc) = self.as_slice_memory_order() {
            let (n, prod) = vec_nfold(slc, T::one, T::n_prod);
            return (n, prod);
        }
        // fall back to normal calculation
        let (n, acc) = self.n_fold_valid(T::one(), |acc, v| acc * *v);
        if n >= 1 {
            (n, acc)
        } else {
            (0, T::nan())
        }
    }

    /// mean and variance of the array on a given axis
    pub fn meanvar_1d(&self, min_periods: usize, stable: bool) -> (f64, f64)
    where
        T: Number,
    {
        let arr = self.as_dim1();
        let (mut m1, mut m2) = (0., 0.);
        if !stable {
            let n = arr.n_apply_valid(|v| {
                let v = v.f64();
                m1 += v;
                m2 += v * v;
            });
            if n < min_periods {
                return (f64::NAN, f64::NAN);
            }
            let n_f64 = n.f64();
            m1 /= n_f64; // E(x)
            m2 /= n_f64; // E(x^2)
            m2 -= m1.powi(2); // variance = E(x^2) - (E(x))^2
            if m2 <= 1e-14 {
                (m1, 0.)
            } else if n >= 2 {
                (m1, m2 * n_f64 / (n - 1).f64())
            } else {
                (f64::NAN, f64::NAN)
            }
        } else {
            // calculate mean of the array
            let mean = arr.mean_1d(min_periods, true);
            if mean.isnan() {
                return (f64::NAN, f64::NAN);
            }
            // elements minus mean then calculate using Kahan summation
            // see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            let (mut c_v, mut c_v2) = (0., 0.);
            let n = arr.n_apply_valid(|v| {
                let v = v.f64() - mean;
                m1 = kh_sum(m1, v, &mut c_v);
                m2 = kh_sum(m2, v * v, &mut c_v2);
            });
            if n < min_periods {
                return (f64::NAN, f64::NAN);
            }
            let n_f64 = n.f64();
            m1 /= n_f64; // E(x)
            m2 /= n_f64; // E(x^2)
            m2 -= m1.powi(2); // variance = E(x^2) - (E(x))^2
            if m2 <= 1e-14 {
                (mean, 0.)
            } else if n >= 2 {
                (mean, m2 * n_f64 / (n - 1).f64())
            } else {
                (f64::NAN, f64::NAN)
            }
        }
    }

    #[cfg(feature = "map")]
    fn umax_1d(&self) -> T
    where
        T: PartialEq + Clone + Number,
    {
        use crate::map::MapExt1d;
        self.sorted_unique_1d().max_1d()
    }

    #[cfg(feature = "map")]
    fn umin_1d(&self) -> T
    where
        T: PartialEq + Clone + Number,
    {
        use crate::map::MapExt1d;
        self.sorted_unique_1d().min_1d()
    }

    pub fn cut_nsum_1d<S2>(
        &self,
        mask: &ArrBase<S2, Ix1>,
        min_periods: usize,
        // stable: bool,
    ) -> (usize, T)
    where
        T: Number,
        S2: Data<Elem = bool>,
        // T2: Cast<bool>
    {
        let mut n = 0_usize;
        let sum = self.fold_with(mask.view(), T::zero(), |acc, v, valid| {
            if (*valid) && v.notnan() {
                n += 1;
                acc + *v
            } else {
                acc
            }
        });
        if n >= min_periods {
            (n, sum)
        } else {
            (n, T::nan())
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
    pub fn cut_mean_1d<S2>(
        &self,
        mask: &ArrBase<S2, Ix1>,
        min_periods: usize,
        // stable: bool,
    ) -> f64
    where
        T: Number,
        S2: Data<Elem = bool>,
    {
        let (n, sum) = self.cut_nsum_1d(mask, min_periods);
        sum.f64() / n.f64()
    }
}

#[arr_agg_ext(lazy = "view", type = "numeric")]
impl<S, D, T> AggExtNd<D, T> for ArrBase<S, D>
where
    S: Data<Elem = T>,
    D: Dimension,
    T: Send + Sync,
{
    /// return -1 if all of the elements are NaN
    fn argmax(&self) -> i32
    where
        T: Number,
    {
        let mut max = T::min_();
        let mut max_idx = -1;
        let mut current_idx = 0;
        self.as_dim1().apply(|v| {
            if *v > max {
                max = *v;
                max_idx = current_idx;
            }
            current_idx += 1;
        });
        max_idx
    }

    /// return -1 if all of the elements are NaN
    fn argmin(&self) -> i32
    where
        T: Number,
    {
        let mut min = T::max_();
        let mut min_idx = -1;
        let mut current_idx = 0;
        self.as_dim1().apply(|v| {
            if *v < min {
                min = *v;
                min_idx = current_idx;
            }
            current_idx += 1;
        });
        min_idx
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
        T: Clone + GetNone,
    {
        for v in self.as_dim1().iter() {
            if !v.clone().is_none() {
                return v.clone();
            }
        }
        T::none()
    }

    /// last valid value
    #[inline]
    fn valid_last(&self) -> T
    where
        T: Clone + GetNone,
    {
        for v in self.as_dim1().iter().rev() {
            if !v.clone().is_none() {
                return v.clone();
            }
        }
        T::none()
    }

    /// Calculate the quantile of the array on a given axis
    fn quantile(&self, q: f64, method: QuantileMethod) -> f64
    where
        T: Number,
    {
        assert!((0. ..=1.).contains(&q), "q must be between 0 and 1");
        use QuantileMethod::*;
        let arr = self.as_dim1();
        let mut out_c = arr.0.to_owned(); // clone the array
        let slc = out_c.as_slice_mut().unwrap();
        let n = arr.count_notnan_1d();
        if n == 0 {
            return f64::NAN;
        } else if n == 1 {
            return slc[0].f64();
        }
        let len_1 = (n - 1).f64();
        let (q, i, j, vi, vj) = if q <= 0.5 {
            let q_idx = len_1 * q;
            let (i, j) = (q_idx.floor().usize(), q_idx.ceil().usize());
            let (head, m, _tail) = slc.select_nth_unstable_by(j, |va, vb| va.nan_sort_cmp(vb));
            if i != j {
                let vi = vec_fold(head, T::min_, T::max_with);
                let vi = if vi == T::min_() { f64::NAN } else { vi.f64() };
                (q, i, j, vi, m.f64())
            } else {
                return m.f64();
            }
        } else {
            // sort from largest to smallest
            let q = 1. - q;
            let q_idx = len_1 * q;
            let (i, j) = (q_idx.floor().usize(), q_idx.ceil().usize());
            let (head, m, _tail) = slc.select_nth_unstable_by(j, |va, vb| va.nan_sort_cmp_rev(vb));
            if i != j {
                let vi = vec_fold(head, T::max_, T::min_with);
                let vi = if vi == T::max_() { f64::NAN } else { vi.f64() };
                match method {
                    Lower => {
                        return m.f64();
                    }
                    Higher => {
                        return vi;
                    }
                    _ => {}
                };
                (q, i, j, vi, m.f64())
            } else {
                return m.f64();
            }
        };
        match method {
            Linear => {
                // `i + (j - i) * fraction`, where `fraction` is the
                // fractional part of the index surrounded by `i` and `j`.
                let (qi, qj) = (i.f64() / len_1, j.f64() / len_1);
                let fraction = (q - qi) / (qj - qi);
                vi + (vj - vi) * fraction
            }
            Lower => vi,                // i
            Higher => vj,               // j
            MidPoint => (vi + vj) / 2., // (i + j) / 2.
        }
    }

    /// Calculate the median of the array on a given axis
    #[inline]
    fn median(&self) -> f64
    where
        T: Number,
    {
        self.quantile_1d(0.5, QuantileMethod::Linear)
    }

    /// sum of the array on a given axis
    #[inline]
    fn prod(&self) -> T
    where
        T: Number,
    {
        self.as_dim1().nprod_1d().1
    }

    /// variance of the array on a given axis
    fn var(&self, min_periods: usize, stable: bool) -> f64
    where
        T: Number,
    {
        self.as_dim1().meanvar_1d(min_periods, stable).1
    }

    /// standard deviation of the array on a given axis
    #[inline]
    fn std(&self, min_periods: usize, stable: bool) -> f64
    where
        T: Number,
    {
        self.var_1d(min_periods, stable).sqrt()
    }

    /// skewness of the array on a given axis
    fn skew(&self, min_periods: usize, stable: bool) -> f64
    where
        T: Number,
    {
        let arr = self.as_dim1();
        let (mut m1, mut m2, mut m3) = (0., 0., 0.);
        let n = if !stable {
            arr.n_apply_valid(|v| {
                let v = v.f64();
                m1 += v;
                let v2 = v * v;
                m2 += v2;
                m3 += v2 * v;
            })
        } else {
            // calculate mean of the array
            let mean = arr.mean_1d(min_periods, true);
            if mean.isnan() {
                return f64::NAN;
            }
            // elements minus mean then calculate using Kahan summation
            // see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            let (mut c_m1, mut c_m2, mut c_m3) = (0., 0., 0.);
            arr.n_apply_valid(|v| {
                let v = v.f64() - mean;
                m1 = kh_sum(m1, v, &mut c_m1);
                let v2 = v * v;
                m2 = kh_sum(m2, v2, &mut c_m2);
                m3 = kh_sum(m3, v2 * v, &mut c_m3);
            })
        };
        if n < min_periods {
            return f64::NAN;
        }
        let mut res = if n >= 3 {
            let n_f64 = n.f64();
            m1 /= n_f64; // Ex
            m2 /= n_f64; // Ex^2
            let var = m2 - m1.powi(2);
            if var <= 1e-14 {
                0.
            } else {
                let std = var.sqrt(); // var^2
                m3 /= n_f64; // Ex^3
                let mean_std = m1 / std; // mean / std
                m3 / std.powi(3) - 3_f64 * mean_std - mean_std.powi(3)
            }
        } else {
            f64::NAN
        };
        if res.notnan() && res != 0. {
            let adjust = (n * (n - 1)).f64().sqrt() / (n - 2).f64();
            res *= adjust;
        }
        res
    }

    /// kurtosis of the array on a given axis
    fn kurt(&self, min_periods: usize, stable: bool) -> f64
    where
        T: Number,
    {
        let arr = self.as_dim1();
        let (mut m1, mut m2, mut m3, mut m4) = (0., 0., 0., 0.);
        let n = if !stable {
            arr.n_apply_valid(|v| {
                let v = v.f64();
                m1 += v;
                let v2 = v * v;
                m2 += v2;
                m3 += v2 * v;
                m4 += v2 * v2;
            })
        } else {
            // calculate mean of the array
            let mean = arr.mean_1d(min_periods, true);
            if mean.isnan() {
                return f64::NAN;
            }
            // elements minus mean then calculate using Kahan summation
            // see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            let (mut c_m1, mut c_m2, mut c_m3, mut c_m4) = (0., 0., 0., 0.);
            arr.n_apply_valid(|v| {
                let v = v.f64() - mean;
                let v2 = v * v;
                m1 = kh_sum(m1, v, &mut c_m1);
                m2 = kh_sum(m2, v2, &mut c_m2);
                m3 = kh_sum(m3, v2 * v, &mut c_m3);
                m4 = kh_sum(m4, v2 * v2, &mut c_m4);
            })
        };
        if n < min_periods {
            return f64::NAN;
        }
        let mut res = if n >= 4 {
            let n_f64 = n.f64();
            m1 /= n_f64; // Ex
            m2 /= n_f64; // Ex^2
            let var = m2 - m1.powi(2);
            if var <= 1e-14 {
                0.
            } else {
                let var2 = var.powi(2); // var^2
                m4 /= n_f64; // Ex^4
                m3 /= n_f64; // Ex^3
                let mean2_var = m1.powi(2) / var; // (mean / std)^2
                (m4 - 4. * m1 * m3) / var2 + 6. * mean2_var + 3. * mean2_var.powi(2)
            }
        } else {
            f64::NAN
        };
        if res.notnan() && res != 0. {
            res = 1. / ((n - 2) * (n - 3)).f64()
                * ((n.pow(2) - 1).f64() * res - (3 * (n - 1).pow(2)).f64())
        }
        res
    }

    /// count NaN number of an array on a given axis
    #[lazy_only]
    fn count_nan(&self) {}

    /// count not NaN number of an array on a given axis
    #[lazy_only]
    fn count_notnan(&self) {}
}

#[allow(dead_code)]
#[derive(Copy, Clone)]
pub enum QuantileMethod {
    Linear,
    Lower,
    Higher,
    MidPoint,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tea_core::prelude::*;

    #[test]
    fn test_argmax() {
        let arr = Arr1::from_vec(vec![1, 2, 3]);
        assert_eq!(arr.argmax_1d(), 2);
        assert_eq!(*arr.argmax(0, false).to_dim1().unwrap().get(0).unwrap(), 2);
    }
}
