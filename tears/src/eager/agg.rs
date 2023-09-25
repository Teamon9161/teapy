use super::super::export::*;
use crate::{BoolType, ExprElement, GetNone};

#[derive(Copy, Clone)]
pub enum QuantileMethod {
    Linear,
    Lower,
    Higher,
    MidPoint,
}

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

    /// production of the array on a given axis, return valid_num n and the production of the array
    pub fn nprod_1d(&self) -> (usize, T)
    where
        T: Number,
    {
        if let Some(slc) = self.as_slice_memory_order() {
            let (n, prod) = utils::vec_nfold(slc, T::one, T::n_prod);
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
}

impl_reduce_nd!(
    argmax,
    #[inline]
    pub fn argmax_1d(&self) -> i32
    {T: Number,}
    {
        let mut max = T::min_();
        let mut max_idx = -1;
        let mut current_idx = 0;
        self.apply(|v| {
            if *v > max {
                max = *v;
                max_idx = current_idx;
            }
            current_idx += 1;
        });
        max_idx
    }
);

impl_reduce_nd!(
    argmin,
    /// return -1 if all of the elements are NaN
    #[inline]
    pub fn argmin_1d(&self) -> i32
    {T: Number,}
    {
        let mut min = T::max_();
        let mut min_idx = -1;
        let mut current_idx = 0;
        self.apply(|v| {
            if *v < min {
                min = *v;
                min_idx = current_idx;
            }
            current_idx += 1;
        });
        min_idx
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
    {T: Number,}
    {
        self.count_by(|v| v.notnan())
    }
);

impl_reduce_nd!(
    count_nan,
    /// count NaN number of an array on a given axis
    #[inline]
    pub fn count_nan_1d(&self) -> i32
    {T: Number,}
    {
        self.count_by(|v| v.isnan())
    }
);

// impl_reduce_nd!(
//     first,
//     /// first valid value
//     #[inline]
//     pub fn first_1d(&self) -> T
//     {T: GetNone; Clone}
//     {
//         if self.len() == 0 {
//             T::none()
//         } else {
//             unsafe {self.uget(0)}.clone()
//         }
//     }
// );

// impl_reduce_nd!(
//     last,
//     /// first valid value
//     #[inline]
//     pub fn last_1d(&self) -> T
//     {T: GetNone; Clone}
//     {
//         let len = self.len();
//         if len == 0 {
//             T::none()
//         } else {
//             unsafe {self.uget(len-1)}.clone()
//         }
//     }
// );

impl_reduce_nd!(
    valid_first,
    /// first valid value
    #[inline]
    pub fn valid_first_1d(&self) -> T
    {T: GetNone; ExprElement}
    {
        // let out = f64::NAN;
        for v in self.iter() {
            if !v.clone().is_none() {
                return v.clone()
            }
        }
        T::none()
    }
);

impl_reduce_nd!(
    valid_last,
    /// last valid value
    #[inline]
    pub fn valid_last_1d(&self) -> T
    {T: GetNone; ExprElement}
    {
        // let out = f64::NAN;
        for v in self.iter().rev() {
            if !v.clone().is_none() {
                return v.clone()
            }
        }
        T::none()
    }
);

impl_reduce_nd!(
    quantile,
    /// Calculate the quantile of the array on a given axis
    pub fn quantile_1d(&self, q: f64, method: QuantileMethod) -> f64
    {T: Number,}
    {
        assert!((0. ..=1.).contains(&q), "q must be between 0 and 1");
        use QuantileMethod::*;
        let mut out_c = self.0.to_owned();  // clone the array
        let slc = out_c.as_slice_mut().unwrap();
        let n = self.count_notnan_1d();
        if n == 0 {
            return f64::NAN;
        } else if n == 1 {
            return slc[0].f64();
        }
        let len_1 = (n - 1).f64();
        let (q, i, j, vi, vj) = if q <= 0.5 {
            let q_idx = len_1 * q;
            let (i, j) =  (q_idx.floor().usize(), q_idx.ceil().usize());
            let (head, m, _tail) = slc.select_nth_unstable_by(j, |va, vb| {va.nan_sort_cmp(vb)});
            if i != j {
                let vi = utils::vec_fold(head, T::min_, T::max_with);
                let vi = if vi == T::min_() {
                    f64::NAN
                } else {
                    vi.f64()
                };
                (q, i, j, vi, m.f64())
            } else {
                return m.f64();
            }
        } else {
            // sort from largest to smallest
            let q = 1. - q;
            let q_idx = len_1 * q;
            let (i, j) =  (q_idx.floor().usize(), q_idx.ceil().usize());
            let (head, m, _tail) = slc.select_nth_unstable_by(j, |va, vb| {va.nan_sort_cmp_rev(vb)});
            if i != j {
                let vi = utils::vec_fold(head, T::max_, T::min_with);
                let vi = if vi == T::max_() {
                    f64::NAN
                } else {
                    vi.f64()
                };
                match method {
                    Lower => {return m.f64();}
                    Higher => {return vi;}
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
            },
            Lower => { vi }, // i
            Higher => { vj }, // j
            MidPoint => { (vi + vj) / 2. }// (i + j) / 2.
        }
    }
);

impl_reduce_nd!(
    median,
    /// Calculate the median of the array on a given axis
    pub fn median_1d(&self) -> f64
    {T: Number,}
    {
        self.quantile_1d(0.5, QuantileMethod::Linear)
    }
);

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
    prod,
    /// sum of the array on a given axis
    #[inline]
    pub fn prod_1d(&self) -> T
    {T: Number,}
    {
        self.nprod_1d().1
    }
);

impl_reduce_nd!(
    mean,
    /// mean of the array on a given axis
    #[inline]
    pub fn mean_1d(&self, stable: bool) -> f64
    {T: Number,}
    {
        let(n, sum) = self.nsum_1d(stable);
        sum.f64() / n.f64()
    }
);

impl_reduce_nd!(
    meanvar,
    /// mean and variance of the array on a given axis
    pub fn meanvar_1d(&self, stable: bool) -> (f64, f64)
    {T: Number,}
    {
        let (mut m1, mut m2) = (0., 0.);
        if !stable {
            let n = self.n_apply_valid(|v| {
                let v = v.f64();
                m1 += v;
                m2 += v * v;
            });
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
            let mean = self.mean_1d(false);
            if mean.isnan() {
                return (f64::NAN, f64::NAN);
            }
            // elements minus mean then calculate using Kahan summation
            // see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            let (mut c_v, mut c_v2) = (0., 0.);
            let n = self.n_apply_valid(|v| {
                let v = v.f64() - mean;
                m1 = kh_sum(m1, v, &mut c_v);
                m2 = kh_sum(m2, v * v, &mut c_v2);
            });
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
);

impl_reduce_nd!(
    var,
    /// variance of the array on a given axis
    pub fn var_1d(&self, stable: bool) -> f64
    {T: Number,}
    {
        self.meanvar_1d(stable).1
    }
);

impl_reduce_nd!(
    std,
    /// standard deviation of the array on a given axis
    #[inline]
    pub fn std_1d(&self, stable: bool) -> f64
    {T: Number,}
    {
        self.var_1d(stable).sqrt()
    }
);

impl_reduce_nd!(
    skew,
    /// skewness of the array on a given axis
    #[inline]
    pub fn skew_1d(&self, stable: bool) -> f64
    {T: Number,}
    {
        let (mut m1, mut m2, mut m3) = (0., 0., 0.);
        let n = if !stable {
            self.n_apply_valid(|v| {
                let v = v.f64();
                m1 += v;
                let v2 = v * v;
                m2 += v2;
                m3 += v2 * v;
            })
        } else {
            // calculate mean of the array
            let mean = self.mean_1d(false);
            if mean.isnan() {
                return f64::NAN;
            }
            // elements minus mean then calculate using Kahan summation
            // see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            let (mut c_m1, mut c_m2, mut c_m3) = (0., 0., 0.);
            self.n_apply_valid(|v| {
                let v = v.f64() - mean;
                m1 = kh_sum(m1, v, &mut c_m1);
                let v2 = v * v;
                m2 = kh_sum(m2, v2, &mut c_m2);
                m3 = kh_sum(m3, v2 * v, &mut c_m3);
            })
        };
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
);

impl_reduce_nd!(
    kurt,
    /// kurtosis of the array on a given axis
    #[inline]
    pub fn kurt_1d(&self, stable: bool) -> f64
    {T: Number,}
    {
        let (mut m1, mut m2, mut m3, mut m4) = (0., 0., 0., 0.);
        let n = if !stable {
            self.n_apply_valid(|v| {
                let v = v.f64();
                m1 += v;
                let v2 = v * v;
                m2 += v2;
                m3 += v2 * v;
                m4 += v2 * v2;
            })
        } else {
            // calculate mean of the array
            let mean = self.mean_1d(false);
            if mean.isnan() {
                return f64::NAN;
            }
            // elements minus mean then calculate using Kahan summation
            // see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            let (mut c_m1, mut c_m2, mut c_m3, mut c_m4) = (0., 0., 0., 0.);
            self.n_apply_valid(|v| {
                let v = v.f64() - mean;
                let v2 = v * v;
                m1 = kh_sum(m1, v, &mut c_m1);
                m2 = kh_sum(m2, v2, &mut c_m2);
                m3 = kh_sum(m3, v2 * v, &mut c_m3);
                m4 = kh_sum(m4, v2 * v2, &mut c_m4);
            })
        };
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
