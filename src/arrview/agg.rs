use super::prelude::*;

impl_arrview!([ArrView1, ArrViewMut1], Number, {
    /// count NaN number of an array
    #[inline]
    pub fn count_nan(&self) -> usize {
        self.count_by(|v| v.isnan())
    }

    /// count not NaN number of an array
    #[inline]
    pub fn count_notnan(&self) -> usize {
        self.count_by(|v| v.notnan())
    }

    /// Max value of the array
    pub fn max(&self) -> f64 {
        let mut max = T::min_();
        let n = self.n_apply_valid(|v| {
            if max < *v {
                max = *v;
            }
        });
        if n >= 1 {
            max.f64()
        } else {
            f64::NAN
        }
    }

    /// Min value of the array
    pub fn min(&self) -> f64 {
        let mut min = T::max_();
        let n = self.n_apply_valid(|v| {
            if min > *v {
                min = *v;
            }
        });
        if n >= 1 {
            min.f64()
        } else {
            f64::NAN
        }
    }

    /// sum of the array
    pub fn sum(&self, stable: bool) -> f64 {
        let (n, acc) = if !stable {
            self.n_acc_valid(0., |v| v.f64())
        } else {
            self.stable_n_acc_valid(0., |v| v.f64())
        };
        if n >= 1 {
            acc
        } else {
            f64::NAN
        }
    }

    /// mean of the array
    pub fn mean(&self, stable: bool) -> f64 {
        let (n, acc) = if !stable {
            self.n_acc_valid(0., |v| v.f64())
        } else {
            self.stable_n_acc_valid(0., |v| v.f64())
        };
        if n >= 1 {
            acc / n.f64()
        } else {
            f64::NAN
        }
    }

    // variance of 1d array
    pub fn var(&self, bias: bool, stable: bool) -> f64 {
        let (mut m1, mut m2) = (0., 0.);
        let n = if !stable {
            self.n_apply_valid(|v| {
                let v = v.f64();
                m1 += v;
                m2 += v * v;
            })
        } else {
            // calculate mean of the array
            let mean = self.mean(false);
            if mean.isnan() {
                return f64::NAN;
            }
            // elements minus mean then calculate using Kahan summation
            // see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            let (mut c_v, mut c_v2) = (0., 0.);
            self.n_apply_valid(|v| {
                let v = v.f64() - mean;
                m1 = kh_sum(m1, v, &mut c_v);
                m2 = kh_sum(m2, v * v, &mut c_v2);
            })
        };
        let n_f64 = n.f64();
        m1 /= n_f64; // E(x)
        m2 /= n_f64; // E(x^2)
        m2 -= m1.powi(2); // variance = E(x^2) - (E(x))^2
        if m2 <= 1e-14 {
            return 0.;
        }
        if bias {
            if n >= 1 {
                m2
            } else {
                f64::NAN
            }
        } else if n >= 2 {
            m2 * n_f64 / (n - 1).f64()
        } else {
            f64::NAN
        }
    }

    /// standard deviation of array
    pub fn std(&self, bias: bool, stable: bool) -> f64 {
        let var = self.var(bias, stable);
        if var == 0. {
            0.
        } else if var.notnan() {
            var.sqrt()
        } else {
            f64::NAN
        }
    }

    /// skewness of array
    pub fn skew(&self, stable: bool) -> f64 {
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
            let mean = self.mean(false);
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

    /// kurtosis of array
    pub fn kurt(&self, stable: bool) -> f64 {
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
            let mean = self.mean(false);
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
});
