use super::prelude::*;

impl_arrview!([ArrView1, ArrViewMut1], Number, {
    /// covariance of 2 array
    pub fn cov<S: Number>(&self, other: &ArrView1<S>, stable: bool) -> f64
    where
        usize: Number,
    {
        let (mut sum_a, mut sum_b, mut sum_ab) = (0., 0., 0.);
        let n = if !stable {
            self.n_apply_valid_with(other, |va, vb| {
                let (va, vb) = (va.f64(), vb.f64());
                sum_a += va;
                sum_b += vb;
                sum_ab += va * vb;
            })
        } else {
            // Kahan summation, see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            let (mut c_a, mut c_b, mut c_ab) = (0., 0., 0.);
            let (mean_a, mean_b) = (self.mean(false), other.mean(false));
            self.n_apply_valid_with(other, |va, vb| {
                let (va, vb) = (va.f64() - mean_a, vb.f64() - mean_b);
                sum_a = kh_sum(sum_a, va, &mut c_a);
                sum_b = kh_sum(sum_b, vb, &mut c_b);
                sum_ab = kh_sum(sum_ab, va * vb, &mut c_ab);
            })
        };
        if n >= 2 {
            (sum_ab - (sum_a * sum_b) / n.f64()) / (n - 1).f64()
        } else {
            f64::NAN
        }
    }

    /// correlation of 2 array
    pub fn corr<S: Number>(&self, other: &ArrView1<S>, stable: bool) -> f64
    where
        usize: Number,
    {
        let (mut sum_a, mut sum2_a, mut sum_b, mut sum2_b, mut sum_ab) = (0., 0., 0., 0., 0.);
        let n = if !stable {
            self.n_apply_valid_with(other, |va, vb| {
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
            let (mean_a, mean_b) = (self.mean(false), other.mean(false));
            self.n_apply_valid_with(other, |va, vb| {
                let (va, vb) = (va.f64() - mean_a, vb.f64() - mean_b);
                sum_a = kh_sum(sum_a, va, &mut c_a);
                sum2_a = kh_sum(sum2_a, va * va, &mut c_a2);
                sum_b = kh_sum(sum_b, vb, &mut c_b);
                sum2_b = kh_sum(sum2_b, vb * vb, &mut c_b2);
                sum_ab = kh_sum(sum_ab, va * vb, &mut c_ab);
            })
        };
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
});
