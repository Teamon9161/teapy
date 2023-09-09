use super::super::export::*;

#[derive(Copy, Clone)]
pub enum CorrMethod {
    Pearson,
    Spearman,
}

impl_reduce2_nd!(
    cov,
    /// covariance of 2 array
    pub fn cov_1d<S2, T2>(&self, other: &ArrBase<S2, Ix1>, stable: bool) -> f64
    {where S2: Data<Elem = T2>, T: Number, T2: Number,}
    {
        assert_eq!(self.len(), other.len(), "Both arrays must be the same length when calculating covariance.");
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
            let (mean_a, mean_b) = (self.mean_1d(false), other.mean_1d(false));
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
);

impl_reduce2_nd!(
    corr_pearson,
    /// Pearson correlation of 2 array
    pub fn corr_pearson_1d<S2, T2>(&self, other: &ArrBase<S2, Ix1>, stable: bool) -> f64
    {where S2: Data<Elem = T2>, T: Number, T2: Number,}
    {
        assert_eq!(self.len(), other.len(), "Both arrays must be the same length when calculating correlation.");
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
            let (mean_a, mean_b) = (self.mean_1d(false), other.mean_1d(false));
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
);

impl_reduce2_nd!(
    corr_spearman,
    /// Spearman correlation of 2 array
    pub fn corr_spearman_1d<S2, T2>(&self, other: &ArrBase<S2, Ix1>, stable: bool) -> f64
    {where S2: Data<Elem = T2>, T: Number, T2: Number,}
    {
        assert_eq!(self.len(), other.len(), "Both arrays must be the same length when calculating correlation.");
        let (arr1, arr2) = self.remove_nan2_1d(other);
        let mut rank1 = Arr1::<f64>::uninit(arr1.raw_dim());
        let mut rank2 = Arr1::<f64>::uninit(arr2.raw_dim());
        arr1.rank_1d(&mut rank1.view_mut(), false, false);
        arr2.rank_1d(&mut rank2.view_mut(), false, false);
        unsafe{
            let rank1 = rank1.assume_init();
            let rank2 = rank2.assume_init();
            rank1.corr_pearson_1d(&rank2, stable)
        }
    }
);

impl_reduce2_nd!(
    corr,
    /// correlation of 2 array
    pub fn corr_1d<S2, T2>(&self, other: &ArrBase<S2, Ix1>, method: CorrMethod, stable: bool) -> f64
    {where S2: Data<Elem = T2>, T: Number, T2: Number,}
    {
        match method {
            CorrMethod::Pearson => self.corr_pearson_1d(other, stable),
            CorrMethod::Spearman => self.corr_spearman_1d(other, stable),
            // _ => panic!("Not supported method: {} in correlation", method),
        }
    }
);
