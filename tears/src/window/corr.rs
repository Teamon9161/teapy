use super::super::export::*;

impl_map2_nd!(
    ts_cov,
    pub fn ts_cov_1d<S2, T2, S3>(
        &self,
        other: &ArrBase<S2, D>,
        out: &mut ArrBase<S3, D>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    {where T: Number, T2: Number,}
    {
        let window = min(self.len(), window);
        let window = min(other.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut sum_a = 0.;
        let mut sum_b = 0.;
        let mut sum_ab = 0.;
        let mut n = 0;
        if !stable {
            self.apply_window_with_to(other, out, window, |va, vb, va_rm, vb_rm| {
                if va.notnan() && vb.notnan() {
                    n += 1;
                    let (va, vb) = (va.f64(), vb.f64());
                    sum_a += va;
                    sum_b += vb;
                    sum_ab += va * vb;
                };
                let res = if n >= min_periods {
                    (sum_ab - (sum_a * sum_b) / n.f64()) / (n - 1).f64()
                } else {
                    f64::NAN
                };
                if let (Some(va), Some(vb)) = (va_rm, vb_rm) {
                    if va.notnan() && vb.notnan() {
                        n -= 1;
                        let (va, vb) = (va.f64(), vb.f64());
                        sum_a -= va;
                        sum_b -= vb;
                        sum_ab -= va * vb;
                    };
                }
                res
            });
        } else {
            define_c!(c1, c2, c3, c4, c5, c6);
            self.stable_apply_window_with_to(other, out, window, |va, vb, va_rm, vb_rm| {
                if va.notnan() && vb.notnan() {
                    n += 1;
                    sum_a.kh_sum(va, c1);
                    sum_b.kh_sum(vb, c2);
                    sum_ab.kh_sum(va * vb, c3);
                };
                let res = if n >= min_periods {
                    (sum_ab - (sum_a * sum_b) / n.f64()) / (n - 1).f64()
                } else {
                    f64::NAN
                };
                if va_rm.notnan() && vb_rm.notnan() {
                    n -= 1;
                    sum_a.kh_sum(-va_rm, c4);
                    sum_b.kh_sum(-vb_rm, c5);
                    sum_ab.kh_sum(-va_rm * vb_rm, c6);
                };
                res
            })
        }
    }
);

impl_map2_nd!(
    ts_corr,
    pub fn ts_corr_1d<S2, T2, S3>(
        &self,
        other: &ArrBase<S2, D>,
        out: &mut ArrBase<S3, D>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    {where T: Number, T2: Number,}
    {
        let window = min(self.len(), window);
        let window = min(other.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut sum_a = 0.;
        let mut sum2_a = 0.;
        let mut sum_b = 0.;
        let mut sum2_b = 0.;
        let mut sum_ab = 0.;
        let mut n = 0;
        if !stable {
            self.apply_window_with_to(other, out, window, |va, vb, va_rm, vb_rm| {
                if va.notnan() && vb.notnan() {
                    n += 1;
                    let (va, vb) = (va.f64(), vb.f64());
                    sum_a += va;
                    sum2_a += va * va;
                    sum_b += vb;
                    sum2_b += vb * vb;
                    sum_ab += va * vb;
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let mean_a = sum_a / n_f64;
                    let mut var_a = sum2_a / n_f64;
                    let mean_b = sum_b / n_f64;
                    let mut var_b = sum2_b / n_f64;
                    var_a -= mean_a.powi(2);
                    var_b -= mean_b.powi(2);
                    if (var_a > 1e-14) & (var_b > 1e-14) {
                        let exy = sum_ab / n_f64;
                        let exey = sum_a * sum_b / n_f64.powi(2);
                        (exy - exey) / (var_a * var_b).sqrt()
                    } else {
                        f64::NAN
                    }
                } else {
                    f64::NAN
                };
                if let (Some(va), Some(vb)) = (va_rm, vb_rm) {
                    if va.notnan() && vb.notnan() {
                        n -= 1;
                        let (va, vb) = (va.f64(), vb.f64());
                        sum_a -= va;
                        sum2_a -= va * va;
                        sum_b -= vb;
                        sum2_b -= vb * vb;
                        sum_ab -= va * vb;
                    };
                }
                res
            });
        } else {
            define_c!(c1, c2, c3, c4, c5, c6, c7, c8, c9, c10);
            self.stable_apply_window_with_to(other, out, window, |va, vb, va_rm, vb_rm| {
                if va.notnan() && vb.notnan() {
                    n += 1;
                    sum_a.kh_sum(va, c1);
                    sum2_a.kh_sum(va * va, c2);
                    sum_b.kh_sum(vb, c3);
                    sum2_b.kh_sum(vb * vb, c4);
                    sum_ab.kh_sum(va * vb, c5);
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let mean_a = sum_a / n_f64;
                    let mut var_a = sum2_a / n_f64;
                    let mean_b = sum_b / n_f64;
                    let mut var_b = sum2_b / n_f64;
                    var_a -= mean_a.powi(2);
                    var_b -= mean_b.powi(2);
                    if (var_a > 1e-14) & (var_b > 1e-14) {
                        let exy = sum_ab / n_f64;
                        let exey = sum_a * sum_b / n_f64.powi(2);
                        (exy - exey) / (var_a * var_b).sqrt()
                    } else {
                        f64::NAN
                    }
                } else {
                    f64::NAN
                };
                if va_rm.notnan() && vb_rm.notnan() {
                    n -= 1;
                    sum_a.kh_sum(-va_rm, c6);
                    sum2_a.kh_sum(-va_rm * va_rm, c7);
                    sum_b.kh_sum(-vb_rm, c8);
                    sum2_b.kh_sum(-vb_rm * vb_rm, c9);
                    sum_ab.kh_sum(-va_rm * vb_rm, c10);
                };
                res
            })
        }
    }
);
