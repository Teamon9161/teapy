use super::super::export::*;

impl_map_nd!(
    ts_reg,
    pub fn ts_reg_1d<S2>(
        &self,
        out: &mut ArrBase<S2, D>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    {where T: Number}
    {
        // 错位相减核心公式：sum_xt(t) = sum_xt(t-1) - sum_x + n * new_element
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut sum = 0.;
        let mut sum_xt = 0.;
        let mut n = 0;
        if !stable {
            self.apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    sum_xt += n.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum += v;
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_add_n = n.mul_add(n, n);
                    let sum_t = (nn_add_n >> 1).f64(); // sum of time from 1 to window
                                                        // denominator of slope
                    let divisor = (n * nn_add_n * n.mul_add(2, 1)).f64() / 6. - sum_t.powi(2);
                    let slope = (n_f64 * sum_xt - sum_t * sum) / divisor;
                    let intercept = sum_t.mul_add(-slope, sum) / n_f64;
                    slope.mul_add(n_f64, intercept)
                } else {
                    f64::NAN
                };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        n -= 1;
                        sum_xt -= sum;
                        sum -= v.f64();
                    };
                }
                res
            });
        } else {
            define_c!(c1, c2, c3, c4);
            self.apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    sum_xt.kh_sum(v.f64() * n.f64(), c1); // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum.kh_sum(v.f64(), c2);
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_add_n = n.mul_add(n, n);
                    let sum_t = (nn_add_n >> 1).f64(); // sum of time from 1 to window
                                                        // denominator of slope
                    let divisor = (n * nn_add_n * n.mul_add(2, 1)).f64() / 6. - sum_t.powi(2);
                    let slope = (n_f64 * sum_xt - sum_t * sum) / divisor;
                    let intercept = sum_t.mul_add(-slope, sum) / n_f64;
                    slope.mul_add(n_f64, intercept)
                } else {
                    f64::NAN
                };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        n -= 1;
                        sum_xt.kh_sum(-sum, c3); // 错位相减法, 忽略nan带来的系数和window不一致问题
                        sum.kh_sum(-v.f64(), c4);
                    };
                }
                res
            })
        }
    }
);

impl_map_nd!(
    ts_tsf,
    pub fn ts_tsf_1d<S2>(
        &self,
        out: &mut ArrBase<S2, D>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    {where T: Number}
    {
        // 错位相减核心公式：sum_xt(t) = sum_xt(t-1) - sum_x + n * new_element
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut sum = 0.;
        let mut sum_xt = 0.;
        let mut n = 0;
        if !stable {
            self.apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    sum_xt += n.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum += v;
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_add_n = n.mul_add(n, n);
                    let sum_t = (nn_add_n >> 1).f64(); // sum of time from 1 to window
                                                        // denominator of slope
                    let divisor = (n * nn_add_n * n.mul_add(2, 1)).f64() / 6. - sum_t.powi(2);
                    let slope = (n_f64 * sum_xt - sum_t * sum) / divisor;
                    let intercept = sum_t.mul_add(-slope, sum) / n_f64;
                    slope.mul_add((n+1).f64(), intercept)
                } else {
                    f64::NAN
                };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        n -= 1;
                        sum_xt -= sum;
                        sum -= v.f64();
                    };
                }
                res
            });
        } else {
            define_c!(c1, c2, c3, c4);
            self.apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    sum_xt.kh_sum(v.f64() * n.f64(), c1); // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum.kh_sum(v.f64(), c2);
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_add_n = n.mul_add(n, n);
                    let sum_t = (nn_add_n >> 1).f64(); // sum of time from 1 to window
                                                        // denominator of slope
                    let divisor = (n * nn_add_n * n.mul_add(2, 1)).f64() / 6. - sum_t.powi(2);
                    let slope = (n_f64 * sum_xt - sum_t * sum) / divisor;
                    let intercept = sum_t.mul_add(-slope, sum) / n_f64;
                    slope.mul_add((n+1).f64(), intercept)
                } else {
                    f64::NAN
                };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        n -= 1;
                        sum_xt.kh_sum(-sum, c3); // 错位相减法, 忽略nan带来的系数和window不一致问题
                        sum.kh_sum(-v.f64(), c4);
                    };
                }
                res
            })
        }
    }
);

impl_map_nd!(
    ts_reg_slope,
    pub fn ts_reg_slope_1d<S2>(
        &self,
        out: &mut ArrBase<S2, D>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    {where T: Number}
    {
        // 错位相减核心公式：sum_xt(t) = sum_xt(t-1) - sum_x + n * new_element
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut sum = 0.;
        let mut sum_xt = 0.;
        let mut n = 0;
        if !stable {
            self.apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    sum_xt += n.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum += v;
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_add_n = n.mul_add(n, n);
                    let sum_t = (nn_add_n >> 1).f64(); // sum of time from 1 to window
                                                        // denominator of slope
                    let divisor = (n * nn_add_n * n.mul_add(2, 1)).f64() / 6. - sum_t.powi(2);
                    (n_f64 * sum_xt - sum_t * sum) / divisor
                } else {
                    f64::NAN
                };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        n -= 1;
                        sum_xt -= sum;
                        sum -= v.f64();
                    };
                }
                res
            });
        } else {
            define_c!(c1, c2, c3, c4);
            self.apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    sum_xt.kh_sum(v.f64() * n.f64(), c1); // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum.kh_sum(v.f64(), c2);
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_add_n = n.mul_add(n, n);
                    let sum_t = (nn_add_n >> 1).f64(); // sum of time from 1 to window
                                                        // denominator of slope
                    let divisor = (n * nn_add_n * n.mul_add(2, 1)).f64() / 6. - sum_t.powi(2);
                    (n_f64 * sum_xt - sum_t * sum) / divisor
                } else {
                    f64::NAN
                };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        n -= 1;
                        sum_xt.kh_sum(-sum, c3); // 错位相减法, 忽略nan带来的系数和window不一致问题
                        sum.kh_sum(-v.f64(), c4);
                    };
                }
                res
            })
        }
    }
);

impl_map_nd!(
    ts_reg_intercept,
    pub fn ts_reg_intercept_1d<S2>(
        &self,
        out: &mut ArrBase<S2, D>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    {where T: Number}
    {
        // 错位相减核心公式：sum_xt(t) = sum_xt(t-1) - sum_x + n * new_element
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut sum = 0.;
        let mut sum_xt = 0.;
        let mut n = 0;
        if !stable {
            self.apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    sum_xt += n.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum += v;
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_add_n = n.mul_add(n, n);
                    let sum_t = (nn_add_n >> 1).f64(); // sum of time from 1 to window
                                                        // denominator of slope
                    let divisor = (n * nn_add_n * n.mul_add(2, 1)).f64() / 6. - sum_t.powi(2);
                    let slope = (n_f64 * sum_xt - sum_t * sum) / divisor;
                    sum_t.mul_add(-slope, sum) / n_f64
                } else {
                    f64::NAN
                };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        n -= 1;
                        sum_xt -= sum;
                        sum -= v.f64();
                    };
                }
                res
            });
        } else {
            define_c!(c1, c2, c3, c4);
            self.apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    sum_xt.kh_sum(v.f64() * n.f64(), c1); // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum.kh_sum(v.f64(), c2);
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_add_n = n.mul_add(n, n);
                    let sum_t = (nn_add_n >> 1).f64(); // sum of time from 1 to window
                                                        // denominator of slope
                    let divisor = (n * nn_add_n * n.mul_add(2, 1)).f64() / 6. - sum_t.powi(2);
                    let slope = (n_f64 * sum_xt - sum_t * sum) / divisor;
                    sum_t.mul_add(-slope, sum) / n_f64
                } else {
                    f64::NAN
                };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        n -= 1;
                        sum_xt.kh_sum(-sum, c3); // 错位相减法, 忽略nan带来的系数和window不一致问题
                        sum.kh_sum(-v.f64(), c4);
                    };
                }
                res
            })
        }
    }
);

impl_map2_nd!(
    ts_regx_beta,
    pub fn ts_regx_beta_1d<S2, T2, S3>(
        &self,
        x: &ArrBase<S2, D>,
        out: &mut ArrBase<S3, D>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    {where T: Number, T2: Number,}
    {
        let window = min(self.len(), window);
        let window = min(x.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut sum_a = 0.;
        let mut sum_b = 0.;
        let mut sum_b2 = 0.;
        let mut sum_ab = 0.;
        let mut n = 0;
        if !stable {
            self.apply_window_with_to(x, out, window, |va, vb, va_rm, vb_rm| {
                if va.notnan() && vb.notnan() {
                    n += 1;
                    let (va, vb) = (va.f64(), vb.f64());
                    sum_a += va;
                    sum_b += vb;
                    sum_b2 += vb.powi(2);
                    sum_ab += va * vb;
                };
                let res = if n >= min_periods {
                    (n.f64() * sum_ab - sum_a * sum_b) / (n.f64() * sum_b2 - sum_b.powi(2))
                } else {
                    f64::NAN
                };
                if let (Some(va), Some(vb)) = (va_rm, vb_rm) {
                    if va.notnan() && vb.notnan() {
                        n -= 1;
                        let (va, vb) = (va.f64(), vb.f64());
                        sum_a -= va;
                        sum_b -= vb;
                        sum_b2 -= vb.powi(2);
                        sum_ab -= va * vb;
                    };
                }
                res
            });
        } else {
            define_c!(c1, c2, c3, c4, c5, c6, c7, c8);
            self.stable_apply_window_with_to(x, out, window, |va, vb, va_rm, vb_rm| {
                if va.notnan() && vb.notnan() {
                    n += 1;
                    sum_a.kh_sum(va, c1);
                    sum_b.kh_sum(vb, c2);
                    sum_ab.kh_sum(va * vb, c3);
                    sum_b2.kh_sum(vb.powi(2), c7);
                };
                let res = if n >= min_periods {
                    (n.f64() * sum_ab - sum_a * sum_b) / (n.f64() * sum_b2 - sum_b.powi(2))
                } else {
                    f64::NAN
                };
                if va_rm.notnan() && vb_rm.notnan() {
                    n -= 1;
                    sum_a.kh_sum(-va_rm, c4);
                    sum_b.kh_sum(-vb_rm, c5);
                    sum_ab.kh_sum(-va_rm * vb_rm, c6);
                    sum_b2.kh_sum(-vb.powi(2), c8);
                };
                res
            })
        }
    }
);

impl_map2_nd!(
    ts_regx_alpha,
    pub fn ts_regx_alpha_1d<S2, T2, S3>(
        &self,
        x: &ArrBase<S2, D>,
        out: &mut ArrBase<S3, D>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    {where T: Number, T2: Number,}
    {
        let window = min(self.len(), window);
        let window = min(x.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut sum_a = 0.;
        let mut sum_b = 0.;
        let mut sum_b2 = 0.;
        let mut sum_ab = 0.;
        let mut n = 0;
        if !stable {
            self.apply_window_with_to(x, out, window, |va, vb, va_rm, vb_rm| {
                if va.notnan() && vb.notnan() {
                    n += 1;
                    let (va, vb) = (va.f64(), vb.f64());
                    sum_a += va;
                    sum_b += vb;
                    sum_b2 += vb.powi(2);
                    sum_ab += va * vb;
                };
                let res = if n >= min_periods {
                    let beta = (n.f64() * sum_ab - sum_a * sum_b) / (n.f64() * sum_b2 - sum_b.powi(2));
                    (sum_a - beta * sum_b) / n.f64()
                } else {
                    f64::NAN
                };
                if let (Some(va), Some(vb)) = (va_rm, vb_rm) {
                    if va.notnan() && vb.notnan() {
                        n -= 1;
                        let (va, vb) = (va.f64(), vb.f64());
                        sum_a -= va;
                        sum_b -= vb;
                        sum_b2 -= vb.powi(2);
                        sum_ab -= va * vb;
                    };
                }
                res
            });
        } else {
            define_c!(c1, c2, c3, c4, c5, c6, c7, c8);
            self.stable_apply_window_with_to(x, out, window, |va, vb, va_rm, vb_rm| {
                if va.notnan() && vb.notnan() {
                    n += 1;
                    sum_a.kh_sum(va, c1);
                    sum_b.kh_sum(vb, c2);
                    sum_ab.kh_sum(va * vb, c3);
                    sum_b2.kh_sum(vb.powi(2), c7);
                };
                let res = if n >= min_periods {
                    let beta = (n.f64() * sum_ab - sum_a * sum_b) / (n.f64() * sum_b2 - sum_b.powi(2));
                    (sum_a - beta * sum_b) / n.f64()
                } else {
                    f64::NAN
                };
                if va_rm.notnan() && vb_rm.notnan() {
                    n -= 1;
                    sum_a.kh_sum(-va_rm, c4);
                    sum_b.kh_sum(-vb_rm, c5);
                    sum_ab.kh_sum(-va_rm * vb_rm, c6);
                    sum_b2.kh_sum(-vb.powi(2), c8);
                };
                res
            })
        }
    }
);

impl_map2_nd!(
    ts_regx_resid_std,
    pub fn ts_regx_resid_std_1d<S2, T2, S3>(
        &self,
        x: &ArrBase<S2, D>,
        out: &mut ArrBase<S3, D>,
        window: usize,
        min_periods: usize,
    ) -> f64
    {where T: Number, T2: Number,}
    {
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut sum_a = 0.;
        let mut sum_b = 0.;
        let mut sum_b2 = 0.;
        let mut sum_ab = 0.;
        let mut n = 0;
        for i in 0..window - 1 {
            // safety：i is inbound
            let (va, vb) = unsafe{(*self.uget(i), *x.uget(i))};
            if va.notnan() && vb.notnan() {
                n += 1;
                let (va, vb) = (va.f64(), vb.f64());
                sum_a += va;
                sum_b += vb;
                sum_b2 += vb.powi(2);
                sum_ab += va * vb;
            };
            if n >= min_periods {
                let beta = (n.f64() * sum_ab - sum_a * sum_b) / (n.f64() * sum_b2 - sum_b.powi(2));
                let alpha = (sum_a - beta * sum_b) / n.f64();
                let resid = (0..=i).map(|j| {
                    let (vy, vx) = unsafe{(*self.uget(j), *x.uget(j))};
                    vy.f64() - alpha - beta * vx.f64()
                }).collect_trusted();
                let std = Arr1::from_vec(resid).std_1d(false);
                unsafe{out.uget_mut(i).write(std)};
            } else {
                unsafe{out.uget_mut(i).write(f64::NAN)};
            };
        }
        for (start, end) in (window - 1..self.len()).enumerate() {
            // safety：start, end is inbound
            let (va, vb) = unsafe{(*self.uget(end), *x.uget(end))};
            if va.notnan() && vb.notnan() {
                n += 1;
                let (va, vb) = (va.f64(), vb.f64());
                sum_a += va;
                sum_b += vb;
                sum_b2 += vb.powi(2);
                sum_ab += va * vb;
            };
            if n >= min_periods {
                let beta = (n.f64() * sum_ab - sum_a * sum_b) / (n.f64() * sum_b2 - sum_b.powi(2));
                let alpha = (sum_a - beta * sum_b) / n.f64();
                let resid = (start..=end).map(|j| {
                    let (vy, vx) = unsafe{(*self.uget(j), *x.uget(j))};
                    vy.f64() - alpha - beta * vx.f64()
                }).collect_trusted();
                let std = Arr1::from_vec(resid).std_1d(false);
                unsafe{out.uget_mut(end).write(std)};
            } else {
                unsafe{out.uget_mut(end).write(f64::NAN)};
            };
            let (va, vb) = unsafe{(*self.uget(start), *x.uget(start))};
            if va.notnan() && vb.notnan() {
                n -= 1;
                let (va, vb) = (va.f64(), vb.f64());
                sum_a -= va;
                sum_b -= vb;
                sum_b2 -= vb.powi(2);
                sum_ab -= va * vb;
            }
        }
    }
);

impl_map2_nd!(
    ts_regx_resid_skew,
    pub fn ts_regx_resid_skew_1d<S2, T2, S3>(
        &self,
        x: &ArrBase<S2, D>,
        out: &mut ArrBase<S3, D>,
        window: usize,
        min_periods: usize,
    ) -> f64
    {where T: Number, T2: Number,}
    {
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut sum_a = 0.;
        let mut sum_b = 0.;
        let mut sum_b2 = 0.;
        let mut sum_ab = 0.;
        let mut n = 0;
        for i in 0..window - 1 {
            // safety：i is inbound
            let (va, vb) = unsafe{(*self.uget(i), *x.uget(i))};
            if va.notnan() && vb.notnan() {
                n += 1;
                let (va, vb) = (va.f64(), vb.f64());
                sum_a += va;
                sum_b += vb;
                sum_b2 += vb.powi(2);
                sum_ab += va * vb;
            };
            if n >= min_periods {
                let beta = (n.f64() * sum_ab - sum_a * sum_b) / (n.f64() * sum_b2 - sum_b.powi(2));
                let alpha = (sum_a - beta * sum_b) / n.f64();
                let resid = (0..=i).map(|j| {
                    let (vy, vx) = unsafe{(*self.uget(j), *x.uget(j))};
                    vy.f64() - alpha - beta * vx.f64()
                }).collect_trusted();
                let std = Arr1::from_vec(resid).skew_1d(false);
                unsafe{out.uget_mut(i).write(std)};
            } else {
                unsafe{out.uget_mut(i).write(f64::NAN)};
            };
        }
        for (start, end) in (window - 1..self.len()).enumerate() {
            // safety：start, end is inbound
            let (va, vb) = unsafe{(*self.uget(end), *x.uget(end))};
            if va.notnan() && vb.notnan() {
                n += 1;
                let (va, vb) = (va.f64(), vb.f64());
                sum_a += va;
                sum_b += vb;
                sum_b2 += vb.powi(2);
                sum_ab += va * vb;
            };
            if n >= min_periods {
                let beta = (n.f64() * sum_ab - sum_a * sum_b) / (n.f64() * sum_b2 - sum_b.powi(2));
                let alpha = (sum_a - beta * sum_b) / n.f64();
                let resid = (start..=end).map(|j| {
                    let (vy, vx) = unsafe{(*self.uget(j), *x.uget(j))};
                    vy.f64() - alpha - beta * vx.f64()
                }).collect_trusted();
                let std = Arr1::from_vec(resid).skew_1d(false);
                unsafe{out.uget_mut(end).write(std)};
            } else {
                unsafe{out.uget_mut(end).write(f64::NAN)};
            };
            let (va, vb) = unsafe{(*self.uget(start), *x.uget(start))};
            if va.notnan() && vb.notnan() {
                n -= 1;
                let (va, vb) = (va.f64(), vb.f64());
                sum_a -= va;
                sum_b -= vb;
                sum_b2 -= vb.powi(2);
                sum_ab -= va * vb;
            }
        }
    }
);
