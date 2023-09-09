use super::super::export::*;

impl_map_nd!(
    ts_stable,
    pub fn ts_stable_1d<S2>(
        &self,
        out: &mut ArrBase<S2, D>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    {where T: Number}
    {
        let window = min(self.len(), window);
        if (window < min_periods) | (window == 1) {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut sum = 0.;
        let mut sum2 = 0.;
        let mut n = 0;
        if !stable {
            self.apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    sum += v;
                    sum2 += v * v
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let mut var = sum2 / n_f64;
                    let mean = sum / n_f64;
                    var -= mean.powi(2);
                    if var > 1e-14 {
                        mean / (var * n_f64 / (n - 1).f64()).sqrt()
                    } else {
                        f64::NAN
                    }
                } else {
                    f64::NAN
                };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        let v = v.f64();
                        n -= 1;
                        sum -= v;
                        sum2 -= v * v
                    };
                }
                res
            });
        } else {
            define_c!(c1, c2, c3, c4);
            self.apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    sum.kh_sum(v, c1);
                    sum2.kh_sum(v * v, c2);
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let mut var = sum2 / n_f64;
                    let mean = sum / n_f64;
                    var -= mean.powi(2);
                    if var > 1e-14 {
                        mean / (var * n_f64 / (n - 1).f64()).sqrt()
                    } else {
                        f64::NAN
                    }
                } else {
                    f64::NAN
                };
                if let Some(v_rm) = v_rm {
                    if v_rm.notnan() {
                        n -= 1;
                        let v_rm = v_rm.f64();
                        sum.kh_sum(-v_rm, c3);
                        sum2.kh_sum(-v_rm * v_rm, c4);
                    };
                }
                res
            })
        }
    }
);

impl_map_nd!(
    ts_meanstdnorm,
    pub fn ts_meanstdnorm_1d<S2>(
        &self,
        out: &mut ArrBase<S2, D>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    {where T: Number}
    {
        let window = min(self.len(), window);
        if (window < min_periods) | (window == 1) {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut sum = 0.;
        let mut sum2 = 0.;
        let mut n = 0;
        if !stable {
            self.apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    sum += v;
                    sum2 += v * v
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let mut var = sum2 / n_f64;
                    let mean = sum / n_f64;
                    var -= mean.powi(2);
                    if var > 1e-14 {
                        (v.f64() - mean) / (var * n_f64 / (n - 1).f64()).sqrt()
                    } else {
                        f64::NAN
                    }
                } else {
                    f64::NAN
                };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        let v = v.f64();
                        n -= 1;
                        sum -= v;
                        sum2 -= v * v
                    };
                }
                res
            });
        } else {
            define_c!(c1, c2, c3, c4);
            self.apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    sum.kh_sum(v, c1);
                    sum2.kh_sum(v * v, c2);
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let mut var = sum2 / n_f64;
                    let mean = sum / n_f64;
                    var -= mean.powi(2);
                    if var > 1e-14 {
                        (v.f64() - mean) / (var * n_f64 / (n - 1).f64()).sqrt()
                    } else {
                        f64::NAN
                    }
                } else {
                    f64::NAN
                };
                if let Some(v_rm) = v_rm {
                    if v_rm.notnan() {
                        n -= 1;
                        let v_rm = v_rm.f64();
                        sum.kh_sum(-v_rm, c3);
                        sum2.kh_sum(-v_rm * v_rm, c4);
                    };
                }
                res
            })
        }
    }
);

impl_map_nd!(
    ts_minmaxnorm,
    pub fn ts_minmaxnorm_1d<S2>(
        &self,
        out: &mut ArrBase<S2, D>,
        window: usize,
        min_periods: usize,
    ) -> f64
    {where T: Number}
    {
        let window = min(self.len(), window);
        if (window < min_periods) | (window == 1) {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {v.write(f64::NAN);});
        }
        let mut max: T = T::min_();
        let mut max_idx = 0;
        let mut min: T = T::max_();
        let mut min_idx = 0;
        let mut n = 0;
        for i in 0..window - 1 {
            // 安全性：i不会超过self和out的长度
            let v = unsafe { *self.uget(i) };
            if v.notnan() {
                n += 1
            };
            if v >= max {
                (max, max_idx) = (v, i);
            }
            if v <= min {
                (min, min_idx) = (v, i);
            }
            let out = unsafe { out.uget_mut(i) };
            if (n >= min_periods) & (max != min) {
                out.write((v - min).f64() / (max - min).f64());
            } else {
                out.write(f64::NAN);
            };
        }
        for (start, end) in (window - 1..self.len()).enumerate() {
            // 安全性：start和end不会超过self和out的长度
            unsafe {
                let v = *self.uget(end);
                if v.notnan() {
                    n += 1
                };
                match (max_idx < start, min_idx < start) {
                    (true, false) => {
                        // 最大值已经失效，重新找最大值
                        max = T::min_();
                        for i in start..end {
                            let v = *self.uget(i);
                            if v >= max {
                                (max, max_idx) = (v, i);
                            }
                        }
                    }
                    (false, true) => {
                        // 最小值已经失效，重新找最小值
                        min = T::max_();
                        for i in start..end {
                            let v = *self.uget(i);
                            if v <= min {
                                (min, min_idx) = (v, i);
                            }
                        }
                    }
                    (true, true) => {
                        // 最大和最小值都已经失效，重新找最大和最小值
                        (max, min) = (T::min_(), T::max_());
                        for i in start..end {
                            let v = *self.uget(i);
                            if v >= max {
                                (max, max_idx) = (v, i);
                            }
                            if v <= min {
                                (min, min_idx) = (v, i);
                            }
                        }
                    }
                    (false, false) => (), // 不需要重新找最大和最小值
                }
                // 检查end位置是否是最大或最小值
                if v >= max {
                    (max, max_idx) = (v, end);
                }
                if v <= min {
                    (min, min_idx) = (v, end);
                }
                let out = out.uget_mut(end);
                if (n >= min_periods) & (max != min) {
                    out.write((v - min).f64() / (max - min).f64());
                } else {
                    out.write(f64::NAN);
                };
                if self.uget(start).notnan() {
                    n -= 1
                };
            }
        }
    }
);
