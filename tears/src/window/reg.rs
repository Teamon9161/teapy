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
