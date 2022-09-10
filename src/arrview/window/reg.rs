use super::super::prelude::*;

impl_arrview!([ArrView1, ArrViewMut1], Number, {
    pub fn ts_reg(
        &self,
        out: &mut ArrViewMut1<f64>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) where
        usize: Number,
    {
        // 错位相减核心公式：sum_xt(t) = sum_xt(t-1) - sum_x + n * new_element
        assert!(
            window >= min_periods,
            "window must be greater than min_periods"
        );
        if window == 1 {
            out.apply_mut(|v| *v = f64::NAN);
            return;
        } // 如果滚动窗口是1则返回全nan
        let mut sum = 0.;
        let mut sum_xt = 0.;
        let mut n = 0;
        if !stable {
            self.apply_valid_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    sum_xt += n.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum += v;
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_mul_n = n * (n + 1);
                    let sum_t = (nn_mul_n >> 1).f64(); // sum of time from 1 to window
                                                       // denominator of slope
                    let divisor = (n * (nn_mul_n * (2 * n + 1))).f64() / 6. - sum_t.powi(2);
                    let slope = (n_f64 * sum_xt - sum_t * sum) / divisor;
                    let intercept = (sum - slope * sum_t) / n_f64;
                    intercept + slope * n_f64
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
            self.apply_valid_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    sum_xt.kh_sum(v.f64() * n.f64(), c1); // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum.kh_sum(v.f64(), c2);
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_mul_n = n * (n + 1);
                    let sum_t = (nn_mul_n >> 1).f64(); // sum of time from 1 to window
                                                       // denominator of slope
                    let divisor = (n * (nn_mul_n * (2 * n + 1))).f64() / 6. - sum_t.powi(2);
                    let slope = (n_f64 * sum_xt - sum_t * sum) / divisor;
                    let intercept = (sum - slope * sum_t) / n_f64;
                    intercept + slope * n_f64
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

    pub fn ts_tsf(
        &self,
        out: &mut ArrViewMut1<f64>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) where
        usize: Number,
    {
        // 错位相减核心公式：sum_xt(t) = sum_xt(t-1) - sum_x + n * new_element
        assert!(
            window >= min_periods,
            "window must be greater than min_periods"
        );
        if window == 1 {
            out.apply_mut(|v| *v = f64::NAN);
            return;
        } // 如果滚动窗口是1则返回全nan
        let mut sum = 0.;
        let mut sum_xt = 0.;
        let mut n = 0;
        if !stable {
            self.apply_valid_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    sum_xt += n.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum += v;
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_mul_n = n * (n + 1);
                    let sum_t = (nn_mul_n >> 1).f64(); // sum of time from 1 to window
                                                       // denominator of slope
                    let divisor = (n * (nn_mul_n * (2 * n + 1))).f64() / 6. - sum_t.powi(2);
                    let slope = (n_f64 * sum_xt - sum_t * sum) / divisor;
                    let intercept = (sum - slope * sum_t) / n_f64;
                    intercept + slope * (n + 1).f64()
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
            self.apply_valid_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    sum_xt.kh_sum(v.f64() * n.f64(), c1); // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum.kh_sum(v.f64(), c2);
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_mul_n = n * (n + 1);
                    let sum_t = (nn_mul_n >> 1).f64(); // sum of time from 1 to window
                                                       // denominator of slope
                    let divisor = (n * (nn_mul_n * (2 * n + 1))).f64() / 6. - sum_t.powi(2);
                    let slope = (n_f64 * sum_xt - sum_t * sum) / divisor;
                    let intercept = (sum - slope * sum_t) / n_f64;
                    intercept + slope * (n + 1).f64()
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

    pub fn ts_reg_slope(
        &self,
        out: &mut ArrViewMut1<f64>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) where
        usize: Number,
    {
        // 错位相减核心公式：sum_xt(t) = sum_xt(t-1) - sum_x + n * new_element
        assert!(
            window >= min_periods,
            "window must be greater than min_periods"
        );
        if window == 1 {
            out.apply_mut(|v| *v = f64::NAN);
            return;
        } // 如果滚动窗口是1则返回全nan
        let mut sum = 0.;
        let mut sum_xt = 0.;
        let mut n = 0;
        if !stable {
            self.apply_valid_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    sum_xt += n.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum += v;
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_mul_n = n * (n + 1);
                    let sum_t = (nn_mul_n >> 1).f64(); // sum of time from 1 to window
                                                       // denominator of slope
                    let divisor = (n * (nn_mul_n * (2 * n + 1))).f64() / 6. - sum_t.powi(2);
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
            self.apply_valid_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    sum_xt.kh_sum(v.f64() * n.f64(), c1); // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum.kh_sum(v.f64(), c2);
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_mul_n = n * (n + 1);
                    let sum_t = (nn_mul_n >> 1).f64(); // sum of time from 1 to window
                                                       // denominator of slope
                    let divisor = (n * (nn_mul_n * (2 * n + 1))).f64() / 6. - sum_t.powi(2);
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

    pub fn ts_reg_intercept(
        &self,
        out: &mut ArrViewMut1<f64>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) where
        usize: Number,
    {
        // 错位相减核心公式：sum_xt(t) = sum_xt(t-1) - sum_x + n * new_element
        assert!(
            window >= min_periods,
            "window must be greater than min_periods"
        );
        if window == 1 {
            out.apply_mut(|v| *v = f64::NAN);
            return;
        } // 如果滚动窗口是1则返回全nan
        let mut sum = 0.;
        let mut sum_xt = 0.;
        let mut n = 0;
        if !stable {
            self.apply_valid_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    sum_xt += n.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum += v;
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_mul_n = n * (n + 1);
                    let sum_t = (nn_mul_n >> 1).f64(); // sum of time from 1 to window
                                                       // denominator of slope
                    let divisor = (n * (nn_mul_n * (2 * n + 1))).f64() / 6. - sum_t.powi(2);
                    let slope = (n_f64 * sum_xt - sum_t * sum) / divisor;
                    (sum - slope * sum_t) / n_f64
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
            self.apply_valid_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    sum_xt.kh_sum(v.f64() * n.f64(), c1); // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum.kh_sum(v.f64(), c2);
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let nn_mul_n = n * (n + 1);
                    let sum_t = (nn_mul_n >> 1).f64(); // sum of time from 1 to window
                                                       // denominator of slope
                    let divisor = (n * (nn_mul_n * (2 * n + 1))).f64() / 6. - sum_t.powi(2);
                    let slope = (n_f64 * sum_xt - sum_t * sum) / divisor;
                    (sum - slope * sum_t) / n_f64
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
});
