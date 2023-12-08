#[cfg(feature = "lazy")]
use lazy::Expr;
use ndarray::{Data, DataMut, Dimension, Ix1, ShapeBuilder};
use std::cmp::min;
use std::mem::MaybeUninit;
use tea_core::prelude::*;
use tea_core::utils::{define_c, kh_sum};

#[arr_map_ext(lazy = "view", type = "numeric")]
impl<T: Send + Sync, S: Data<Elem = T>, D: Dimension> FeatureTs for ArrBase<S, D> {
    fn ts_sum<SO>(
        &self,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T: Number,
    {
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut sum = 0.;
        let mut n = 0;
        if !stable {
            self.as_dim1().apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    sum += v.f64();
                };
                let res = if n >= min_periods { sum } else { f64::NAN };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        n -= 1;
                        sum -= v.f64();
                    };
                }
                res
            });
        } else {
            define_c!(c, c1);
            self.as_dim1().apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    sum.kh_sum(v.f64(), c);
                };
                let res = if n >= min_periods { sum } else { f64::NAN };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        n -= 1;
                        sum.kh_sum(-v.f64(), c1);
                    };
                }
                res
            });
        }
    }

    fn ts_sma<SO>(
        &self,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T: Number,
    {
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut sum = 0.;
        let mut n = 0;
        if !stable {
            self.as_dim1().apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    sum += v.f64();
                };
                let res = if n >= min_periods {
                    sum / n.f64()
                } else {
                    f64::NAN
                };
                if let Some(v_rm) = v_rm {
                    if v_rm.notnan() {
                        n -= 1;
                        sum -= v_rm.f64();
                    };
                }
                res
            });
        } else {
            define_c!(c, c1);
            self.as_dim1().apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    sum.kh_sum(v.f64(), c);
                };
                let res = if n >= min_periods {
                    sum / n.f64()
                } else {
                    f64::NAN
                };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        n -= 1;
                        sum.kh_sum(-v.f64(), c1);
                    };
                }
                res
            });
        }
    }

    fn ts_ewm<SO>(
        &self,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T: Number,
    {
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        // 错位相减核心公式：
        // q_x(t) = 1 * new_element - alpha(q_x(t-1 without 1st element)) - 1st element * oma ^ (n-1)
        let mut q_x = 0.; // 权重的分子部分 * 元素，使用错位相减法来计算
        let alpha = 2. / window.f64();
        let oma = 1. - alpha; // one minus alpha
        let mut n = 0;
        if !stable {
            self.as_dim1().apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    q_x += v.f64() - alpha * q_x.f64();
                };
                let res = if n >= min_periods {
                    q_x.f64() * alpha / (1. - oma.powi(n as i32))
                } else {
                    f64::NAN
                };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        n -= 1;
                        // 本应是window-1，不过本身window就要自然减一，调整一下顺序
                        q_x -= v.f64() * oma.powi(n as i32);
                    };
                }
                res
            });
        } else {
            define_c!(c1, c2);
            self.as_dim1().apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    q_x.kh_sum(v.f64() - alpha * q_x, c1);
                };
                let res = if n >= min_periods {
                    q_x.f64() * alpha / (1. - oma.powi(n as i32))
                } else {
                    f64::NAN
                };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        n -= 1;
                        q_x.kh_sum(-v.f64() * oma.powi(n as i32), c2);
                    };
                }
                res
            })
        }
    }

    fn ts_wma<SO>(
        &self,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T: Number,
    {
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut sum = 0.;
        let mut sum_xt = 0.;
        let mut n = 0;
        if !stable {
            self.as_dim1().apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    sum_xt += n.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum += v;
                };
                let res = if n >= min_periods {
                    let divisor = (n * (n + 1)) >> 1;
                    sum_xt / divisor.f64()
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
            self.as_dim1().apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    sum_xt.kh_sum(v.f64() * n.f64(), c1); // 错位相减法, 忽略nan带来的系数和window不一致问题
                    sum.kh_sum(v.f64(), c2);
                };
                let res = if n >= min_periods {
                    let divisor = (n * (n + 1)) >> 1;
                    sum_xt / divisor.f64()
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

    fn ts_std<SO>(
        &self,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T: Number,
    {
        let window = min(self.len(), window);
        if (window < min_periods) | (window == 1) {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }

        let window = min(self.len(), window);
        let mut sum = 0.;
        let mut sum2 = 0.;
        let mut n = 0;
        if !stable {
            self.as_dim1().apply_window_to(out, window, |v, v_rm| {
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
                    // var肯定大于等于0，否则只能是精度问题
                    if var > 1e-14 {
                        (var * n_f64 / (n - 1).f64()).sqrt()
                    } else {
                        0.
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
            let mut mean = 0.;
            // Welford's method for the online variance-calculation
            // using Kahan summation
            // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            self.as_dim1()
                .stable_apply_window_to(out, window, |v, v_rm| {
                    if v.notnan() {
                        n += 1;
                        let delta = kh_sum(v, -mean, c1);
                        mean += delta / n.f64();
                        let delta2 = kh_sum(v, -mean, c3);
                        sum2 += delta * delta2;
                    };
                    let res = if n >= min_periods {
                        let n_f64 = n.f64();
                        let var = sum2 / n_f64;
                        // var肯定大于等于0，否则只能是精度问题
                        if var > 1e-14 {
                            (var * n_f64 / (n - 1).f64()).sqrt()
                        } else {
                            0.
                        }
                    } else {
                        f64::NAN
                    };
                    if v_rm.notnan() {
                        n -= 1;
                        if n > 0 {
                            let delta = kh_sum(v_rm, -mean, c2);
                            mean -= delta / n.f64();
                            let delta2 = kh_sum(v_rm, -mean, c4);
                            sum2 -= delta * delta2;
                        }
                    };
                    res
                })
        }
    }

    fn ts_var<SO>(
        &self,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T: Number,
    {
        let window = min(self.len(), window);
        if (window < min_periods) | (window == 1) {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut sum = 0.;
        let mut sum2 = 0.;
        let mut n = 0;
        if !stable {
            self.as_dim1().apply_window_to(out, window, |v, v_rm| {
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
                    // var肯定大于等于0，否则只能是精度问题
                    if var > 1e-14 {
                        var * n_f64 / (n - 1).f64()
                    } else {
                        0.
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
            let mut mean = 0.;
            // Welford's method for the online variance-calculation
            // using Kahan summation
            // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            self.as_dim1()
                .stable_apply_window_to(out, window, |v, v_rm| {
                    if v.notnan() {
                        n += 1;
                        let delta = kh_sum(v, -mean, c1);
                        mean += delta / n.f64();
                        let delta2 = kh_sum(v, -mean, c3);
                        sum2 += delta * delta2;
                    };
                    let res = if n >= min_periods {
                        let n_f64 = n.f64();
                        let var = sum2 / n_f64;
                        // var肯定大于等于0，否则只能是精度问题
                        if var > 1e-14 {
                            var * n_f64 / (n - 1).f64()
                        } else {
                            0.
                        }
                    } else {
                        f64::NAN
                    };
                    if v_rm.notnan() {
                        n -= 1;
                        if n > 0 {
                            let delta = kh_sum(v_rm, -mean, c2);
                            mean -= delta / n.f64();
                            let delta2 = kh_sum(v_rm, -mean, c4);
                            sum2 -= delta * delta2;
                        }
                    };
                    res
                })
        }
    }

    fn ts_skew<SO>(
        &self,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T: Number,
    {
        let min_periods = min_periods.max(3);
        let window = min(self.len(), window);
        if (window < min_periods) | (window < 3) {
            // 如果滚动窗口是1或2则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut sum = 0.;
        let mut sum2 = 0.;
        let mut sum3 = 0.;
        let mut n = 0;
        if !stable {
            self.as_dim1().apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    sum += v;
                    let v2 = v * v;
                    sum2 += v2;
                    sum3 += v2 * v;
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let mut var = sum2 / n_f64;
                    let mut mean = sum / n_f64;
                    var -= mean.powi(2);
                    if var <= 1e-14 {
                        // 标准差为0， 则偏度为0
                        0.
                    } else {
                        let std = var.sqrt(); // std
                        let res = sum3 / n_f64; // Ex^3
                        mean /= std; // mean / std
                        let adjust = (n * (n - 1)).f64().sqrt() / (n - 2).f64();
                        adjust * (res / std.powi(3) - 3. * mean - mean.powi(3))
                    }
                } else {
                    f64::NAN
                };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        let v = v.f64();
                        n -= 1;
                        sum -= v;
                        let v2 = v * v;
                        sum2 -= v2;
                        sum3 -= v2 * v;
                    };
                }
                res
            });
        } else {
            define_c!(c1, c2, c3, c4, c5, c6);
            self.as_dim1()
                .stable_apply_window_to(out, window, |v, v_rm| {
                    if v.notnan() {
                        n += 1;
                        sum.kh_sum(v, c1);
                        let v2 = v * v;
                        sum2.kh_sum(v2, c2);
                        sum3.kh_sum(v2 * v, c3);
                    };
                    let res = if n >= min_periods {
                        let n_f64 = n.f64();
                        let mut var = sum2 / n_f64;
                        let mut mean = sum / n_f64;
                        var -= mean.powi(2);
                        if var <= 1e-14 {
                            // 标准差为0， 则偏度为0
                            0.
                        } else {
                            let std = var.sqrt(); // std
                            let res = sum3 / n_f64; // Ex^3
                            mean /= std; // mean / std
                            let adjust = (n * (n - 1)).f64().sqrt() / (n - 2).f64();
                            adjust * (res / std.powi(3) - 3. * mean - mean.powi(3))
                        }
                    } else {
                        f64::NAN
                    };
                    if v_rm.notnan() {
                        n -= 1;
                        sum.kh_sum(-v_rm, c4);
                        let v_rm2 = v_rm * v_rm;
                        sum2.kh_sum(-v_rm2, c5);
                        sum3.kh_sum(-v_rm2 * v_rm, c6);
                    };
                    res
                })
        }
    }

    fn ts_kurt<SO>(
        &self,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T: Number,
    {
        let min_periods = min_periods.max(4);
        let window = min(self.len(), window);
        if (window < min_periods) | (window < 4) {
            // 如果滚动窗口是小于4则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut sum = 0.;
        let mut sum2 = 0.;
        let mut sum3 = 0.;
        let mut sum4 = 0.;
        let mut n = 0;
        if !stable {
            self.as_dim1().apply_window_to(out, window, |v, v_rm| {
                if v.notnan() {
                    n += 1;
                    let v = v.f64();
                    sum += v;
                    let v2 = v * v;
                    sum2 += v2;
                    sum3 += v2 * v;
                    sum4 += v2 * v2;
                };
                let res = if n >= min_periods {
                    let n_f64 = n.f64();
                    let mut var = sum2 / n_f64;
                    let mean = sum / n_f64;
                    var -= mean.powi(2);
                    if var <= 1e-14 {
                        // 标准差为0， 则峰度为0
                        0.
                    } else {
                        let n_f64 = n.f64();
                        let var2 = var * var; // var^2
                        let ex4 = sum4 / n_f64; // Ex^4
                        let ex3 = sum3 / n_f64; // Ex^3
                        let mean2_var = mean * mean / var; // (mean / std)^2
                        let out = (ex4 - 4. * mean * ex3) / var2
                            + 6. * mean2_var
                            + 3. * mean2_var.powi(2);
                        1. / ((n - 2) * (n - 3)).f64()
                            * ((n.pow(2) - 1).f64() * out - (3 * (n - 1).pow(2)).f64())
                    }
                } else {
                    f64::NAN
                };
                if let Some(v) = v_rm {
                    if v.notnan() {
                        let v = v.f64();
                        n -= 1;
                        sum -= v;
                        let v2 = v * v;
                        sum2 -= v2;
                        sum3 -= v2 * v;
                        sum4 -= v2 * v2;
                    };
                }
                res
            });
        } else {
            define_c!(c1, c2, c3, c4, c5, c6, c7, c8);
            self.as_dim1()
                .stable_apply_window_to(out, window, |v, v_rm| {
                    if v.notnan() {
                        n += 1;
                        sum.kh_sum(v, c1);
                        let v2 = v * v;
                        sum2.kh_sum(v2, c2);
                        sum3.kh_sum(v2 * v, c3);
                        sum4.kh_sum(v2 * v2, c4);
                    };
                    let res = if n >= min_periods {
                        let n_f64 = n.f64();
                        let mut var = sum2 / n_f64;
                        let mean = sum / n_f64;
                        var -= mean.powi(2);
                        if var <= 1e-14 {
                            // 标准差为0， 则峰度为0
                            0.
                        } else {
                            let n_f64 = n.f64();
                            let var2 = var * var; // var^2
                            let ex4 = sum4 / n_f64; // Ex^4
                            let ex3 = sum3 / n_f64; // Ex^3
                            let mean2_var = mean * mean / var; // (mean / std)^2
                            let out = (ex4 - 4. * mean * ex3) / var2
                                + 6. * mean2_var
                                + 3. * mean2_var.powi(2);
                            1. / ((n - 2) * (n - 3)).f64()
                                * ((n.pow(2) - 1).f64() * out - (3 * (n - 1).pow(2)).f64())
                        }
                    } else {
                        f64::NAN
                    };
                    if v_rm.notnan() {
                        n -= 1;
                        sum.kh_sum(-v_rm, c5);
                        let v_rm2 = v_rm * v_rm;
                        sum2.kh_sum(-v_rm2, c6);
                        sum3.kh_sum(-v_rm2 * v_rm, c7);
                        sum4.kh_sum(-v_rm2 * v_rm2, c8);
                    };
                    res
                })
        }
    }

    fn ts_prod<SO>(&self, out: &mut ArrBase<SO, Ix1>, window: usize, min_periods: usize) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T: Number,
    {
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut prod = 1.;
        let mut zero_num = 0;
        let mut n = 0;
        self.as_dim1().apply_window_to(out, window, |v, v_rm| {
            if v.notnan() {
                n += 1;
                if *v != T::zero() {
                    prod *= v.f64();
                } else {
                    zero_num += 1;
                }
            };
            let res = if n >= min_periods {
                if zero_num == 0 {
                    prod
                } else {
                    0.
                }
            } else {
                f64::NAN
            };
            if let Some(v) = v_rm {
                if v.notnan() {
                    n -= 1;
                    if *v != T::zero() {
                        prod /= v.f64();
                    } else {
                        zero_num -= 1;
                    }
                };
            }
            res
        });
    }

    fn ts_prod_mean<SO>(&self, out: &mut ArrBase<SO, Ix1>, window: usize, min_periods: usize) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T: Number,
    {
        let window = min(self.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut prod = 1.;
        let mut zero_num = 0;
        let mut n = 0;
        self.as_dim1().apply_window_to(out, window, |v, v_rm| {
            if v.notnan() {
                n += 1;
                if *v != T::zero() {
                    prod *= v.f64();
                } else {
                    zero_num += 1;
                }
            };
            let res = if n >= min_periods {
                if zero_num == 0 {
                    prod.powf(1. / n.f64())
                } else {
                    0.
                }
            } else {
                f64::NAN
            };
            if let Some(v) = v_rm {
                if v.notnan() {
                    n -= 1;
                    if *v != T::zero() {
                        prod /= v.f64();
                    } else {
                        zero_num -= 1;
                    }
                };
            }
            res
        });
    }
}
