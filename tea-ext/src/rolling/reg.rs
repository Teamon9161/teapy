use ndarray::{Data, DataMut, DimMax, Dimension, Ix1, ShapeBuilder};
use num::traits::MulAdd;
use std::cmp::min;
use std::mem::MaybeUninit;
use tea_core::prelude::*;
use tea_core::utils::define_c;

#[cfg(feature = "agg")]
use tea_core::utils::CollectTrustedToVec;

#[cfg(feature = "agg")]
use crate::agg::*;
#[cfg(feature = "lazy")]
use lazy::Expr;

#[arr_map_ext(lazy = "view", type = "numeric")]
impl<T: Send + Sync, S: Data<Elem = T>, D: Dimension> RegTs for ArrBase<S, D> {
    fn ts_reg<SO>(
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
        let arr = self.as_dim1();
        // 错位相减核心公式：sum_xt(t) = sum_xt(t-1) - sum_x + n * new_element
        let window = min(arr.len(), window);
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
            arr.apply_window_to(out, window, |v, v_rm| {
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
            arr.apply_window_to(out, window, |v, v_rm| {
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

    fn ts_tsf<SO>(
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
        let arr = self.as_dim1();
        // 错位相减核心公式：sum_xt(t) = sum_xt(t-1) - sum_x + n * new_element
        let window = min(arr.len(), window);
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
            arr.apply_window_to(out, window, |v, v_rm| {
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
                    slope.mul_add((n + 1).f64(), intercept)
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
            arr.apply_window_to(out, window, |v, v_rm| {
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
                    slope.mul_add((n + 1).f64(), intercept)
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

    fn ts_reg_slope<SO>(
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
        let arr = self.as_dim1();
        // 错位相减核心公式：sum_xt(t) = sum_xt(t-1) - sum_x + n * new_element
        let window = min(arr.len(), window);
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
            arr.apply_window_to(out, window, |v, v_rm| {
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
            arr.apply_window_to(out, window, |v, v_rm| {
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

    fn ts_reg_intercept<SO>(
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
        let arr = self.as_dim1();
        // 错位相减核心公式：sum_xt(t) = sum_xt(t-1) - sum_x + n * new_element
        let window = min(arr.len(), window);
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
            arr.apply_window_to(out, window, |v, v_rm| {
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
            arr.apply_window_to(out, window, |v, v_rm| {
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

    fn ts_reg_resid_mean<SO>(
        &self,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: usize,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T: Number,
    {
        let arr = self.as_dim1();
        // 错位相减核心公式：sum_xt(t) = sum_xt(t-1) - sum_x + n * new_element
        let window = min(arr.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut sum = 0.;
        let mut sum_xx = 0.;
        let mut sum_xt = 0.;
        let mut n = 0;
        arr.apply_window_to(out, window, |v, v_rm| {
            if v.notnan() {
                n += 1;
                let v = v.f64();
                sum_xt += n.f64() * v; // 错位相减法, 忽略nan带来的系数和window不一致问题
                sum += v;
                sum_xx += v * v;
            };
            let res = if n >= min_periods {
                let n_f64 = n.f64();
                let nn_add_n = n.mul_add(n, n);
                let sum_t = (nn_add_n >> 1).f64(); // sum of time from 1 to window
                                                   // denominator of slope
                let sum_tt = (n * nn_add_n * n.mul_add(2, 1)).f64() / 6.;
                let divisor = sum_tt - sum_t.powi(2);
                let beta = (n_f64 * sum_xt - sum_t * sum) / divisor;
                let alpha = sum_t.mul_add(-beta, sum) / n_f64;
                let resid_sum = sum_xx - 2. * alpha * sum - 2. * beta * sum_xt
                    + alpha * alpha * n_f64
                    + 2. * alpha * beta * sum_t
                    + beta * beta * sum_tt;
                resid_sum / n_f64
            } else {
                f64::NAN
            };
            if let Some(v) = v_rm {
                if v.notnan() {
                    let v = v.f64();
                    n -= 1;
                    sum_xt -= sum;
                    sum -= v;
                    sum_xx -= v * v;
                };
            }
            res
        });
    }
}

#[arr_map2_ext(lazy = "view2", type = "numeric", type2 = "numeric")]
impl<T: Send + Sync, S: Data<Elem = T>, D: Dimension> Reg2Ts for ArrBase<S, D> {
    fn ts_regx_beta<S2, D2, T2, SO>(
        &self,
        x: &ArrBase<S2, D2>,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        S2: Data<Elem = T2>,
        D2: Dimension,
        D: DimMax<D2>,
        T: Number,
        T2: Number,
    {
        let arr = self.as_dim1();
        let x1 = x.as_dim1();
        let window = min(arr.len(), window);
        let window = min(x1.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut sum_a = 0.;
        let mut sum_b = 0.;
        let mut sum_b2 = 0.;
        let mut sum_ab = 0.;
        let mut n = 0;
        if !stable {
            arr.apply_window_with_to(&x1, out, window, |va, vb, va_rm, vb_rm| {
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
            arr.stable_apply_window_with_to(&x1, out, window, |va, vb, va_rm, vb_rm| {
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

    fn ts_regx_alpha<S2, D2, T2, SO>(
        &self,
        x: &ArrBase<S2, D2>,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: usize,
        stable: bool,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        S2: Data<Elem = T2>,
        D2: Dimension,
        D: DimMax<D2>,
        T: Number,
        T2: Number,
    {
        let arr = self.as_dim1();
        let x1 = x.as_dim1();
        let window = min(arr.len(), window);
        let window = min(x1.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut sum_a = 0.;
        let mut sum_b = 0.;
        let mut sum_b2 = 0.;
        let mut sum_ab = 0.;
        let mut n = 0;
        if !stable {
            arr.apply_window_with_to(&x1, out, window, |va, vb, va_rm, vb_rm| {
                if va.notnan() && vb.notnan() {
                    n += 1;
                    let (va, vb) = (va.f64(), vb.f64());
                    sum_a += va;
                    sum_b += vb;
                    sum_b2 += vb.powi(2);
                    sum_ab += va * vb;
                };
                let res = if n >= min_periods {
                    let beta =
                        (n.f64() * sum_ab - sum_a * sum_b) / (n.f64() * sum_b2 - sum_b.powi(2));
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
            arr.stable_apply_window_with_to(&x1, out, window, |va, vb, va_rm, vb_rm| {
                if va.notnan() && vb.notnan() {
                    n += 1;
                    sum_a.kh_sum(va, c1);
                    sum_b.kh_sum(vb, c2);
                    sum_ab.kh_sum(va * vb, c3);
                    sum_b2.kh_sum(vb.powi(2), c7);
                };
                let res = if n >= min_periods {
                    let beta =
                        (n.f64() * sum_ab - sum_a * sum_b) / (n.f64() * sum_b2 - sum_b.powi(2));
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

    #[cfg(feature = "agg")]
    fn ts_regx_resid_mean<S2, D2, T2, SO>(
        &self,
        x: &ArrBase<S2, D2>,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: usize,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        S2: Data<Elem = T2>,
        D2: Dimension,
        D: DimMax<D2>,
        T: Number,
        T2: Number,
    {
        let arr = self.as_dim1();
        let x1 = x.as_dim1();
        let window = min(arr.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut sum_a = 0.;
        let mut sum_b = 0.;
        let mut sum_b2 = 0.;
        let mut sum_ab = 0.;
        let mut n = 0;
        for i in 0..window - 1 {
            // safety：i is inbound
            let (va, vb) = unsafe { (*arr.uget(i), *x1.uget(i)) };
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
                let resid = (0..=i)
                    .map(|j| {
                        let (vy, vx) = unsafe { (*arr.uget(j), *x1.uget(j)) };
                        vy.f64() - alpha - beta * vx.f64()
                    })
                    .collect_trusted();
                let mean = Arr1::from_vec(resid).mean_1d(1, false);
                unsafe { out.uget_mut(i).write(mean) };
            } else {
                unsafe { out.uget_mut(i).write(f64::NAN) };
            };
        }
        for (start, end) in (window - 1..arr.len()).enumerate() {
            // safety：start, end is inbound
            let (va, vb) = unsafe { (*arr.uget(end), *x1.uget(end)) };
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
                let resid = (start..=end)
                    .map(|j| {
                        let (vy, vx) = unsafe { (*arr.uget(j), *x1.uget(j)) };
                        vy.f64() - alpha - beta * vx.f64()
                    })
                    .collect_trusted();
                let mean = Arr1::from_vec(resid).mean_1d(1, false);
                unsafe { out.uget_mut(end).write(mean) };
            } else {
                unsafe { out.uget_mut(end).write(f64::NAN) };
            };
            let (va, vb) = unsafe { (*arr.uget(start), *x1.uget(start)) };
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

    #[cfg(feature = "agg")]
    fn ts_regx_resid_std<S2, D2, T2, SO>(
        &self,
        x: &ArrBase<S2, D2>,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: usize,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        S2: Data<Elem = T2>,
        D2: Dimension,
        D: DimMax<D2>,
        T: Number,
        T2: Number,
    {
        let arr = self.as_dim1();
        let x1 = x.as_dim1();
        let window = min(arr.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut sum_a = 0.;
        let mut sum_b = 0.;
        let mut sum_b2 = 0.;
        let mut sum_ab = 0.;
        let mut n = 0;
        for i in 0..window - 1 {
            // safety：i is inbound
            let (va, vb) = unsafe { (*arr.uget(i), *x1.uget(i)) };
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
                let resid = (0..=i)
                    .map(|j| {
                        let (vy, vx) = unsafe { (*arr.uget(j), *x1.uget(j)) };
                        vy.f64() - alpha - beta * vx.f64()
                    })
                    .collect_trusted();
                let std = Arr1::from_vec(resid).std_1d(2, false);
                unsafe { out.uget_mut(i).write(std) };
            } else {
                unsafe { out.uget_mut(i).write(f64::NAN) };
            };
        }
        for (start, end) in (window - 1..arr.len()).enumerate() {
            // safety：start, end is inbound
            let (va, vb) = unsafe { (*arr.uget(end), *x1.uget(end)) };
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
                let resid = (start..=end)
                    .map(|j| {
                        let (vy, vx) = unsafe { (*arr.uget(j), *x1.uget(j)) };
                        vy.f64() - alpha - beta * vx.f64()
                    })
                    .collect_trusted();
                let std = Arr1::from_vec(resid).std_1d(2, false);
                unsafe { out.uget_mut(end).write(std) };
            } else {
                unsafe { out.uget_mut(end).write(f64::NAN) };
            };
            let (va, vb) = unsafe { (*arr.uget(start), *x1.uget(start)) };
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

    #[cfg(feature = "agg")]
    fn ts_regx_resid_skew<S2, D2, T2, SO>(
        &self,
        x: &ArrBase<S2, D2>,
        out: &mut ArrBase<SO, Ix1>,
        window: usize,
        min_periods: usize,
    ) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        S2: Data<Elem = T2>,
        D2: Dimension,
        D: DimMax<D2>,
        T: Number,
        T2: Number,
    {
        let arr = self.as_dim1();
        let x1 = x.as_dim1();
        let window = min(arr.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut sum_a = 0.;
        let mut sum_b = 0.;
        let mut sum_b2 = 0.;
        let mut sum_ab = 0.;
        let mut n = 0;
        for i in 0..window - 1 {
            // safety：i is inbound
            let (va, vb) = unsafe { (*arr.uget(i), *x1.uget(i)) };
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
                let resid = (0..=i)
                    .map(|j| {
                        let (vy, vx) = unsafe { (*arr.uget(j), *x1.uget(j)) };
                        vy.f64() - alpha - beta * vx.f64()
                    })
                    .collect_trusted();
                let std = Arr1::from_vec(resid).skew_1d(3, false);
                unsafe { out.uget_mut(i).write(std) };
            } else {
                unsafe { out.uget_mut(i).write(f64::NAN) };
            };
        }
        for (start, end) in (window - 1..arr.len()).enumerate() {
            // safety：start, end is inbound
            let (va, vb) = unsafe { (*arr.uget(end), *x1.uget(end)) };
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
                let resid = (start..=end)
                    .map(|j| {
                        let (vy, vx) = unsafe { (*arr.uget(j), *x1.uget(j)) };
                        vy.f64() - alpha - beta * vx.f64()
                    })
                    .collect_trusted();
                let std = Arr1::from_vec(resid).skew_1d(3, false);
                unsafe { out.uget_mut(end).write(std) };
            } else {
                unsafe { out.uget_mut(end).write(f64::NAN) };
            };
            let (va, vb) = unsafe { (*arr.uget(start), *x1.uget(start)) };
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
}
