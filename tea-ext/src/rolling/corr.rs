use ndarray::{Data, DataMut, DimMax, Dimension, Ix1, ShapeBuilder};
use std::cmp::min;
use std::mem::MaybeUninit;
use tea_core::prelude::*;
use tea_core::utils::define_c;

#[cfg(feature = "lazy")]
use lazy::Expr;

#[arr_map2_ext(lazy = "view2", type = "numeric", type2 = "numeric")]
impl<T: Send + Sync, S: Data<Elem = T>, D: Dimension> CorrTs for ArrBase<S, D> {
    fn ts_cov<S2, D2, T2, SO>(
        &self,
        other: &ArrBase<S2, D2>,
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
        let window = min(arr.len(), window);
        let window = min(other.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut sum_a = 0.;
        let mut sum_b = 0.;
        let mut sum_ab = 0.;
        let mut n = 0;
        if !stable {
            self.as_dim1().apply_window_with_to(
                &other.as_dim1(),
                out,
                window,
                |va, vb, va_rm, vb_rm| {
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
                },
            );
        } else {
            define_c!(c1, c2, c3, c4, c5, c6);
            self.as_dim1().stable_apply_window_with_to(
                &other.as_dim1(),
                out,
                window,
                |va, vb, va_rm, vb_rm| {
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
                },
            )
        }
    }

    fn ts_corr<S2, D2, T2, SO>(
        &self,
        other: &ArrBase<S2, D2>,
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
        let window = min(self.len(), window);
        let window = min(other.len(), window);
        if window < min_periods {
            // 如果滚动窗口是1则返回全nan
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
        }
        let mut sum_a = 0.;
        let mut sum2_a = 0.;
        let mut sum_b = 0.;
        let mut sum2_b = 0.;
        let mut sum_ab = 0.;
        let mut n = 0;
        if !stable {
            self.as_dim1().apply_window_with_to(
                &other.as_dim1(),
                out,
                window,
                |va, vb, va_rm, vb_rm| {
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
                },
            );
        } else {
            define_c!(c1, c2, c3, c4, c5, c6, c7, c8, c9, c10);
            self.as_dim1().stable_apply_window_with_to(
                &other.as_dim1(),
                out,
                window,
                |va, vb, va_rm, vb_rm| {
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
                },
            )
        }
    }
}
