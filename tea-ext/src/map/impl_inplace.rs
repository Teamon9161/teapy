#[cfg(feature = "lazy")]
use lazy::Expr;
use ndarray::{DataMut, Dimension, Zip};
use std::ptr::read;
use tea_core::prelude::*;

/// the method to use when fillna
/// Ffill: use forward value to fill nan.
/// Bfill: use backward value to fill nan.
/// Vfill: use a specified value to fill nan
#[derive(Copy, Clone)]
#[allow(dead_code)]
pub enum FillMethod {
    Ffill,
    Bfill,
    Vfill,
}

#[cfg(feature = "agg")]
#[derive(Copy, Clone)]
#[allow(dead_code)]
pub enum WinsorizeMethod {
    Quantile,
    Median,
    Sigma,
}

#[arr_inplace_ext(lazy = "view_mut")]
impl<T: Send + Sync, S: DataMut<Elem = T>, D: Dimension> InplaceExt for ArrBase<S, D> {
    #[lazy_exclude]
    pub fn shift(&mut self, n: i32, fill: Option<T>)
    where
        T: GetNone + Clone,
    {
        if self.is_empty() || (n == 0) {
            return;
        }
        let fill = fill.unwrap_or_else(T::none);
        let mut arr = self.as_dim1_mut();
        let len = arr.len();
        let n_usize = n.unsigned_abs() as usize;
        if n_usize >= len {
            // special case, shift more than length
            arr.fill(fill);
        } else if n > 0 {
            for i in (n_usize..len).rev() {
                unsafe {
                    let src = arr.uget(i - n_usize);
                    *arr.uget_mut(i) = read(src);
                }
            }
            // Fill the first n elements with fill
            for i in 0..n_usize {
                unsafe {
                    *arr.uget_mut(i) = fill.clone();
                }
            }
        } else {
            // Shift to the left
            for i in 0..len - n_usize {
                unsafe {
                    let src = arr.uget(i + n_usize);
                    *arr.uget_mut(i) = read(src);
                }
            }
            // Fill the last n elements with fill
            for i in len - n_usize..len {
                unsafe {
                    *arr.uget_mut(i) = fill.clone();
                }
            }
        }
    }

    #[lazy_exclude]
    pub fn diff(&mut self, n: i32, fill: Option<T>)
    where
        T: GetNone + Clone + std::ops::Sub<T, Output = T>,
    {
        if self.is_empty() || (n == 0) {
            return;
        }
        let fill = fill.unwrap_or_else(T::none);
        let mut arr = self.as_dim1_mut();
        let len = arr.len();
        let n_usize = n.unsigned_abs() as usize;
        if n_usize >= len {
            // special case, shift more than length
            arr.fill(fill);
        } else if n > 0 {
            for i in (n_usize..len).rev() {
                unsafe {
                    let src = read(arr.uget(i - n_usize));
                    let v = arr.uget_mut(i);
                    // let tmp = read(v);
                    *v = read(v) - src;
                }
            }
            // Fill the first n elements with fill
            for i in 0..n_usize {
                unsafe {
                    *arr.uget_mut(i) = fill.clone();
                }
            }
        } else {
            // Shift to the left
            for i in 0..len - n_usize {
                unsafe {
                    let src = read(arr.uget(i + n_usize));
                    let v = arr.uget_mut(i);
                    *v = read(v) - src;
                }
            }
            // Fill the last n elements with fill
            for i in len - n_usize..len {
                unsafe {
                    *arr.uget_mut(i) = fill.clone();
                }
            }
        }
    }

    #[lazy_exclude]
    pub fn fillna<T2>(&mut self, method: FillMethod, value: Option<T2>)
    where
        T: GetNone + Clone,
        T2: Cast<T> + Clone + Send + Sync,
    {
        use FillMethod::*;
        let mut arr = self.as_dim1_mut();
        match method {
            Ffill | Bfill => {
                let mut last_valid: Option<T> = None;
                let value = value.map(|v| v.cast());
                let mut f = |v: &mut T| {
                    if v.is_none() {
                        if let Some(lv) = last_valid.as_ref() {
                            *v = lv.clone();
                        } else if let Some(value) = &value {
                            *v = value.clone();
                        }
                    } else {
                        // v is valid, update last_valid
                        last_valid = Some(v.clone());
                    }
                };
                if let Ffill = method {
                    arr.apply_mut(f)
                } else {
                    for v in arr.iter_mut().rev() {
                        f(v);
                    }
                }
            }
            Vfill => {
                let value = value.expect("Fill value must be pass when using value to fillna");
                let value: T = value.cast();
                arr.apply_mut(|v| {
                    if v.is_none() {
                        *v = value.clone()
                    }
                });
            }
        }
    }

    #[lazy_exclude]
    fn clip<T2, T3>(&mut self, min: T2, max: T3)
    where
        T: Number,
        T2: Number + Cast<T>,
        T3: Number + Cast<T>,
    {
        let (min, max) = (T::fromas(min), T::fromas(max));
        assert!(min <= max, "min must smaller than max in clamp");
        assert!(
            min.notnan() & max.notnan(),
            "min and max should not be NaN in clamp"
        );
        self.as_dim1_mut().apply_mut(|v| {
            if *v > max {
                // Note that NaN is excluded
                *v = max;
            } else if *v < min {
                *v = min;
            }
        })
    }

    #[teapy(type = "numeric")]
    #[cfg(feature = "agg")]
    #[inline]
    /// Sandardize the array using zscore method on a given axis
    fn zscore(&mut self, min_periods: usize, stable: bool)
    where
        T: Number,
        f64: Cast<T>,
    {
        use crate::agg::AggExt1d;
        let mut arr = self.as_dim1_mut();
        let (mean, var) = arr.meanvar_1d(min_periods, stable);
        if var == 0. {
            arr.apply_mut(|v| *v = 0.0.cast());
        } else if var.isnan() {
            arr.apply_mut(|v| *v = f64::NAN.cast());
        } else {
            arr.apply_mut(|v| *v = ((v.f64() - mean) / var.sqrt()).cast());
        }
    }

    #[teapy(type = "numeric")]
    #[cfg(feature = "agg")]
    fn winsorize(&mut self, method: WinsorizeMethod, method_params: Option<f64>, stable: bool)
    where
        T: Number,
        f64: Cast<T>,
    {
        use crate::agg::*;
        use WinsorizeMethod::*;
        let mut arr = self.as_dim1_mut();
        match method {
            Quantile => {
                // default method is clip 1% and 99% quantile
                use crate::QuantileMethod::*;
                let method_params = method_params.unwrap_or(0.01);
                let min = arr.quantile_1d(method_params, Linear);
                let max = arr.quantile_1d(1. - method_params, Linear);
                if min.notnan() && (min != max) {
                    arr.clip_1d(min, max);
                }
            }
            Median => {
                // default method is clip median - 3 * mad, median + 3 * mad
                let method_params = method_params.unwrap_or(3.);
                let median = arr.median_1d();
                if median.notnan() {
                    let mad = arr.mapv(|v| (v.f64() - median).abs()).median_1d();
                    let min = median - method_params * mad;
                    let max = median + method_params * mad;
                    arr.clip_1d(min, max);
                }
            }
            Sigma => {
                // default method is clip mean - 3 * std, mean + 3 * std
                let method_params = method_params.unwrap_or(3.);
                let (mean, var) = arr.meanvar_1d(2, stable);
                if mean.notnan() {
                    let std = var.sqrt();
                    let min = mean - method_params * std;
                    let max = mean + method_params * std;
                    arr.clip_1d(min, max);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_shift() {
        let mut arr = Arr1::from_vec(vec![1, 2, 3, 4, 5]);
        arr.shift_1d(2, Some(0));
        let res = arr.into_raw_vec();
        assert_eq!(res, vec![0, 0, 1, 2, 3]);
        let mut arr = Arr1::from_vec(vec![1, 2, 3, 4, 5]);
        arr.shift_1d(-1, Some(100));
        assert_eq!(arr.clone().into_raw_vec(), vec![2, 3, 4, 5, 100]);
        arr.shift_1d(0, Some(100));
        assert_eq!(arr.into_raw_vec(), vec![2, 3, 4, 5, 100]);
    }

    #[test]
    fn test_diff() {
        let data = vec![1, 3, 5, 3, 9];
        let mut arr = Arr1::from_vec(data.clone());
        arr.diff_1d(2, Some(0));
        let res = arr.into_raw_vec();
        assert_eq!(res, vec![0, 0, 4, 0, 4]);
        let mut arr = Arr1::from_vec(data.clone());
        arr.diff_1d(-1, Some(100));
        assert_eq!(arr.into_raw_vec(), vec![-2, -2, 2, -6, 100]);
    }
}
