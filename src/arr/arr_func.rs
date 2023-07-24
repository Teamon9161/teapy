use std::hash::Hash;

use ahash::RandomState;

use super::{export::*, ArrView1, GetNone};
// use super::groupby::CollectTrustedToVec;

/// the method to use when fillna
/// Ffill: use forward value to fill nan.
/// Bfill: use backward value to fill nan.
/// Vfill: use a specified value to fill nan
#[derive(Copy, Clone)]
pub enum FillMethod {
    Ffill,
    Bfill,
    Vfill,
}

#[derive(Copy, Clone)]
pub enum WinsorizeMethod {
    Quantile,
    Median,
    Sigma,
}

impl<T, S> ArrBase<S, Ix1>
where
    S: Data<Elem = T>,
{
    /// Remove NaN values in 1d array.
    #[inline]
    pub fn remove_nan_1d(self) -> Arr1<T>
    where
        T: Number,
    {
        Arr1::from_iter(self.into_iter().filter(|v| v.notnan()))
    }

    /// return -1 if there are not enough valid elements
    /// sort: whether to sort the result by the size of the element
    pub fn arg_partition_1d<S2>(&self, mut out: ArrBase<S2, Ix1>, kth: usize, sort: bool, rev: bool)
    where
        T: Number,
        S2: DataMut<Elem = i32>,
    {
        let n = self.count_notnan_1d() as usize;
        if n <= kth + 1 {
            if !sort {
                let mut out_pos = 0;
                for (i, v) in self.iter().enumerate() {
                    if v.notnan() {
                        unsafe { *out.uget_mut(out_pos) = i as i32 }
                        out_pos += 1;
                    }
                }
                for i in n..kth + 1 {
                    unsafe { *out.uget_mut(i) = -1 }
                }
            } else {
                let mut idx_sorted = Vec::from_iter(0..self.len() as i32);
                // let mut arr = self.0.to_owned().wrap();  // clone the array
                if !rev {
                    idx_sorted.sort_unstable_by(|a: &i32, b: &i32| {
                        let (va, vb) =
                            unsafe { (*self.uget((*a) as usize), *self.uget((*b) as usize)) }; // safety: out不超过self的长度
                        va.nan_sort_cmp(&vb)
                    })
                } else {
                    idx_sorted.sort_unstable_by(|a: &i32, b: &i32| {
                        let (va, vb) =
                            unsafe { (*self.uget((*a) as usize), *self.uget((*b) as usize)) }; // safety: out不超过self的长度
                        va.nan_sort_cmp_rev(&vb)
                    })
                }
                for (i, v) in idx_sorted.iter().take(n).enumerate() {
                    unsafe { *out.uget_mut(i) = *v }
                }
                for i in n..kth + 1 {
                    unsafe { *out.uget_mut(i) = -1 }
                }
            }
            return;
        }
        let mut out_c = self.0.to_owned(); // clone the array
        let slc = out_c.as_slice_mut().unwrap();
        let mut idx_sorted = Vec::from_iter(0..slc.len() as i32);
        if !rev {
            let sort_func = |a: &i32, b: &i32| {
                let (va, vb) = unsafe { (*self.uget((*a) as usize), *self.uget((*b) as usize)) }; // safety: out不超过self的长度
                va.nan_sort_cmp(&vb)
            };
            idx_sorted.select_nth_unstable_by(kth, sort_func);
            idx_sorted.truncate(kth + 1);
            if sort {
                idx_sorted.sort_unstable_by(sort_func)
            }
            out.apply_mut_with(&Arr1::from_vec(idx_sorted), |vo, v| *vo = *v);
        } else {
            let sort_func = |a: &i32, b: &i32| {
                let (va, vb) = unsafe { (*self.uget((*a) as usize), *self.uget((*b) as usize)) }; // safety: out不超过self的长度
                va.nan_sort_cmp_rev(&vb)
            };
            idx_sorted.select_nth_unstable_by(kth, sort_func);
            idx_sorted.truncate(kth + 1);
            if sort {
                idx_sorted.sort_unstable_by(sort_func)
            }
            out.apply_mut_with(&Arr1::from_vec(idx_sorted), |vo, v| *vo = *v);
        }
    }

    /// Hash each element of the array.
    #[inline]
    pub fn hash_1d(self, hasher: &RandomState) -> Arr1<u64>
    where
        T: Hash,
    {
        self.map(|v| hasher.hash_one(v))
    }

    /// Remove NaN values in two 1d arrays.
    #[inline]
    pub fn remove_nan2_1d<S2, T2>(&self, other: &ArrBase<S2, Ix1>) -> (Arr1<T>, Arr1<T2>)
    where
        T: Number,
        S2: Data<Elem = T2>,
        T2: Number,
    {
        let (out1, out2): (Vec<_>, Vec<_>) = zip(self, other)
            .filter(|(v1, v2)| v1.notnan() & v2.notnan())
            .unzip();
        (Arr1::from_vec(out1), Arr1::from_vec(out2))
    }

    /// Take value on a given axis and clone to a new array, just work on 1d array
    ///
    /// if you want to along axis, select arbitrary subviews corresponding to indices and and
    /// copy them into a new array, use select instead.
    ///
    /// # Safety
    ///
    /// The index in `slc` must be correct.
    #[inline]
    pub unsafe fn take_clone_1d_unchecked<S2, S3>(
        &self,
        mut out: ArrBase<S2, Ix1>,
        slc: ArrBase<S3, Ix1>,
    ) where
        T: Clone,
        S2: DataMut<Elem = T>,
        S3: Data<Elem = usize> + Send + Sync,
    {
        let value = slc
            .iter()
            .map(|idx| self.uget(*idx).clone())
            .collect_trusted();
        let value_view = ArrView1::<_>::from_ref_vec(out.raw_dim(), &value);
        out.apply_mut_with(&value_view, |vo, v| *vo = v.clone());
    }

    /// Take value on a given axis and clone to a new array, just work on 1d array
    ///
    /// # Safety
    ///
    /// The index in `slc` must be correct.
    #[inline]
    pub unsafe fn take_option_clone_1d_unchecked<S2, S3>(
        &self,
        mut out: ArrBase<S2, Ix1>,
        slc: ArrBase<S3, Ix1>,
    ) where
        T: Clone + GetNone,
        S2: DataMut<Elem = T>,
        S3: Data<Elem = Option<usize>> + Send + Sync,
    {
        let value = slc
            .iter()
            .map(|idx| {
                if let Some(idx) = idx {
                    self.uget(*idx).clone()
                } else {
                    T::none()
                }
            })
            .collect_trusted();

        let value_view = ArrView1::<_>::from_ref_vec(out.raw_dim(), &value);
        out.apply_mut_with(&value_view, |vo, v| *vo = v.clone());
    }

    #[inline]
    pub fn filter_1d<S2, S3>(&self, mut out: ArrBase<S2, Ix1>, mask: ArrBase<S3, Ix1>)
    where
        T: Clone,
        S2: DataMut<Elem = T>,
        S3: Data<Elem = bool> + Send + Sync,
    {
        zip(self, mask.into_iter())
            .filter(|(_v, m)| *m)
            .zip(out.iter_mut())
            .for_each(|((v, _m), o)| *o = v.clone())
    }

    #[inline]
    pub fn put_mask_1d<S2, S3>(&mut self, mask: &ArrBase<S2, Ix1>, value: &ArrBase<S3, Ix1>)
    where
        T: Clone,
        S: DataMut<Elem = T>,
        S3: Data<Elem = T> + Send + Sync,
        S2: Data<Elem = bool> + Send + Sync,
    {
        zip(self, mask)
            .filter(|(_v, m)| **m)
            .zip(value)
            .for_each(|((a, _m), v)| *a = v.clone())
    }
}

impl<T, S, D> ArrBase<S, D>
where
    S: Data<Elem = T>,
    D: Dimension,
{
    pub fn is_nan(&self) -> Arr<bool, D>
    where
        T: Number,
    {
        self.mapv(|v| v.isnan())
    }

    pub fn not_nan(&self) -> Arr<bool, D>
    where
        T: Number,
    {
        self.mapv(|v| v.notnan())
    }

    pub fn filter<S2>(&self, mask: &ArrBase<S2, Ix1>, axis: i32, par: bool) -> Arr<T, D>
    where
        T: Default + Send + Sync + Clone,
        D: Dimension,
        S2: Data<Elem = bool> + Send + Sync,
    {
        let f_flag = !self.is_standard_layout();
        let mut new_dim = self.raw_dim();
        let axis = self.norm_axis(axis);
        assert_eq!(
            new_dim.slice()[axis.index()],
            mask.len(),
            "Number of elements on the axis must equal to the length of mask array"
        );
        new_dim.slice_mut()[axis.index()] = mask.count_v_1d(true) as usize;
        let shape = new_dim.into_shape().set_f(f_flag);
        let mut out = Arr::<T, D>::default(shape);
        let mut out_wr = out.view_mut();
        self.apply_along_axis(&mut out_wr, axis, par, |x_1d, out_1d| {
            x_1d.wrap().filter_1d(out_1d, mask.view())
        });
        out
    }

    /// Take value on a given axis and clone to a new array,
    ///
    /// if you want to along axis, select arbitrary subviews corresponding to indices and and
    /// copy them into a new array, use select instead.
    ///
    /// # Safety
    ///
    /// The index in `slc` must be correct.
    pub unsafe fn take_clone_unchecked<S2>(
        &self,
        slc: ArrBase<S2, Ix1>,
        axis: i32,
        par: bool,
    ) -> Arr<T, D>
    where
        T: Clone + Default + Send + Sync,
        D: Dimension,
        S2: Data<Elem = usize> + Send + Sync,
    {
        let f_flag = !self.is_standard_layout();
        let axis = self.norm_axis(axis);
        let mut new_dim = self.raw_dim();
        new_dim.slice_mut()[axis.index()] = slc.len();
        let shape = new_dim.into_shape().set_f(f_flag);
        let mut out = Arr::<T, D>::default(shape);
        let mut out_wr = out.view_mut();
        self.apply_along_axis(&mut out_wr, axis, par, |x_1d, out_1d| {
            x_1d.wrap().take_clone_1d_unchecked(out_1d, slc.view())
        });
        out
    }

    /// Take value on a given axis and clone to a new array,
    ///
    /// # Safety
    ///
    /// The index in `slc` must be correct.
    pub unsafe fn take_option_clone_unchecked<S2>(
        &self,
        slc: ArrBase<S2, Ix1>,
        axis: i32,
        par: bool,
    ) -> Arr<T, D>
    where
        T: Clone + Default + GetNone + Send + Sync,
        D: Dimension,
        S2: Data<Elem = Option<usize>> + Send + Sync,
    {
        let f_flag = !self.is_standard_layout();
        let axis = self.norm_axis(axis);
        let mut new_dim = self.raw_dim();
        new_dim.slice_mut()[axis.index()] = slc.len();
        let shape = new_dim.into_shape().set_f(f_flag);
        let mut out = Arr::<T, D>::default(shape);
        let mut out_wr = out.view_mut();
        self.apply_along_axis(&mut out_wr, axis, par, |x_1d, out_1d| {
            x_1d.wrap()
                .take_option_clone_1d_unchecked(out_1d, slc.view())
        });
        out
    }

    /// Take value on a given axis and clone to a new array,
    ///
    /// if you want to along axis, select arbitrary subviews corresponding to indices and and
    /// copy them into a new array, use select instead.
    ///
    /// This function is safe because the slice are checked.
    pub fn take_clone<S2>(&self, slc: ArrBase<S2, Ix1>, axis: i32, par: bool) -> Arr<T, D>
    where
        T: Clone + Default + Send + Sync,
        D: Dimension,
        S2: Data<Elem = usize> + Send + Sync,
    {
        let axis = self.norm_axis(axis);
        let max_idx = self.shape()[axis.index()] - 1;
        assert!(slc.max_1d() <= max_idx, "The index to take is out of bound");
        unsafe { self.take_clone_unchecked(slc, axis.index() as i32, par) }
    }

    pub fn arg_partition(
        &self,
        mut kth: usize,
        sort: bool,
        rev: bool,
        axis: i32,
        par: bool,
    ) -> Arr<i32, D>
    where
        T: Number + Send + Sync,
        D: Dimension,
    {
        let f_flag = !self.is_standard_layout();
        let mut new_dim = self.raw_dim();
        let axis = self.norm_axis(axis);
        if kth >= new_dim.slice_mut()[axis.index()] {
            // panic!(
            //     "kth(={}) out of bounds ({})",
            //     kth,
            //     new_dim.slice_mut()[axis.index()]
            // )
            kth = new_dim.slice_mut()[axis.index()] - 1
        }
        new_dim.slice_mut()[axis.index()] = kth + 1;
        let shape = new_dim.into_shape().set_f(f_flag);
        let mut out = Arr::<i32, D>::default(shape);
        let mut out_wr = out.view_mut();
        self.apply_along_axis(&mut out_wr, axis, par, |x_1d, out_1d| {
            x_1d.arg_partition_1d(out_1d, kth, sort, rev)
        });
        out
    }
}

impl_map_nd!(
    cumsum,
    pub fn cumsum_1d<S2>(&self, out: &mut ArrBase<S2, D>, stable: bool) -> T
    {where T: Number,}
    {
        let mut sum = T::zero();
        if !stable {
            out.apply_mut_with(self, |vo, v| {
                if v.notnan() {
                    sum += *v;
                    *vo = sum;
                } else {
                    *vo = T::nan();
                }
            });
        } else {
            let c = &mut T::zero();
            out.apply_mut_with(self, |vo, v| {
                if v.notnan() {
                    sum.kh_sum(*v, c);
                    *vo = sum;
                } else {
                    *vo = T::nan();
                }
            });
        }

    }
);

impl_map_nd!(
    cumprod,
    pub fn cumprod_1d<S2>(&self, out: &mut ArrBase<S2, D>) -> T
    {where T: Number,}
    {
        let mut prod = T::one();
        out.apply_mut_with(self, |vo, v| {
            if v.notnan() {
                prod *= *v;
                *vo = prod;
            } else {
                *vo = T::nan();
            }
        });
    }
);

impl_map_nd!(
    zscore,
    /// Sandardize the array using zscore method on a given axis
    pub fn zscore_1d<S2>(&self, out: &mut ArrBase<S2, D>, stable: bool) -> T
    {where T: Number, f64: AsPrimitive<T>}
    {
        let (mean, var) = self.meanvar_1d(stable);
        if var == 0. {
            out.apply_mut(|v| *v = T::zero());
        } else if var.isnan() {
            out.apply_mut(|v| *v = T::nan());
        } else {
            out.apply_mut_with(self, |vo, v| *vo = ((v.f64() - mean) / var.sqrt()).as_());
        }
    }
);

impl_map_nd!(
    shift,
    pub fn shift_1d<S2>(&self, out: &mut ArrBase<S2, D>, n: i32, fill: T) -> T
    {where T: Clone; Send; Sync}
    {
        if n == 0 {
            out.apply_mut_with(self, |vo, v| *vo = v.clone());
        } else if n.unsigned_abs() as usize > self.shape()[0] -1 {
            out.apply_mut_with(self, |vo, _| *vo = fill.clone());
        } else if n > 0 {
            self.apply_window_to(out, n.unsigned_abs() as usize + 1, |_v, v_rm| {
                if let Some(v_rm) = v_rm {
                    v_rm.clone()
                } else {
                    fill.clone()
                }
            });
        } else {
            self.apply_revwindow_to(out, n.unsigned_abs() as usize + 1, |_v, v_rm| {
                if let Some(v_rm) = v_rm {
                    v_rm.clone()
                } else {
                    fill.clone()
                }
            });
        }
    }
);

impl_map_nd!(
    diff,
    pub fn diff_1d<S2>(&self, out: &mut ArrBase<S2, D>, n: i32) -> f64
    {where T: Number}
    {
        if n == 0 {
            out.apply_mut_with(self, |vo, _| *vo = 0.);
        } else if n.unsigned_abs() as usize > self.shape()[0] -1 {
            out.apply_mut_with(self, |vo, _| *vo = f64::NAN);
        } else if n > 0 {
            self.apply_window_to(out, n.unsigned_abs() as usize + 1, |v, v_rm| {
                if let Some(v_rm) = v_rm {
                    (*v - *v_rm).f64()
                } else {
                    f64::NAN
                }
            });
        } else {
            self.apply_revwindow_to(out, n.unsigned_abs() as usize + 1, |v, v_rm| {
                if let Some(v_rm) = v_rm {
                    (*v - *v_rm).f64()
                } else {
                    f64::NAN
                }
            });
        }
    }
);

impl_map_nd!(
    pct_change,
    pub fn pct_change_1d<S2>(&self, out: &mut ArrBase<S2, D>, n: i32) -> f64
    {where T: Number}
    {
        if n == 0 {
            out.apply_mut_with(self, |vo, _| *vo = 0.);
        } else if n.unsigned_abs() as usize > self.shape()[0] -1 {
            out.apply_mut_with(self, |vo, _| *vo = f64::NAN);
        } else if n > 0 {
            self.apply_window_to(out, n.unsigned_abs() as usize + 1, |v, v_rm| {
                if let Some(v_rm) = v_rm {
                    v.f64() / v_rm.f64() - 1.
                } else {
                    f64::NAN
                }
            });
        } else {
            self.apply_revwindow_to(out, n.unsigned_abs() as usize + 1, |v, v_rm| {
                if let Some(v_rm) = v_rm {
                    v.f64() / v_rm.f64() - 1.
                } else {
                    f64::NAN
                }
            });
        }
    }
);

impl_map_nd!(
    clip,
    pub fn clip_1d<S2, T2, T3>(&self, out: &mut ArrBase<S2, D>, min: T2, max: T3) -> T
    {
        where
        T: Number, T2: Number; AsPrimitive<T>, T3: Number; AsPrimitive<T>
    }
    {
        let (min, max) = (T::fromas(min), T::fromas(max));
        assert!(min <= max, "min must smaller than max in clamp");
        assert!(
            min.notnan() & max.notnan(),
            "min and max should not be NaN in clamp"
        );
        out.apply_mut_with(self, |vo, v| {
            if *v > max {
                // Note that NaN is excluded
                *vo = max
            } else if *v < min {
                *vo = min;
            } else {
                *vo = *v
            }
        })
    }
);

impl_map_nd!(
    fillna,
    pub fn fillna_1d<S2, T2>(&self, out: &mut ArrBase<S2, D>, method: FillMethod, value: Option<T2>) -> T
    {
        where
        T: Number,
        f64: AsPrimitive<T>,
        T2: AsPrimitive<T>; Send; Sync
    }
    {
        use FillMethod::*;
        let method = if value.is_some() {
            Vfill
        } else {
            method
        };
        match method {
            Bfill | Ffill => {
                let mut last_valid: Option<T> = None;
                let mut f = |vo: &mut T, v: &T| {
                    if v.isnan() {
                        if let Some(lv) = last_valid {
                            *vo = lv;
                        } else {
                            *vo = f64::NAN.as_();
                        }
                    } else { // v is valid, update last_valid
                        *vo = *v;
                        last_valid = Some(*v);
                    }
                };
                if let Ffill = method {
                    out.apply_mut_with(self, f)
                } else {
                    for (vo, v) in zip(out, self).rev() {
                        f(vo, v);
                    }
                }
            }
            Vfill => {
                let value = value.expect("Fill value must be pass when using value to fillna");
                let value: T = value.as_();
                out.apply_mut_with(self, |vo, v| if v.isnan() {
                    *vo = value;
                } else {
                    *vo = *v;
                });
            }
        }
    }
);

impl_map_nd!(
    winsorize,
    pub fn winsorize_1d<S2>(&self, out: &mut ArrBase<S2, D>, method: WinsorizeMethod, method_params: Option<f64>, stable: bool) -> T
    {
        where
        T: Number,
        f64: AsPrimitive<T>
    }
    {
        use WinsorizeMethod::*;
        match method {
            Quantile => {
                // default method is clip 1% and 99% quantile
                use super::QuantileMethod::*;
                let method_params = method_params.unwrap_or(0.01);
                let min = self.quantile_1d(method_params, Linear);
                let max = self.quantile_1d(1. - method_params, Linear);
                if min.notnan() && (min != max) {
                    self.clip_1d(out, min, max);
                } else {
                    // elements in the given axis are all NaN or equal to a constant
                    self.clone_to(out);
                }
            },
            Median => {
                // default method is clip median - 3 * mad, median + 3 * mad
                let method_params = method_params.unwrap_or(3.);
                let median = self.median_1d();
                if median.notnan() {
                    let mad = self.mapv(|v| (v.f64() - median).abs()).median_1d();
                    let min = median - method_params * mad;
                    let max = median + method_params * mad;
                    self.clip_1d(out, min, max);
                } else {
                    self.clone_to(out);
                }
            },
            Sigma => {
                    // default method is clip mean - 3 * std, mean + 3 * std
                let method_params = method_params.unwrap_or(3.);
                let (mean, var) = self.meanvar_1d(stable);
                if mean.notnan() {
                    let std = var.sqrt();
                    let min = mean - method_params * std;
                    let max = mean + method_params * std;
                    self.clip_1d(out, min, max);
                } else {
                    self.clone_to(out);
                }
            }
        }
    }
);

impl_map_nd!(
    argsort,
    pub fn argsort_1d<S2>(&self, out: &mut ArrBase<S2, D>, rev: bool) -> i32 {
        where
        T: Number
    }
    {
        assert!(out.len() >= self.len());
        let mut i = 0;
        out.apply_mut(|v| {
            *v = i;
            i += 1;
        }); // set elements of out array
        if !rev {
            out.sort_unstable_by(|a, b| {
                let (va, vb) = unsafe { (*self.uget((*a) as usize), *self.uget((*b) as usize)) }; // safety: out不超过self的长度
                va.nan_sort_cmp(&vb)
            });
        } else {
            out.sort_unstable_by(|a, b| {
                let (va, vb) = unsafe { (*self.uget((*a) as usize), *self.uget((*b) as usize)) }; // safety: out不超过self的长度
                va.nan_sort_cmp_rev(&vb)
            });
        }

    }
);

impl_map_nd!(
    rank,
    /// rank the array in a given axis
    #[allow(unused_assignments)]
    pub fn rank_1d<S2>(&self, out: &mut ArrBase<S2, D>, pct: bool, rev: bool) -> f64 {
        where
        T: Number
    }
    {
        let len = self.len();
        assert!(
            out.len() >= len,
            "the length of the input array not equal to the length of the output array"
        );
        if len == 0 {
            return;
        } else if len == 1 {
            // safety: out.len() == self.len() == 1
            unsafe { *out.uget_mut(0) = 1. };
            return;
        }
        // argsort at first
        let mut idx_sorted = Arr1::from_iter(0..len);
        if !rev {
            idx_sorted.sort_unstable_by(|a, b| {
                let (va, vb) = unsafe { (*self.uget(*a), *self.uget(*b)) }; // safety: out不超过self的长度
                va.nan_sort_cmp(&vb)
            });
        } else {
            idx_sorted.sort_unstable_by(|a, b| {
                let (va, vb) = unsafe { (*self.uget(*a), *self.uget(*b)) }; // safety: out不超过self的长度
                va.nan_sort_cmp_rev(&vb)
            });
        }

        // if the smallest value is nan then all the elements are nan
        if unsafe { *self.uget(*idx_sorted.uget(0)) }.isnan() {
            return out.apply_mut(|v| *v = f64::NAN);
        }
        let mut repeat_num = 1usize;
        let mut nan_flag = false;
        let (mut cur_rank, mut sum_rank) = (1usize, 0usize);
        let (mut idx, mut idx1) = (0, 0);
        if !pct {
            unsafe {
                for i in 0..len - 1 {
                    // safe because i_max = self.len()-2 and self.len() >= 2
                    (idx, idx1) = (*idx_sorted.uget(i), *idx_sorted.uget(i + 1));
                    let (v, v1) = (*self.uget(idx), *self.uget(idx1)); // 下一个值，safe because idx1 < arr.len()
                    if v1.isnan() {
                        // 下一个值是nan，说明后面的值全是nan
                        sum_rank += cur_rank;
                        cur_rank += 1;
                        for j in 0..repeat_num {
                            // safe because i >= repeat_num
                            *out.uget_mut(*idx_sorted.uget(i - j)) =
                                sum_rank.f64() / repeat_num.f64()
                        }
                        idx = i + 1;
                        nan_flag = true;
                        break;
                    } else if v == v1 {
                        // 当前值和下一个值相同，说明开始重复
                        repeat_num += 1;
                        sum_rank += cur_rank;
                        cur_rank += 1;
                    } else if repeat_num == 1 {
                        // 无重复，可直接得出排名
                        *out.uget_mut(idx) = cur_rank as f64;
                        cur_rank += 1;
                    } else {
                        // 当前元素是最后一个重复元素
                        sum_rank += cur_rank;
                        cur_rank += 1;
                        for j in 0..repeat_num {
                            // safe because i >= repeat_num
                            *out.uget_mut(*idx_sorted.uget(i - j)) =
                                sum_rank.f64() / repeat_num.f64()
                        }
                        sum_rank = 0; // rank和归零
                        repeat_num = 1; // 重复计数归一
                    }
                }
                if nan_flag {
                    for i in idx..len {
                        *out.uget_mut(*idx_sorted.uget(i)) = f64::NAN;
                    }
                } else {
                    sum_rank += cur_rank;
                    for i in len - repeat_num..len {
                        // safe because repeat_num <= len
                        *out.uget_mut(*idx_sorted.uget(i)) = sum_rank.f64() / repeat_num.f64()
                    }
                }
            }
        } else {
            let notnan_count = self.count_notnan_1d() as usize;
            unsafe {
                for i in 0..len - 1 {
                    // safe because i_max = arr.len()-2 and arr.len() >= 2
                    (idx, idx1) = (*idx_sorted.uget(i), *idx_sorted.uget(i + 1));
                    let (v, v1) = (*self.uget(idx), *self.uget(idx1)); // 下一个值，safe because idx1 < arr.len()
                    if v1.isnan() {
                        // 下一个值是nan，说明后面的值全是nan
                        sum_rank += cur_rank;
                        cur_rank += 1;
                        for j in 0..repeat_num {
                            // safe because i >= repeat_num
                            *out.uget_mut(*idx_sorted.uget(i - j)) =
                                sum_rank.f64() / (repeat_num * notnan_count).f64()
                        }
                        idx = i + 1;
                        nan_flag = true;
                        break;
                    } else if v == v1 {
                        // 当前值和下一个值相同，说明开始重复
                        repeat_num += 1;
                        sum_rank += cur_rank;
                        cur_rank += 1;
                    } else if repeat_num == 1 {
                        // 无重复，可直接得出排名
                        *out.uget_mut(idx) = cur_rank.f64() / notnan_count.f64();
                        cur_rank += 1;
                    } else {
                        // 当前元素是最后一个重复元素
                        sum_rank += cur_rank;
                        cur_rank += 1;
                        for j in 0..repeat_num {
                            // safe because i >= repeat_num
                            *out.uget_mut(*idx_sorted.uget(i - j)) =
                                sum_rank.f64() / (repeat_num * notnan_count).f64()
                        }
                        sum_rank = 0; // rank和归零
                        repeat_num = 1; // 重复计数归一
                    }
                }
                if nan_flag {
                    for i in idx..len {
                        *out.uget_mut(*idx_sorted.uget(i)) = f64::NAN;
                    }
                } else {
                    sum_rank += cur_rank;
                    for i in len - repeat_num..len {
                        // safe because repeat_num <= len
                        *out.uget_mut(*idx_sorted.uget(i)) =
                            sum_rank.f64() / (repeat_num * notnan_count).f64()
                    }
                }
            }
        }
    }
);

impl_map_nd!(
    split_group,
    /// Split values in several group by size.
    #[inline]
    pub fn split_group_1d<S2>(&self, out: &mut ArrBase<S2, D>, group: usize, rev: bool) -> i32
    {where T: Number}
    {
        let valid_count = self.count_notnan_1d();
        let mut rank = Arr1::<f64>::zeros(self.raw_dim());
        self.rank_1d(&mut rank, false, rev);
        out.apply_mut_with(&rank, |vo, v| {
            *vo = ((*v * group.f64()) / valid_count.f64()).ceil() as i32;
        })
    }
);

impl_map_inplace_nd!(
    fillna_inplace,
    pub fn fillna_inplace_1d<T2>(&mut self, method: FillMethod, value: Option<T2>)
    {
        where
        T: Number,
        T2: AsPrimitive<T>; Send; Sync
    }
    {
        use FillMethod::*;
        let method = if value.is_some() {
            Vfill
        } else {
            method
        };
        match method {
            Ffill | Bfill => {
                let mut last_valid: Option<T> = None;
                let mut f = |v: &mut T| {
                    if v.isnan() {
                        if let Some(lv) = last_valid {
                            *v = lv;
                        }
                    } else { // v is valid, update last_valid
                        last_valid = Some(*v);
                    }
                };
                if let Ffill = method {
                    self.apply_mut(f)
                } else {
                    for v in self.iter_mut().rev() {
                        f(v);
                    }
                }
            }
            Vfill => {
                let value = value.expect("Fill value must be pass when using value to fillna");
                let value: T = value.as_();
                self.apply_mut(|v| if v.isnan() {
                    *v = value
                });
            }
        }
    }
);

impl_map_inplace_nd!(
    clip_inplace,
    pub fn clip_inplace_1d<T2, T3>(&mut self, min: T2, max: T3)
    {
        where
        T: Number,
        T2: Number; AsPrimitive<T>,
        T3: Number; AsPrimitive<T>,
    }
    {
        let (min, max) = (T::fromas(min), T::fromas(max));
        assert!(min <= max, "min must smaller than max in clamp");
        assert!(
            min.notnan() & max.notnan(),
            "min and max should not be NaN in clamp"
        );
        self.apply_mut(|v| {
            if *v > max {
                // Note that NaN is excluded
                *v = max;
            } else if *v < min {
                *v = min;
            }
        })
    }
);

impl_map_inplace_nd!(
    zscore_inplace,
    /// Sandardize the array using zscore method on a given axis
    #[inline]
    pub fn zscore_inplace_1d(&mut self, stable: bool) {
    where
        T: Number,
        f64: AsPrimitive<T>
    }
    {
        let (mean, var) = self.meanvar_1d(stable);
        if var == 0. {
            self.apply_mut(|v| *v = 0.0.as_());
        } else if var.isnan() {
            self.apply_mut(|v| *v = f64::NAN.as_());
        } else {
            self.apply_mut(|v| *v = ((v.f64() - mean) / var.sqrt()).as_());
        }
    }
);

impl_map_inplace_nd!(
    winsorize_inplace,
    pub fn winsorize_inplace_1d(&mut self, method: WinsorizeMethod, method_params: Option<f64>, stable: bool)
    {
        where
        T: Number,
        f64: AsPrimitive<T>
    }
    {
        use WinsorizeMethod::*;
        match method {
            Quantile => {
                // default method is clip 1% and 99% quantile
                use super::QuantileMethod::*;
                let method_params = method_params.unwrap_or(0.01);
                let min = self.quantile_1d(method_params, Linear);
                let max = self.quantile_1d(1. - method_params, Linear);
                if min.notnan() && (min != max) {
                    self.clip_inplace_1d(min, max);
                }
            },
            Median => {
                // default method is clip median - 3 * mad, median + 3 * mad
                let method_params = method_params.unwrap_or(3.);
                let median = self.median_1d();
                if median.notnan() {
                    let mad = self.mapv(|v| (v.f64() - median).abs()).median_1d();
                    let min = median - method_params * mad;
                    let max = median + method_params * mad;
                    self.clip_inplace_1d(min, max);
                }
            },
            Sigma => {
                    // default method is clip mean - 3 * std, mean + 3 * std
                let method_params = method_params.unwrap_or(3.);
                let (mean, var) = self.meanvar_1d(stable);
                if mean.notnan() {
                    let std = var.sqrt();
                    let min = mean - method_params * std;
                    let max = mean + method_params * std;
                    self.clip_inplace_1d(min, max);
                }
            },
        }
    }
);
