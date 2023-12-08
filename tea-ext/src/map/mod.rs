mod impl_1d;
mod impl_arrok;
mod impl_inplace;
mod impl_string;

#[cfg(feature = "lazy")]
mod impl_lazy;
#[cfg(feature = "time")]
mod impl_time;

pub use impl_1d::MapExt1d;
pub use impl_arrok::ArrOkExt;
pub use impl_inplace::*;
pub use impl_string::StringExt;

#[cfg(feature = "lazy")]
pub use impl_lazy::*;
#[cfg(feature = "time")]
pub use impl_time::TimeExt;

use ndarray::{Data, DataMut, Dimension, Ix1, ShapeBuilder, Zip};
use std::{fmt::Debug, mem::MaybeUninit};
use tea_core::prelude::*;

#[cfg(feature = "lazy")]
use lazy::Expr;

// #[cfg(feature = "groupby")]
// use crate::hash::TpHash;
// #[cfg(feature = "groupby")]
// use ahash::RandomState;
// #[cfg(feature = "groupby")]
// use std::hash::Hash;

#[ext_trait(lazy = "view")]
impl<T, S: Data<Elem = T>, D: Dimension> MapExt for ArrBase<S, D> {
    fn is_nan(&self) -> ArrD<bool>
    where
        T: GetNone,
    {
        self.map(|v| v.is_none()).to_dimd()
    }

    fn not_nan(&self) -> ArrD<bool>
    where
        T: GetNone,
    {
        self.map(|v| !v.is_none()).to_dimd()
    }

    #[lazy_exclude]
    fn is_in(&self, other: &[T]) -> ArrD<bool>
    where
        T: PartialEq,
    {
        self.map(|v| other.contains(v)).to_dimd()
    }

    #[lazy_exclude]
    fn filter<SO>(&self, mask: &ArrBase<SO, Ix1>, axis: i32, par: bool) -> ArrD<T>
    where
        T: Default + Send + Sync + Clone,
        D: Dimension,
        SO: Data<Elem = bool> + Send + Sync,
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
        out.to_dimd()
    }

    /// Take value on a given axis and clone to a new array,
    ///
    /// if you want to along axis, select arbitrary subviews corresponding to indices and and
    /// copy them into a new array, use select instead.
    ///
    /// # Safety
    ///
    /// The index in `slc` must be correct.
    #[lazy_exclude]
    unsafe fn take_clone_unchecked<SO>(
        &self,
        slc: ArrBase<SO, Ix1>,
        axis: i32,
        par: bool,
    ) -> ArrD<T>
    where
        T: Clone + Default + Debug + Send + Sync,
        D: Dimension,
        SO: Data<Elem = usize> + Send + Sync,
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
        out.to_dimd()
    }

    /// Take value on a given axis and clone to a new array,
    ///
    /// # Safety
    ///
    /// The index in `slc` must be correct.
    #[lazy_exclude]
    unsafe fn take_option_clone_unchecked<SO>(
        &self,
        slc: ArrBase<SO, Ix1>,
        axis: i32,
        par: bool,
    ) -> ArrD<T>
    where
        T: Clone + Default + GetNone + Send + Sync,
        D: Dimension,
        SO: Data<Elem = OptUsize> + Send + Sync,
    {
        let f_flag = !self.is_standard_layout();
        let axis = self.norm_axis(axis);
        let mut new_dim = self.raw_dim();
        new_dim.slice_mut()[axis.index()] = slc.len();
        let shape = new_dim.into_shape().set_f(f_flag);
        let mut out = Arr::<T, D>::uninit(shape);
        let f = |x_1d: ArrView1<T>, out_1d: ArrViewMut1<MaybeUninit<T>>| {
            x_1d.wrap()
                .take_option_clone_1d_unchecked(out_1d, slc.view())
        };
        // we should not use apply_along_axis here because it won't do anything when this axis is empty
        let ndim = self.ndim();
        if ndim == 1 {
            let view = self.view().to_dim1().unwrap();
            f(view, out.view_mut().to_dim1().unwrap());
        } else {
            let arr_zip = Zip::from(self.lanes(axis)).and(out.lanes_mut(axis));
            if !par || (ndim == 1) {
                // non-parallel
                arr_zip.for_each(|a, b| f(a.wrap(), b.wrap()));
            } else {
                // parallel
                arr_zip.par_for_each(|a, b| f(a.wrap(), b.wrap()));
            }
        }
        unsafe { out.assume_init() }.to_dimd()
        // out
    }

    #[lazy_exclude]
    pub fn put_mask<SO, S3, D2, D3>(
        &mut self,
        mask: &ArrBase<SO, D2>,
        value: &ArrBase<S3, D3>,
        axis: i32,
        par: bool,
    ) -> TpResult<()>
    where
        T: Clone + Send + Sync,
        D2: Dimension,
        D3: Dimension,
        S: DataMut<Elem = T>,
        SO: Data<Elem = bool>,
        S3: Data<Elem = T>,
    {
        let axis = self.norm_axis(axis);
        let value = if self.ndim() == value.ndim() && self.shape() == value.shape() {
            value.view().to_dim::<D>().unwrap()
        } else if let Some(value) = value.broadcast(self.raw_dim()) {
            value
        } else {
            // the number of value array's elements are equal to the number of true values in mask
            let mask = mask
                .view()
                .to_dim1()
                .map_err(|e| StrError::from(format!("{e}")))?;
            // .expect("mask should be dim1 when set value to masked data");
            let value = value
                .view()
                .to_dim1()
                .map_err(|e| StrError::from(format!("{e}")))?;
            // .expect("value should be dim1 when set value to masked data");
            let true_num = mask.count_v_1d(true) as usize;
            if true_num != value.len_of(axis) {
                return Err(StrError::from(
                    "number of value are not equal to number of true mask",
                ));
            }
            let ndim = self.ndim();
            if par && (ndim > 1) {
                Zip::from(self.lanes_mut(axis))
                    .par_for_each(|x_1d| x_1d.wrap().put_mask_1d(&mask, &value));
            } else {
                Zip::from(self.lanes_mut(axis))
                    .for_each(|x_1d| x_1d.wrap().put_mask_1d(&mask, &value));
            };
            return Ok(());
        };
        let mask = if self.ndim() == mask.ndim() && self.shape() == mask.shape() {
            mask.view().to_dim::<D>().unwrap()
        } else {
            mask.broadcast(self.raw_dim()).unwrap()
        };
        if !par {
            Zip::from(&mut self.0)
                .and(&mask.0)
                .and(&value.0)
                .for_each(|a, m, v| {
                    if *m {
                        *a = v.clone()
                    }
                });
        } else {
            Zip::from(&mut self.0)
                .and(&mask.0)
                .and(&value.0)
                .par_for_each(|a, m, v| {
                    if *m {
                        *a = v.clone()
                    }
                });
        }
        Ok(())
    }

    #[cfg(feature = "agg")]
    #[teapy(type = "numeric")]
    fn arg_partition(
        &self,
        mut kth: usize,
        sort: bool,
        rev: bool,
        axis: i32,
        par: bool,
    ) -> ArrD<i32>
    where
        T: Number + Send + Sync,
        D: Dimension,
    {
        let f_flag = !self.is_standard_layout();
        let mut new_dim = self.raw_dim();
        let axis = self.norm_axis(axis);
        if kth >= new_dim.slice_mut()[axis.index()] {
            kth = new_dim.slice_mut()[axis.index()] - 1
        }
        new_dim.slice_mut()[axis.index()] = kth + 1;
        let shape = new_dim.into_shape().set_f(f_flag);
        let mut out = Arr::<i32, D>::default(shape);
        let mut out_wr = out.view_mut();
        self.apply_along_axis(&mut out_wr, axis, par, |x_1d, out_1d| {
            x_1d.arg_partition_1d(out_1d, kth, sort, rev)
        });
        out.to_dimd()
    }

    #[cfg(feature = "agg")]
    #[teapy(type = "numeric")]
    fn partition(&self, mut kth: usize, sort: bool, rev: bool, axis: i32, par: bool) -> ArrD<T>
    where
        T: Number + Send + Sync,
        D: Dimension,
    {
        let f_flag = !self.is_standard_layout();
        let mut new_dim = self.raw_dim();
        let axis = self.norm_axis(axis);
        if kth >= new_dim.slice_mut()[axis.index()] {
            kth = new_dim.slice_mut()[axis.index()] - 1
        }
        new_dim.slice_mut()[axis.index()] = kth + 1;
        let shape = new_dim.into_shape().set_f(f_flag);
        let mut out = Arr::<T, D>::default(shape);
        let mut out_wr = out.view_mut();
        self.apply_along_axis(&mut out_wr, axis, par, |x_1d, out_1d| {
            x_1d.partition_1d(out_1d, kth, sort, rev)
        });
        out.to_dimd()
    }
}

#[arr_map_ext(lazy = "view", type = "numeric")]
impl<T, S: Data<Elem = T>, D: Dimension> MapExtNd for ArrBase<S, D> {
    fn cumsum<SO>(&self, out: &mut ArrBase<SO, Ix1>, stable: bool) -> T
    where
        SO: DataMut<Elem = MaybeUninit<T>>,
        T: Number,
    {
        let mut sum = T::zero();
        if !stable {
            out.apply_mut_with(&self.as_dim1(), |vo, v| {
                if v.notnan() {
                    sum += *v;
                    vo.write(sum);
                } else {
                    vo.write(T::nan());
                }
            });
        } else {
            let c = &mut T::zero();
            out.apply_mut_with(&self.as_dim1(), |vo, v| {
                if v.notnan() {
                    sum.kh_sum(*v, c);
                    vo.write(sum);
                } else {
                    vo.write(T::nan());
                }
            });
        }
    }

    fn cumprod<SO>(&self, out: &mut ArrBase<SO, Ix1>) -> T
    where
        SO: DataMut<Elem = MaybeUninit<T>>,
        T: Number,
    {
        let mut prod = T::one();
        out.apply_mut_with(&self.as_dim1(), |vo, v| {
            if v.notnan() {
                prod *= *v;
                vo.write(prod);
            } else {
                vo.write(T::nan());
            }
        });
    }

    fn argsort<SO>(&self, out: &mut ArrBase<SO, Ix1>, rev: bool) -> i32
    where
        SO: DataMut<Elem = MaybeUninit<i32>>,
        T: Number,
    {
        let arr = self.as_dim1();
        assert!(out.len() >= arr.len());
        let mut i = 0;
        out.apply_mut(|v| {
            v.write(i);
            i += 1;
        }); // set elements of out array
        if !rev {
            out.sort_unstable_by(|a, b| {
                let (va, vb) = unsafe {
                    (
                        *arr.uget((a.assume_init_read()) as usize),
                        *arr.uget((b.assume_init_read()) as usize),
                    )
                }; // safety: out不超过self的长度
                va.nan_sort_cmp(&vb)
            });
        } else {
            out.sort_unstable_by(|a, b| {
                let (va, vb) = unsafe {
                    (
                        *arr.uget((a.assume_init_read()) as usize),
                        *arr.uget((b.assume_init_read()) as usize),
                    )
                }; // safety: out不超过self的长度
                va.nan_sort_cmp_rev(&vb)
            });
        }
    }

    /// rank the array in a given axis
    #[allow(unused_assignments)]
    fn rank<SO>(&self, out: &mut ArrBase<SO, Ix1>, pct: bool, rev: bool) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T: Number,
    {
        let arr = self.as_dim1();
        let len = arr.len();
        assert!(
            out.len() >= len,
            "the length of the input array not equal to the length of the output array"
        );
        if len == 0 {
            return;
        } else if len == 1 {
            // safety: out.len() == self.len() == 1
            // unsafe { *out.uget_mut(0) = 1. };
            return unsafe {
                out.uget_mut(0).write(1.);
            };
        }
        // argsort at first
        let mut idx_sorted = Arr1::from_iter(0..len);
        if !rev {
            idx_sorted.sort_unstable_by(|a, b| {
                let (va, vb) = unsafe { (*arr.uget(*a), *arr.uget(*b)) }; // safety: out不超过self的长度
                va.nan_sort_cmp(&vb)
            });
        } else {
            idx_sorted.sort_unstable_by(|a, b| {
                let (va, vb) = unsafe { (*arr.uget(*a), *arr.uget(*b)) }; // safety: out不超过self的长度
                va.nan_sort_cmp_rev(&vb)
            });
        }

        // if the smallest value is nan then all the elements are nan
        if unsafe { *arr.uget(*idx_sorted.uget(0)) }.isnan() {
            return out.apply_mut(|v| {
                v.write(f64::NAN);
            });
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
                    let (v, v1) = (*arr.uget(idx), *arr.uget(idx1)); // 下一个值，safe because idx1 < arr.len()
                    if v1.isnan() {
                        // 下一个值是nan，说明后面的值全是nan
                        sum_rank += cur_rank;
                        cur_rank += 1;
                        for j in 0..repeat_num {
                            // safe because i >= repeat_num
                            let out = out.uget_mut(*idx_sorted.uget(i - j));
                            out.write(sum_rank.f64() / repeat_num.f64());
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
                        let out = out.uget_mut(idx);
                        out.write(cur_rank as f64);
                        cur_rank += 1;
                    } else {
                        // 当前元素是最后一个重复元素
                        sum_rank += cur_rank;
                        cur_rank += 1;
                        for j in 0..repeat_num {
                            // safe because i >= repeat_num
                            let out = out.uget_mut(*idx_sorted.uget(i - j));
                            out.write(sum_rank.f64() / repeat_num.f64());
                        }
                        sum_rank = 0; // rank和归零
                        repeat_num = 1; // 重复计数归一
                    }
                }
                if nan_flag {
                    for i in idx..len {
                        let out = out.uget_mut(*idx_sorted.uget(i));
                        out.write(f64::NAN);
                    }
                } else {
                    sum_rank += cur_rank;
                    for i in len - repeat_num..len {
                        // safe because repeat_num <= len
                        let out = out.uget_mut(*idx_sorted.uget(i));
                        out.write(sum_rank.f64() / repeat_num.f64());
                    }
                }
            }
        } else {
            let notnan_count = arr.count_notnan_1d() as usize;
            unsafe {
                for i in 0..len - 1 {
                    // safe because i_max = arr.len()-2 and arr.len() >= 2
                    (idx, idx1) = (*idx_sorted.uget(i), *idx_sorted.uget(i + 1));
                    let (v, v1) = (*arr.uget(idx), *arr.uget(idx1)); // 下一个值，safe because idx1 < arr.len()
                    if v1.isnan() {
                        // 下一个值是nan，说明后面的值全是nan
                        sum_rank += cur_rank;
                        cur_rank += 1;
                        for j in 0..repeat_num {
                            // safe because i >= repeat_num
                            let out = out.uget_mut(*idx_sorted.uget(i - j));
                            out.write(sum_rank.f64() / (repeat_num * notnan_count).f64());
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
                        let out = out.uget_mut(idx);
                        out.write(cur_rank.f64() / notnan_count.f64());
                        cur_rank += 1;
                    } else {
                        // 当前元素是最后一个重复元素
                        sum_rank += cur_rank;
                        cur_rank += 1;
                        for j in 0..repeat_num {
                            // safe because i >= repeat_num
                            let out = out.uget_mut(*idx_sorted.uget(i - j));
                            out.write(sum_rank.f64() / (repeat_num * notnan_count).f64());
                        }
                        sum_rank = 0; // rank和归零
                        repeat_num = 1; // 重复计数归一
                    }
                }
                if nan_flag {
                    for i in idx..len {
                        let out = out.uget_mut(*idx_sorted.uget(i));
                        out.write(f64::NAN);
                    }
                } else {
                    sum_rank += cur_rank;
                    for i in len - repeat_num..len {
                        // safe because repeat_num <= len
                        let out = out.uget_mut(*idx_sorted.uget(i));
                        out.write(sum_rank.f64() / (repeat_num * notnan_count).f64());
                    }
                }
            }
        }
    }

    /// Split values in several group by size.
    #[inline]
    fn split_group<SO>(&self, out: &mut ArrBase<SO, Ix1>, group: usize, rev: bool) -> i32
    where
        SO: DataMut<Elem = MaybeUninit<i32>>,
        T: Number,
    {
        let arr = self.as_dim1();
        let valid_count = arr.count_notnan_1d();
        let mut rank = Arr1::<f64>::uninit(arr.raw_dim());
        arr.rank_1d(&mut rank, false, rev);
        let rank = unsafe { rank.assume_init() };
        out.apply_mut_with(&rank, |vo, v| {
            vo.write(((*v * group.f64()) / valid_count.f64()).ceil() as i32);
        })
    }

    fn pct_change<SO>(&self, out: &mut ArrBase<SO, Ix1>, n: i32) -> f64
    where
        SO: DataMut<Elem = MaybeUninit<f64>>,
        T: Number,
    {
        if self.is_empty() {
            return;
        }
        let arr = self.as_dim1();
        if n == 0 {
            out.apply_mut_with(&arr, |vo, _| {
                vo.write(0.);
            });
        } else if n.unsigned_abs() as usize > arr.len() - 1 {
            out.apply_mut_with(&arr, |vo, _| {
                vo.write(f64::NAN);
            });
        } else if n > 0 {
            arr.apply_window_to(out, n.unsigned_abs() as usize + 1, |v, v_rm| {
                if let Some(v_rm) = v_rm {
                    let v_rm = v_rm.f64();
                    if v_rm != 0. {
                        v.f64() / v_rm.f64() - 1.
                    } else {
                        f64::NAN
                    }
                } else {
                    f64::NAN
                }
            });
        } else {
            arr.apply_revwindow_to(out, n.unsigned_abs() as usize + 1, |v, v_rm| {
                if let Some(v_rm) = v_rm {
                    let v_rm = v_rm.f64();
                    if v_rm != 0. {
                        v.f64() / v_rm.f64() - 1.
                    } else {
                        f64::NAN
                    }
                } else {
                    f64::NAN
                }
            });
        }
    }
}

#[cfg(feature = "lazy")]
#[ext_trait(lazy_only, lazy = "f64_func", type = "numeric")]
impl<'a> F64FuncExt for Expr<'a> {
    fn sqrt(&self) {}

    fn cbrt(&self) {}

    fn ln(&self) {}

    fn ln_1p(&self) {}

    fn log2(&self) {}

    fn log10(&self) {}

    fn exp(&self) {}

    fn exp_m1(&self) {}

    fn exp2(&self) {}

    fn acos(&self) {}

    fn asin(&self) {}

    fn atan(&self) {}

    fn sin(&self) {}

    fn cos(&self) {}

    fn tan(&self) {}

    fn ceil(&self) {}

    fn floor(&self) {}

    fn fract(&self) {}

    fn trunc(&self) {}

    fn is_finite(&self) {}

    fn is_infinite(&self) {}

    fn log(&self, _base: f64) {}
}
