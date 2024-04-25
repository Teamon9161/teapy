use ndarray::{Data, DataMut, Ix1};
use std::{fmt::Debug, iter::zip, mem::MaybeUninit};
use tea_core::prelude::*;
use tea_core::utils::CollectTrustedToVec;

#[ext_trait]
impl<T, S: Data<Elem = T>> MapExt1d for ArrBase<S, Ix1> {
    /// Remove NaN values in 1d array.
    #[inline]
    fn dropna_1d(self) -> Arr1<T>
    where
        T: GetNone,
    {
        Arr1::from_iter(self.into_iter().filter(|v| !v.is_none()))
    }

    #[allow(clippy::unnecessary_filter_map)]
    fn get_sorted_unique_idx_1d(&self, keep: String) -> Arr1<usize>
    where
        T: PartialEq,
    {
        let len = self.len();
        if len == 0 {
            return Arr1::from_vec(vec![]);
        }
        let out = if &keep == "first" {
            let mut value = unsafe { self.uget(0) };
            vec![0]
                .into_iter()
                .chain((1..len).filter_map(|i| {
                    let v = unsafe { self.uget(i) };
                    if v != value {
                        value = v;
                        Some(i)
                    } else {
                        None
                    }
                }))
                .collect::<Vec<_>>()
        } else if &keep == "last" {
            let mut value = unsafe { self.uget(0) };
            let lst_idx = len - 1;
            let mut out = (0..lst_idx)
                .filter_map(|i| {
                    let v = unsafe { self.uget(i + 1) };
                    if v != value {
                        value = v;
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            if unsafe { self.uget(lst_idx) } == value {
                out.push(lst_idx)
            }
            out
        } else {
            panic!("keep must be either first or last")
        };
        Arr1::from_vec(out)
    }

    #[allow(clippy::unnecessary_filter_map)]
    fn sorted_unique_1d(&self) -> Arr1<T>
    where
        T: PartialEq + Clone,
    {
        let len = self.len();
        if len == 0 {
            return Arr1::from_vec(vec![]);
        }
        let mut value = unsafe { self.uget(0) };
        let out = vec![value.clone()]
            .into_iter()
            .chain((1..len).filter_map(|i| {
                let v = unsafe { self.uget(i) };
                if v != value {
                    value = v;
                    Some(v.clone())
                } else {
                    None
                }
            }))
            .collect::<Vec<_>>();
        Arr1::from_vec(out)
    }

    /// return -1 if there are not enough valid elements
    /// sort: whether to sort the result by the size of the element
    #[cfg(feature = "agg")]
    fn arg_partition_1d<SO>(&self, mut out: ArrBase<SO, Ix1>, kth: usize, sort: bool, rev: bool)
    where
        T: Number,
        SO: DataMut<Elem = i32>,
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

    /// sort: whether to sort the result by the size of the element
    #[cfg(feature = "agg")]
    fn partition_1d<SO>(&self, mut out: ArrBase<SO, Ix1>, kth: usize, sort: bool, rev: bool)
    where
        T: Number,
        SO: DataMut<Elem = T>,
    {
        let n = self.count_notnan_1d() as usize;
        if n <= kth + 1 {
            if !sort {
                out.apply_mut_with(self, |vo, v| *vo = *v);
            } else {
                let mut arr = self.to_owned(); // clone the array
                if !rev {
                    arr.sort_unstable_by(|a, b| a.nan_sort_cmp(b));
                } else {
                    arr.sort_unstable_by(|a, b| a.nan_sort_cmp_rev(b));
                }
                out.apply_mut_with(&arr, |vo, v| *vo = *v);
            }
            return;
        }
        let mut out_c = self.0.to_owned().into_raw_vec(); // clone the array
        let sort_func = if !rev {
            T::nan_sort_cmp
        } else {
            T::nan_sort_cmp_rev
        };
        out_c.select_nth_unstable_by(kth, sort_func);
        out_c.truncate(kth + 1);
        if sort {
            out_c.sort_unstable_by(sort_func)
        }
        out.apply_mut_with(&Arr1::from_vec(out_c), |vo, v| *vo = *v);
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
    unsafe fn take_clone_1d_unchecked<SO, S3>(
        &self,
        mut out: ArrBase<SO, Ix1>,
        slc: ArrBase<S3, Ix1>,
    ) where
        T: Clone + Debug,
        SO: DataMut<Elem = T>,
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
    unsafe fn take_option_clone_1d_unchecked<SO, S3>(
        &self,
        mut out: ArrBase<SO, Ix1>,
        slc: ArrBase<S3, Ix1>,
    ) where
        T: Clone + GetNone,
        SO: DataMut<Elem = MaybeUninit<T>>,
        S3: Data<Elem = OptUsize> + Send + Sync,
    {
        let value = slc
            .iter()
            .map(|idx| {
                if let Some(idx) = Into::<Option<usize>>::into(*idx) {
                    self.uget(idx).clone()
                } else {
                    T::none()
                }
            })
            .collect_trusted();

        let value_view = ArrView1::<_>::from_ref_vec(out.raw_dim(), &value);
        out.apply_mut_with(&value_view, |vo, v| {
            vo.write(v.clone());
        });
    }

    #[inline]
    fn filter_1d<SO, S3>(&self, mut out: ArrBase<SO, Ix1>, mask: ArrBase<S3, Ix1>)
    where
        T: Clone,
        SO: DataMut<Elem = T>,
        S3: Data<Elem = bool> + Send + Sync,
    {
        zip(self, mask)
            .filter(|(_v, m)| *m)
            .zip(out.iter_mut())
            .for_each(|((v, _m), o)| *o = v.clone())
    }

    #[inline]
    fn put_mask_1d<SO, S3>(&mut self, mask: &ArrBase<SO, Ix1>, value: &ArrBase<S3, Ix1>)
    where
        T: Clone,
        S: DataMut<Elem = T>,
        S3: Data<Elem = T> + Send + Sync,
        SO: Data<Elem = bool> + Send + Sync,
    {
        zip(self, mask)
            .filter(|(_v, m)| **m)
            .zip(value)
            .for_each(|((a, _m), v)| *a = v.clone())
    }
}
