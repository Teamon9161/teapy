use ndarray::{Data, DataMut, Ix1};
use std::{fmt::Debug, iter::zip, mem::MaybeUninit};
use tea_core::prelude::*;
// use tea_core::utils::CollectTrustedToVec;

#[ext_trait]
impl<T, S: Data<Elem = T>> MapExt1d for ArrBase<S, Ix1> {
    /// Remove NaN values in 1d array.
    #[inline]
    fn dropna_1d(self) -> Arr1<T>
    where
        T: IsNone,
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
    #[inline]
    fn arg_partition_1d<SO>(&self, mut out: ArrBase<SO, Ix1>, kth: usize, sort: bool, rev: bool)
    where
        T: IsNone + Copy + Send + Sync,
        T::Inner: Number,
        SO: DataMut<Elem = MaybeUninit<i32>>,
    {
        let arr = self.as_dim1().0;
        let out_iter = arr.varg_partition(kth, sort, rev);
        out.0.view_mut().write_trust_iter(out_iter).unwrap();
    }

    /// sort: whether to sort the result by the size of the element
    #[cfg(feature = "agg")]
    #[inline]
    fn partition_1d<SO>(&self, mut out: ArrBase<SO, Ix1>, kth: usize, sort: bool, rev: bool)
    where
        T: IsNone + Send + Sync,
        T::Inner: Number,
        SO: DataMut<Elem = MaybeUninit<T>>,
    {
        let arr = self.as_dim1().0;
        let out_iter = arr.vpartition(kth, sort, rev);
        out.0.view_mut().write_trust_iter(out_iter).unwrap();
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
            .titer()
            .map(|idx| self.uget(idx).clone())
            .collect_trusted_to_vec();
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
        T: Clone + IsNone,
        SO: DataMut<Elem = MaybeUninit<T>>,
        S3: Data<Elem = Option<usize>> + Send + Sync,
    {
        let value = slc
            .titer()
            .map(|idx| {
                if let Some(idx) = idx {
                    self.uget(idx).clone()
                } else {
                    T::none()
                }
            })
            .collect_trusted_to_vec();

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
