use crate::prelude::*;
use ndarray::{Data, DataMut, Ix1, RawData};
use std::cmp::Ordering;
use std::mem::MaybeUninit;
use std::ops::Add;
use utils::kh_sum;

impl<T, S> ArrBase<S, Ix1>
where
    S: RawData<Elem = T>,
{
    // pub fn first_unwrap(&self) -> T
    // where
    //     S: Data,
    //     T: Clone,
    // {
    //     self.first().unwrap().clone()
    // }

    // pub fn last_unwrap(&self) -> T
    // where
    //     S: Data,
    //     T: Clone,
    // {
    //     self.last().unwrap().clone()
    // }

    #[inline]
    pub fn apply_mut_on<F>(&mut self, mut f: F, start: usize, end: usize)
    where
        S: DataMut,
        F: FnMut(&mut T),
    {
        assert!(end <= self.len()); // note start >= 0;
        for i in start..end {
            // safety: 0 <= i <= len-1
            let v = unsafe { self.uget_mut(i) };
            f(v);
        }
    }

    #[inline(always)]
    pub fn apply_mut<F>(&mut self, f: F)
    where
        S: DataMut,
        F: FnMut(&mut T),
    {
        self.apply_mut_on(f, 0, self.len());
    }

    /// Accumulate value using function f
    #[inline]
    pub fn fold<U, F>(&self, init: U, mut f: F) -> U
    where
        S: Data,
        F: FnMut(U, &T) -> U,
    {
        let mut acc = init;
        for i in 0..self.0.len() {
            let v = unsafe { self.uget(i) };
            acc = f(acc, v);
        }
        acc
    }

    /// Accumulate value using function f
    #[inline]
    pub fn fold_with<U, S2, T2, F>(&self, other: ArrBase<S2, Ix1>, init: U, mut f: F) -> U
    where
        S: Data,
        S2: Data<Elem = T2>,
        F: FnMut(U, &T, &T2) -> U,
    {
        assert_eq!(
            self.len(),
            other.len(),
            "lengh of the array and the other array must be equal when using fold with"
        );
        let mut acc = init;
        for i in 0..self.0.len() {
            let (v, vo) = unsafe { (self.uget(i), other.uget(i)) };
            acc = f(acc, v, vo);
        }
        acc
    }

    /// Accumulate valid value using function f
    #[inline]
    pub fn fold_valid<U, F>(&self, init: U, mut f: F) -> U
    where
        T: Number,
        S: Data,
        F: FnMut(U, &T) -> U,
    {
        let mut acc = init;
        for i in 0..self.0.len() {
            let v = unsafe { self.uget(i) };
            if v.notnan() {
                acc = f(acc, v);
            }
        }
        acc
    }

    /// Accumulate value using function f only when both elements are valid
    #[inline]
    pub fn fold_valid_with<U, S2, T2, F>(&self, other: &ArrBase<S2, Ix1>, init: U, mut f: F) -> U
    where
        S: Data,
        S2: Data<Elem = T2>,
        T: Number,
        T2: Number,
        F: FnMut(U, &T, &T2) -> U,
    {
        assert_eq!(
            self.len(),
            other.len(),
            "lengh of the array and the other array must be equal when using fold valid with"
        );
        let mut acc = init;
        for i in 0..self.0.len() {
            let (v, vo) = unsafe { (self.uget(i), other.uget(i)) };
            if v.notnan() && vo.notnan() {
                acc = f(acc, v, vo);
            }
        }
        acc
    }

    /// Accumulate value using function f, also return the number of
    /// valid elements.
    #[inline]
    pub fn n_fold_valid<U, F>(&self, init: U, mut f: F) -> (usize, U)
    where
        S: Data,
        T: Number,
        F: FnMut(U, &T) -> U,
    {
        let mut acc = init;
        let mut n = 0;
        for i in 0..self.0.len() {
            let v = unsafe { self.uget(i) };
            if v.notnan() {
                n += 1;
                acc = f(acc, v);
            }
        }
        (n, acc)
    }

    /// Accumulate value using function f, also return the number of
    /// valid elements.
    #[inline]
    pub fn n_fold_valid_with<U, S2, T2, F>(
        &self,
        other: &ArrBase<S2, Ix1>,
        init: U,
        mut f: F,
    ) -> (usize, U)
    where
        S: Data,
        S2: Data<Elem = T2>,
        T: Number,
        T2: Number,
        F: FnMut(U, &T, &T2) -> U,
    {
        assert!(self.len() == other.len());
        let mut acc = init;
        let mut n = 0;
        for i in 0..self.0.len() {
            let (v, vo) = unsafe { (self.uget(i), other.uget(i)) };
            if v.notnan() && vo.notnan() {
                n += 1;
                acc = f(acc, v, vo);
            }
        }
        (n, acc)
    }

    /// Call `f` on each element and accumulate,
    #[inline(always)]
    pub fn acc<U, F>(&self, init: U, mut f: F) -> U
    where
        U: Add<Output = U>,
        S: Data,
        F: FnMut(&T) -> U,
    {
        self.fold(init, |acc, v| acc + f(v))
    }

    /// Count the number of valid values and accumulate valid values using function f
    #[inline]
    pub fn n_acc_valid<U, F>(&self, init: U, mut f: F) -> (usize, U)
    where
        T: Number,
        U: Add<Output = U>,
        S: Data,
        F: FnMut(&T) -> U,
    {
        self.n_fold_valid(init, |acc, v| acc + f(v))
    }

    /// Only accumulate valild value using function f
    #[inline(always)]
    pub fn acc_valid<U, F>(&self, init: U, mut f: F) -> U
    where
        T: Number,
        U: Add<Output = U>,
        S: Data,
        F: FnMut(&T) -> U,
    {
        self.fold_valid(init, |acc, v| acc + f(v))
    }

    /// Count the number of valid values and accumulate value using Kahan summation
    #[inline(always)]
    pub fn stable_n_acc_valid<U, F>(&self, init: U, mut f: F) -> (usize, U)
    where
        S: Data,
        U: Number,
        T: Number,
        F: FnMut(&T) -> U,
    {
        let c = &mut U::zero();
        self.n_fold_valid(init, |acc, v| kh_sum(acc, f(v), c))
    }

    /// Only accumulate valild value using kahan summation
    #[inline(always)]
    pub fn stable_acc_valid<U, F>(&self, init: U, mut f: F) -> U
    where
        S: Data,
        U: Number,
        T: Number,
        F: FnMut(&T) -> U,
    {
        let c = &mut U::zero();
        self.fold_valid(init, |acc, v| kh_sum(acc, f(v), c))
    }

    /// Apply a function to each element
    #[inline(always)]
    pub fn apply<F>(&self, mut f: F)
    where
        S: Data,
        F: FnMut(&T),
    {
        self.fold((), move |(), elt| f(elt))
    }

    /// Apply a function to each element of the array and the other array
    #[inline]
    pub fn apply_with<S2, T2, F>(&self, other: ArrBase<S2, Ix1>, mut f: F)
    where
        S: Data,
        S2: Data<Elem = T2>,
        F: FnMut(&T, &T2),
    {
        self.fold_with(other, (), move |(), elt1, elt2| f(elt1, elt2));
    }

    /// Apply a function to each mut element
    #[inline]
    pub fn apply_mut_with<S2, T2, F>(&mut self, other: &ArrBase<S2, Ix1>, mut f: F)
    where
        S: DataMut,
        S2: Data<Elem = T2>,
        F: FnMut(&mut T, &T2),
    {
        // we must guarantee that the length of other is greater than
        // the length of self
        assert!(other.len() >= self.0.len());
        let len = self.len();
        if len == 0 {
            return;
        }
        for i in 0..len {
            // safety: 0 <= i <= len-1
            let (v, vo) = unsafe { (self.uget_mut(i), other.uget(i)) };
            f(v, vo);
        }
    }

    /// Apply a function to each valid element
    #[inline(always)]
    pub fn apply_valid<F>(&self, mut f: F)
    where
        T: Number,
        S: Data,
        F: FnMut(&T),
    {
        self.fold_valid((), move |(), elt| f(elt))
    }

    /// Apply a function to each valid element and return the number of valid values
    #[inline(always)]
    pub fn n_apply_valid<F>(&self, mut f: F) -> usize
    where
        T: Number,
        S: Data,
        F: FnMut(&T),
    {
        self.n_fold_valid((), move |(), elt| f(elt)).0
    }

    /// Apply a function to self and other only when both elements are valid
    #[inline(always)]
    pub fn apply_valid_with<S2, T2, F>(&self, other: &ArrBase<S2, Ix1>, mut f: F)
    where
        S: Data,
        S2: Data<Elem = T2>,
        T: Number,
        T2: Number,
        F: FnMut(&T, &T2),
    {
        self.fold_valid_with(other, (), move |(), elt1, elt2| f(elt1, elt2))
    }

    /// Apply a function to self and other only when both elements are valid
    #[inline(always)]
    pub fn n_apply_valid_with<S2, T2, F>(&self, other: &ArrBase<S2, Ix1>, mut f: F) -> usize
    where
        S: Data,
        S2: Data<Elem = T2>,
        T: Number,
        T2: Number,
        F: FnMut(&T, &T2),
    {
        self.n_fold_valid_with(other, (), move |(), elt1, elt2| f(elt1, elt2))
            .0
    }

    /// Apply function `f` on each element,
    /// if the function return `true`,
    /// `count += 1`, then return count
    #[inline(always)]
    pub fn count_by<F>(&self, mut f: F) -> i32
    where
        S: Data,
        F: FnMut(&T) -> bool,
    {
        self.fold(0, move |acc, elt| acc + (f(elt) as i32))
    }

    /// Apply a window function on self and set values returned by `f`,
    ///
    /// the first argument of `f` is the new value in the window.
    ///
    /// the second argument of `f` is the value to be removed in the window.
    #[inline]
    pub fn apply_window_to<U, S2, F>(&self, out: &mut ArrBase<S2, Ix1>, window: usize, mut f: F)
    where
        U: Clone,
        S: Data,
        S2: DataMut<Elem = MaybeUninit<U>>,
        F: FnMut(&T, Option<&T>) -> U,
    {
        let len = self.len();
        assert!(
            out.len() == len,
            "length of output array must equal to length of the array"
        );
        let window = window.min(len);
        if window == 0 {
            return;
        }
        // within the first window
        for i in 0..window - 1 {
            let (v, vo) = unsafe { (self.uget(i), out.uget_mut(i)) };
            // no value should be removed in the first window
            vo.write(f(v, None));
        }
        // other windows
        for (start, end) in (window - 1..len).enumerate() {
            // new valid value
            let (v_rm, v, vo) = unsafe { (self.uget(start), self.uget(end), out.uget_mut(end)) };
            vo.write(f(v, Some(v_rm)));
        }
    }

    #[inline]
    pub fn apply_window_with_to<U, S2, T2, S3, F>(
        &self,
        other: &ArrBase<S2, Ix1>,
        out: &mut ArrBase<S3, Ix1>,
        window: usize,
        mut f: F,
    ) where
        U: Clone,
        S: Data,
        S2: Data<Elem = T2>,
        S3: DataMut<Elem = MaybeUninit<U>>,
        F: FnMut(&T, &T2, Option<&T>, Option<&T2>) -> U,
    {
        let len = self.len();
        let window = window.min(len);
        if window == 0 {
            return;
        }
        // assert!(window > 0, "window must be greater than 0");
        assert!(
            out.len() == len,
            "length of output array must equal to length of the array"
        );
        // within the first window
        for i in 0..window - 1 {
            let (v1, v2, vo) = unsafe { (self.uget(i), other.uget(i), out.uget_mut(i)) };
            // no value should be removed in the first window
            vo.write(f(v1, v2, None, None));
        }
        // other windows
        for (start, end) in (window - 1..len).enumerate() {
            // new valid value
            let (v1_rm, v1, vo) = unsafe { (self.uget(start), self.uget(end), out.uget_mut(end)) };
            let (v2_rm, v2) = unsafe { (other.uget(start), other.uget(end)) };
            vo.write(f(v1, v2, Some(v1_rm), Some(v2_rm)));
        }
    }

    /// calculate mean value first and minus mean value for each element.
    #[inline]
    pub fn stable_apply_window_to<S2, F>(&self, out: &mut ArrBase<S2, Ix1>, window: usize, mut f: F)
    where
        T: Number,
        S: Data,
        S2: DataMut<Elem = MaybeUninit<f64>>,
        F: FnMut(f64, f64) -> f64,
    {
        let len = self.len();
        let window = window.min(len);
        if window == 0 {
            return;
        }
        // assert!(window > 0, "window must be greater than 0");
        assert!(
            out.len() == len,
            "length of output array must equal to length of the array"
        );
        let mean = self.mean_1d(0, true);
        // within the first window
        for i in 0..window - 1 {
            let (v, vo) = unsafe { (self.uget(i), out.uget_mut(i)) };
            // no value should be removed in the first window
            vo.write(f(v.f64() - mean, f64::NAN));
        }
        // other windows
        for (start, end) in (window - 1..len).enumerate() {
            // new valid value
            let (v_rm, v, vo) = unsafe { (self.uget(start), self.uget(end), out.uget_mut(end)) };
            let (v_rm, v) = (v_rm.f64() - mean, v.f64() - mean);
            vo.write(f(v, v_rm));
        }
    }

    #[inline]
    pub fn stable_apply_window_with_to<S2, T2, S3, F>(
        &self,
        other: &ArrBase<S2, Ix1>,
        out: &mut ArrBase<S3, Ix1>,
        window: usize,
        mut f: F,
    ) where
        T: Number,
        T2: Number,
        S: Data,
        S2: Data<Elem = T2>,
        S3: DataMut<Elem = MaybeUninit<f64>>,
        F: FnMut(f64, f64, f64, f64) -> f64,
    {
        let len = self.len();
        let window = window.min(len);
        if window == 0 {
            return;
        }
        // assert!(window > 0, "window must be greater than 0");
        assert!(
            out.len() == len,
            "length of output array must equal to length of the array"
        );
        let (mean1, mean2) = (self.mean_1d(0, true), other.mean_1d(0, true));
        // within the first window
        for i in 0..window - 1 {
            let (v1, v2, vo) = unsafe { (self.uget(i), other.uget(i), out.uget_mut(i)) };
            // no value should be removed in the first window
            vo.write(f(v1.f64() - mean1, v2.f64() - mean2, f64::NAN, f64::NAN));
        }
        // other windows
        for (start, end) in (window - 1..len).enumerate() {
            // new valid value
            let (v1_rm, v1, vo) = unsafe { (self.uget(start), self.uget(end), out.uget_mut(end)) };
            let (v2_rm, v2) = unsafe { (other.uget(start), other.uget(end)) };
            let (v1_rm, v1) = (v1_rm.f64() - mean1, v1.f64() - mean1);
            let (v2_rm, v2) = (v2_rm.f64() - mean2, v2.f64() - mean2);
            vo.write(f(v1, v2, v1_rm, v2_rm));
        }
    }

    /// Apply a reverse window function on self and set values returned by `f`,
    ///
    /// the first argument of `f` is the new value in the window.
    ///
    /// the second argument of `f` is the value to be removed in the window.
    #[inline]
    pub fn apply_revwindow_to<U, S2, F>(&self, out: &mut ArrBase<S2, Ix1>, window: usize, mut f: F)
    where
        U: Clone,
        S: Data,
        S2: DataMut<Elem = MaybeUninit<U>>,
        F: FnMut(&T, Option<&T>) -> U,
    {
        let len = self.len();
        let window = window.min(len);
        if window == 0 {
            return;
        }
        // assert!(window > 0, "window must be greater than 0");
        assert!(
            out.len() == len,
            "length of output array must equal to length of the array"
        );
        // other windows
        for (start, end) in (window - 1..len).enumerate() {
            // new valid value
            let (v, v_rm, vo) = unsafe { (self.uget(start), self.uget(end), out.uget_mut(start)) };
            vo.write(f(v, Some(v_rm)));
        }
        // within the last window
        for i in len - window + 1..len {
            let (v, vo) = unsafe { (self.uget(i), out.uget_mut(i)) };
            // no value should be removed in the first window
            vo.write(f(v, None));
        }
    }

    /// sort 1d array using a compare function, but might not preserve the order of equal elements.
    #[inline]
    pub fn sort_unstable_by<F>(&mut self, compare: F)
    where
        T: Clone,
        S: DataMut,
        F: FnMut(&T, &T) -> Ordering,
    {
        if self.0.is_standard_layout() {
            let slice = self.as_slice_mut().unwrap();
            slice.sort_unstable_by(compare);
        } else {
            let mut out_c = self.to_owned();
            let slice = out_c.as_slice_mut().unwrap();
            slice.sort_unstable_by(compare);
            out_c.clone_to(self);
        }
    }
}
