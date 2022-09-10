use crate::algos::kh_sum;
use crate::datatype::Number;
use ndarray::{ArrayView1, ArrayViewMut1};
use std::cmp::Ordering;
use std::ops::{AddAssign, Deref, DerefMut};

#[macro_use]
mod macros;
pub mod agg;
pub mod compare;
pub mod corr;
mod prelude;
pub mod window;

define_arrview!(ArrView1, ArrayView1);
define_arrview!(ArrViewMut1, ArrayViewMut1);

impl_arrview!([ArrView1, ArrViewMut1], {
    /// Accumulate value using function f
    #[inline]
    pub fn fold<U, F>(&self, init: U, mut f: F) -> U
    where
        F: FnMut(U, &T) -> U,
    {
        let mut acc = init;
        for i in 0..self.0.len() {
            let v = unsafe { self.uget(i) };
            acc = f(acc, v);
        }
        acc
    }

    /// Apply a function to each element
    #[inline]
    pub fn apply<F>(&self, mut f: F)
    where
        F: FnMut(&T),
    {
        self.fold((), move |(), elt| f(elt))
    }

    /// Apply function `f` on each element,
    /// if the function return `true`,
    /// `count += 1`, then return count
    #[inline]
    pub fn count_by<F>(&self, mut f: F) -> usize
    where
        F: FnMut(&T) -> bool,
    {
        self.fold(0, move |acc, elt| acc + (f(elt) as usize))
    }
});

impl_arrview!([ArrView1, ArrViewMut1], Number, {
    /// Accumulate value using function f
    #[inline]
    pub fn fold_valid<U, F>(&self, init: U, mut f: F) -> U
    where
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
    pub fn fold_valid_with<U, S, F>(&self, other: &ArrView1<S>, init: U, mut f: F) -> U
    where
        S: Number,
        F: FnMut(U, &T, &S) -> U,
    {
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
    pub fn n_fold_valid_with<U, S, F>(&self, other: &ArrView1<S>, init: U, mut f: F) -> (usize, U)
    where
        S: Number,
        F: FnMut(U, &T, &S) -> U,
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
    #[inline]
    pub fn acc<U, F>(&self, init: U, mut f: F) -> U
    where
        U: Number,
        F: FnMut(&T) -> U,
    {
        let mut acc = init;
        for i in 0..self.0.len() {
            let v = unsafe { self.uget(i) };
            acc += f(v);
        }
        acc
    }

    /// Count the number of valid values and accumulate valid values using function f
    #[inline]
    pub fn n_acc_valid<U, F>(&self, init: U, mut f: F) -> (usize, U)
    where
        U: AddAssign,
        F: FnMut(&T) -> U,
    {
        let mut acc = init;
        let mut n = 0;
        for i in 0..self.0.len() {
            let v = unsafe { self.uget(i) };
            if v.notnan() {
                n += 1;
                acc += f(v);
            }
        }
        (n, acc)
    }

    /// Count the number of valid values and accumulate value using Kahan summation
    #[inline]
    pub fn stable_n_acc_valid<U, F>(&self, init: U, mut f: F) -> (usize, U)
    where
        U: Number,
        F: FnMut(&T) -> U,
    {
        let mut acc = init;
        let mut c = U::zero();
        let mut n = 0;
        for i in 0..self.0.len() {
            let v = unsafe { self.uget(i) };
            if v.notnan() {
                n += 1;
                acc = kh_sum(acc, f(v), &mut c)
            }
        }
        (n, acc)
    }

    /// Only accumulate valild value using function f
    #[inline]
    pub fn acc_valid<U, F>(&self, init: U, f: F) -> U
    where
        U: AddAssign,
        F: FnMut(&T) -> U,
    {
        self.n_acc_valid(init, f).1
    }

    /// Only accumulate valild value using kahan summation
    #[inline]
    pub fn stable_acc_valid<U, F>(&self, init: U, f: F) -> U
    where
        U: Number,
        F: FnMut(&T) -> U,
    {
        self.stable_n_acc_valid(init, f).1
    }

    /// Apply a function to each valid element
    #[inline]
    pub fn apply_valid<F>(&self, mut f: F)
    where
        F: FnMut(&T),
    {
        self.fold_valid((), move |(), elt| f(elt))
    }

    /// Apply a function to self and other only when both elements are valid
    #[inline]
    pub fn apply_valid_with<S, F>(&self, other: &ArrView1<S>, mut f: F)
    where
        S: Number,
        F: FnMut(&T, &S),
    {
        self.fold_valid_with(other, (), move |(), elt1, elt2| f(elt1, elt2))
    }

    /// Apply a function to each valid element and return the number of valid values
    #[inline]
    pub fn n_apply_valid<F>(&self, mut f: F) -> usize
    where
        F: FnMut(&T),
    {
        self.n_fold_valid((), move |(), elt| f(elt)).0
    }

    /// Apply a function to self and other only when both elements are valid
    ///  and return the number of valid values
    #[inline]
    pub fn n_apply_valid_with<S, F>(&self, other: &ArrView1<S>, mut f: F) -> usize
    where
        S: Number,
        F: FnMut(&T, &S),
    {
        self.n_fold_valid_with(other, (), move |(), elt1, elt2| f(elt1, elt2))
            .0
    }

    /// Apply a window function on self and set values returned by `f`,
    ///
    /// the first argument of `f` is the new value in the window.
    ///
    /// the second argument of `f` is the value to be removed in the window.
    #[inline]
    pub fn apply_valid_window_to<U, F>(&self, out: &mut ArrViewMut1<U>, window: usize, mut f: F)
    where
        U: Copy,
        F: FnMut(&T, Option<&T>) -> U,
    {
        let len = self.len();
        assert!(
            window <= len,
            "window should not be greater than the length of the array"
        );
        assert!(
            out.len() == len,
            "length of output array must equal to length of the array"
        );
        // within the first window
        for i in 0..window - 1 {
            let (v, vo) = unsafe { (self.uget(i), out.uget_mut(i)) };
            // no value should be removed in the first window
            *vo = f(v, None);
        }
        // other windows
        for (start, end) in (window - 1..len).enumerate() {
            // new valid value
            let (v_rm, v, vo) = unsafe { (self.uget(start), self.uget(end), out.uget_mut(end)) };
            *vo = f(v, Some(v_rm));
        }
    }

    #[inline]
    pub fn apply_valid_window_with_to<U, S, F>(
        &self,
        other: &ArrView1<S>,
        out: &mut ArrViewMut1<U>,
        window: usize,
        mut f: F,
    ) where
        U: Copy,
        S: Number,
        F: FnMut(&T, &S, Option<&T>, Option<&S>) -> U,
    {
        let len = self.len();
        assert!(
            window <= len,
            "window should not be greater than the length of the array"
        );
        assert!(
            out.len() == len,
            "length of output array must equal to length of the array"
        );
        // within the first window
        for i in 0..window - 1 {
            let (v1, v2, vo) = unsafe { (self.uget(i), other.uget(i), out.uget_mut(i)) };
            // no value should be removed in the first window
            *vo = f(v1, v2, None, None);
        }
        // other windows
        for (start, end) in (window - 1..len).enumerate() {
            // new valid value
            let (v1_rm, v1, vo) = unsafe { (self.uget(start), self.uget(end), out.uget_mut(end)) };
            let (v2_rm, v2) = unsafe { (other.uget(start), other.uget(end)) };
            *vo = f(v1, v2, Some(v1_rm), Some(v2_rm));
        }
    }

    /// calculate mean value first and minus mean value for each element.
    #[inline]
    pub fn stable_apply_valid_window_to<F>(
        &self,
        out: &mut ArrViewMut1<f64>,
        window: usize,
        mut f: F,
    ) where
        F: FnMut(f64, f64) -> f64,
    {
        let len = self.len();
        assert!(
            window <= len,
            "window should not be greater than the length of the array"
        );
        assert!(
            out.len() == len,
            "length of output array must equal to length of the array"
        );
        let _mean = self.mean(false);
        // within the first window
        for i in 0..window - 1 {
            let (v, vo) = unsafe { (self.uget(i), out.uget_mut(i)) };
            // no value should be removed in the first window
            *vo = f(v.f64() - _mean, f64::NAN);
        }
        // other windows
        for (start, end) in (window - 1..len).enumerate() {
            // new valid value
            let (v_rm, v, vo) = unsafe { (self.uget(start), self.uget(end), out.uget_mut(end)) };
            let (v_rm, v) = (v_rm.f64() - _mean, v.f64() - _mean);
            *vo = f(v, v_rm);
        }
    }

    #[inline]
    pub fn stable_apply_valid_window_with_to<S, F>(
        &self,
        other: &ArrView1<S>,
        out: &mut ArrViewMut1<f64>,
        window: usize,
        mut f: F,
    ) where
        S: Number,
        F: FnMut(f64, f64, f64, f64) -> f64,
    {
        let len = self.len();
        assert!(
            window <= len,
            "window should not be greater than the length of the array"
        );
        assert!(
            out.len() == len,
            "length of output array must equal to length of the array"
        );
        let (_mean1, _mean2) = (self.mean(false), other.mean(false));
        // within the first window
        for i in 0..window - 1 {
            let (v1, v2, vo) = unsafe { (self.uget(i), other.uget(i), out.uget_mut(i)) };
            // no value should be removed in the first window
            *vo = f(v1.f64() - _mean1, v2.f64() - _mean2, f64::NAN, f64::NAN);
        }
        // other windows
        for (start, end) in (window - 1..len).enumerate() {
            // new valid value
            let (v1_rm, v1, vo) = unsafe { (self.uget(start), self.uget(end), out.uget_mut(end)) };
            let (v2_rm, v2) = unsafe { (other.uget(start), other.uget(end)) };
            let (v1_rm, v1) = (v1_rm.f64() - _mean1, v1.f64() - _mean1);
            let (v2_rm, v2) = (v2_rm.f64() - _mean2, v2.f64() - _mean2);
            *vo = f(v1, v2, v1_rm, v2_rm);
        }
    }
});

impl<'a, T> DerefMut for ArrViewMut1<'a, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> ArrViewMut1<'_, T> {
    /// Apply a function to each mut valid element from start to end
    #[inline(always)]
    pub fn apply_mut_on<F>(&mut self, mut f: F, start: usize, end: usize)
    where
        F: FnMut(&mut T),
    {
        assert!(end <= self.len()); // note start >= 0;
        for i in start..end {
            // safety: 0 <= i <= len-1
            let v = unsafe { self.uget_mut(i) };
            f(v);
        }
    }

    /// Apply a function to each mut element
    #[inline]
    pub fn apply_mut<F>(&mut self, f: F)
    where
        F: FnMut(&mut T),
    {
        self.apply_mut_on(f, 0, self.len());
    }

    /// Apply a function to each mut element and another arrays element
    pub fn apply_mut_with<S, F>(&mut self, other: &ArrView1<S>, mut f: F)
    where
        F: FnMut(&mut T, &S),
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
}

impl<T: Copy + Clone> ArrViewMut1<'_, T> {
    #[inline]
    pub fn sort_unstable_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {   
        if self.0.is_standard_layout() {
            let slice = self.as_slice_mut().unwrap();
            slice.sort_unstable_by(compare);
        } else {
            let mut out_c = self.0.to_owned();
            let slice = out_c.as_slice_mut().unwrap();
            slice.sort_unstable_by(compare);
            self.apply_mut_with(&ArrView1(out_c.view()), |v, vo| *v = *vo);
        }
    }
}
