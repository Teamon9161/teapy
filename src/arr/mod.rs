mod agg;
pub mod algos;
mod arr_func;
mod corr;
pub mod datatype;
mod groupby;
mod iterators;
mod macros;
mod prelude;
mod utils;
mod window;

use algos::kh_sum;
use datatype::Number;
use iterators::{Iter, IterMut};
use ndarray::{
    Array1, ArrayBase, Axis, Data, DataMut, DataOwned, Dimension, Ix1, NdIndex, OwnedRepr, RawData,
    RawDataClone, RemoveAxis, ShapeBuilder, ShapeError, ViewRepr, Zip,
};
use num::Zero;
use std::cmp::Ordering;
use std::ops::{Add, Deref, DerefMut};

pub struct ArrBase<S, D>(pub ArrayBase<S, D>)
where
    S: RawData;

impl<S: RawData, D> Deref for ArrBase<S, D> {
    type Target = ArrayBase<S, D>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: RawData, D> DerefMut for ArrBase<S, D> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub trait Dim1: Dimension + RemoveAxis {}
impl Dim1 for Ix1 {}

pub type Arr<T, D> = ArrBase<OwnedRepr<T>, D>;
pub type ArrView<'a, T, D> = ArrBase<ViewRepr<&'a T>, D>;
pub type ArrViewMut<'a, T, D> = ArrBase<ViewRepr<&'a mut T>, D>;
pub type ArrBase1<S> = ArrBase<S, Ix1>;
pub type Arr1<T> = Arr<T, Ix1>;
pub type ArrViewMut1<'a, T> = ArrViewMut<'a, T, Ix1>;
pub type ArrView1<'a, T> = ArrView<'a, T, Ix1>;

impl<T, S, D> ArrBase<S, D>
where
    S: RawData<Elem = T>,
    D: Dimension,
{
    pub fn new(a: ArrayBase<S, D>) -> Self {
        Self(a)
    }

    pub fn from_vec(v: Vec<T>) -> Arr1<T>
    where
        S: DataOwned,
        D: Dim1,
    {
        Array1::from_vec(v).wrap()
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I: IntoIterator<Item = T>>(iterable: I) -> Arr1<T>
    where
        D: Dim1,
        S: DataOwned<Elem = T>,
    {
        Array1::from_iter(iterable).wrap()
    }

    /// Create a 1d array from slice, need clone.
    pub fn from_slice(slc: &[T]) -> Arr1<T>
    where
        T: Clone,
        D: Dim1,
    {
        Array1::from_vec(slc.to_vec()).wrap()
    }

    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        S: DataOwned<Elem = T>,
        T: Clone + Zero,
        Sh: ShapeBuilder<Dim = D>,
    {
        ArrayBase::zeros(shape).wrap()
    }

    /// Change the array to dim1.
    ///
    /// Note that the original array must be dim1.
    #[inline]
    pub fn to_dim1(self) -> ArrBase<S, Ix1> {
        self.to_dim::<Ix1>().unwrap()
    }

    /// Change the array to another dim.
    #[inline]
    pub fn to_dim<D2: Dimension>(self) -> Result<ArrBase<S, D2>, ShapeError> {
        let res = self.0.into_dimensionality::<D2>();
        res.map(|arr| ArrBase(arr))
    }

    /// Clone the elements in the array to `out` array.
    #[inline]
    pub fn clone_to<S2>(&self, out: &mut ArrBase<S2, D>)
    where
        T: Clone + Copy,
        S: Data,
        S2: DataMut<Elem = T>,
        D: Dim1,
        usize: NdIndex<D>,
    {
        out.apply_mut_with(self, |vo, v| *vo = *v);
    }

    /// Return a read-only view of the array
    pub fn view(&self) -> ArrView<'_, T, D>
    where
        S: Data,
    {
        ArrBase(self.0.view())
    }

    /// Return a read-write view of the array
    pub fn view_mut(&mut self) -> ArrViewMut<'_, T, D>
    where
        S: DataMut,
    {
        ArrBase(self.0.view_mut())
    }

    /// Return an uniquely owned copy of the array.
    pub fn to_owned(&self) -> Arr<T, D>
    where
        T: Clone,
        S: Data,
    {
        self.0.to_owned().wrap()
    }
    /// Call `f` by reference on each element and create a new array
    /// with the new values.
    ///
    /// Elements are visited in arbitrary order.
    ///
    /// Return an array with the same shape as `self`.
    pub fn map<'a, T2, F>(&'a self, f: F) -> Arr<T2, D>
    where
        F: FnMut(&'a T) -> T2,
        T: 'a,
        S: Data,
    {
        self.0.map(f).wrap()
    }

    /// Call `f` by **v**alue on each element and create a new array
    /// with the new values.
    ///
    /// Elements are visited in arbitrary order.
    ///
    /// Return an array with the same shape as `self`.
    pub fn mapv<T2, F>(&self, mut f: F) -> Arr<T2, D>
    where
        F: FnMut(T) -> T2,
        T: Clone,
        S: Data,
    {
        self.map(move |x| f(x.clone()))
    }

    pub fn apply_along_axis<S2, T2, F>(
        &self,
        out: &mut ArrayBase<S2, D>,
        axis: Axis,
        par: bool,
        f: F,
    ) where
        T: Send + Sync,
        T2: Send + Sync,
        S: Data,
        S2: DataMut<Elem = T2>,
        F: Fn(ArrayBase<ViewRepr<&T>, Ix1>, ArrayBase<ViewRepr<&mut T2>, Ix1>) + Send + Sync,
    {
        let arr_zip = Zip::from(self.lanes(axis)).and(out.lanes_mut(axis));
        let ndim = self.ndim();
        if !par || (ndim == 1) {
            // 非并行
            arr_zip.for_each(f);
        } else {
            // 并行
            arr_zip.par_for_each(f);
        }
    }

    pub fn apply_along_axis_with<S2, T2, S3, T3, F>(
        &self,
        other: &ArrayBase<S2, D>,
        out: &mut ArrayBase<S3, D>,
        axis: Axis,
        par: bool,
        f: F,
    ) where
        T: Send + Sync,
        T2: Send + Sync,
        T3: Send + Sync,
        S: Data,
        S2: Data<Elem = T2>,
        S3: DataMut<Elem = T3>,
        F: Fn(
                ArrayBase<ViewRepr<&T>, Ix1>,
                ArrayBase<ViewRepr<&T2>, Ix1>,
                ArrayBase<ViewRepr<&mut T3>, Ix1>,
            ) + Send
            + Sync,
    {
        let arr_zip = Zip::from(self.lanes(axis))
            .and(other.lanes(axis))
            .and(out.lanes_mut(axis));
        let ndim = self.ndim();
        if !par || (ndim == 1) {
            // 非并行
            arr_zip.for_each(f);
        } else {
            // 并行
            arr_zip.par_for_each(f);
        }
    }

    #[inline]
    pub fn apply_mut_on<F>(&mut self, mut f: F, start: usize, end: usize)
    where
        D: Dim1,
        usize: NdIndex<D>,
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

    #[inline]
    pub fn apply_mut<F>(&mut self, f: F)
    where
        D: Dim1,
        usize: NdIndex<D>,
        S: DataMut,
        F: FnMut(&mut T),
    {
        self.apply_mut_on(f, 0, self.len());
    }

    /// Accumulate value using function f
    #[inline]
    pub fn fold<U, F>(&self, init: U, mut f: F) -> U
    where
        D: Dim1,
        usize: NdIndex<D>,
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

    /// Accumulate valid value using function f
    #[inline]
    pub fn fold_valid<U, F>(&self, init: U, mut f: F) -> U
    where
        T: Number,
        S: Data,
        usize: NdIndex<D>,
        D: Dim1,
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
    pub fn fold_valid_with<U, S2, T2, F>(&self, other: &ArrBase<S2, D>, init: U, mut f: F) -> U
    where
        S: Data,
        S2: Data<Elem = T2>,
        T: Number,
        T2: Number,
        usize: NdIndex<D>,
        D: Dim1,
        F: FnMut(U, &T, &T2) -> U,
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
        S: Data,
        usize: NdIndex<D>,
        D: Dim1,
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
        other: &ArrBase<S2, D>,
        init: U,
        mut f: F,
    ) -> (usize, U)
    where
        S: Data,
        S2: Data<Elem = T2>,
        T: Number,
        T2: Number,
        usize: NdIndex<D>,
        D: Dim1,
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
    #[inline]
    pub fn acc<U, F>(&self, init: U, mut f: F) -> U
    where
        U: Add<Output = U>,
        S: Data,
        usize: NdIndex<D>,
        D: Dim1,
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
        usize: NdIndex<D>,
        D: Dim1,
        F: FnMut(&T) -> U,
    {
        self.n_fold_valid(init, |acc, v| acc + f(v))
    }

    /// Only accumulate valild value using function f
    #[inline]
    pub fn acc_valid<U, F>(&self, init: U, mut f: F) -> U
    where
        T: Number,
        U: Add<Output = U>,
        S: Data,
        usize: NdIndex<D>,
        D: Dim1,
        F: FnMut(&T) -> U,
    {
        self.fold_valid(init, |acc, v| acc + f(v))
    }

    /// Count the number of valid values and accumulate value using Kahan summation
    #[inline]
    pub fn stable_n_acc_valid<U, F>(&self, init: U, mut f: F) -> (usize, U)
    where
        S: Data,
        U: Number,
        T: Number,
        usize: NdIndex<D>,
        D: Dim1,
        F: FnMut(&T) -> U,
    {
        let c = &mut U::zero();
        self.n_fold_valid(init, |acc, v| kh_sum(acc, f(v), c))
    }

    /// Only accumulate valild value using kahan summation
    #[inline]
    pub fn stable_acc_valid<U, F>(&self, init: U, mut f: F) -> U
    where
        S: Data,
        U: Number,
        T: Number,
        usize: NdIndex<D>,
        D: Dim1,
        F: FnMut(&T) -> U,
    {
        let c = &mut U::zero();
        self.fold_valid(init, |acc, v| kh_sum(acc, f(v), c))
    }

    /// Apply a function to each element
    #[inline]
    pub fn apply<F>(&self, mut f: F)
    where
        S: Data,
        usize: NdIndex<D>,
        D: Dim1,
        F: FnMut(&T),
    {
        self.fold((), move |(), elt| f(elt))
    }

    /// Apply a function to each mut element
    #[inline]
    pub fn apply_mut_with<S2, T2, F>(&mut self, other: &ArrBase<S2, D>, mut f: F)
    where
        S: DataMut,
        S2: Data<Elem = T2>,
        usize: NdIndex<D>,
        D: Dim1,
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
    #[inline]
    pub fn apply_valid<F>(&self, mut f: F)
    where
        T: Number,
        S: Data,
        usize: NdIndex<D>,
        D: Dim1,
        F: FnMut(&T),
    {
        self.fold_valid((), move |(), elt| f(elt))
    }

    /// Apply a function to each valid element and return the number of valid values
    #[inline]
    pub fn n_apply_valid<F>(&self, mut f: F) -> usize
    where
        T: Number,
        S: Data,
        usize: NdIndex<D>,
        D: Dim1,
        F: FnMut(&T),
    {
        self.n_fold_valid((), move |(), elt| f(elt)).0
    }

    /// Apply a function to self and other only when both elements are valid
    #[inline]
    pub fn apply_valid_with<S2, T2, F>(&self, other: &ArrBase<S2, D>, mut f: F)
    where
        S: Data,
        S2: Data<Elem = T2>,
        T: Number,
        T2: Number,
        usize: NdIndex<D>,
        D: Dim1,
        F: FnMut(&T, &T2),
    {
        self.fold_valid_with(other, (), move |(), elt1, elt2| f(elt1, elt2))
    }

    /// Apply a function to self and other only when both elements are valid
    #[inline]
    pub fn n_apply_valid_with<S2, T2, F>(&self, other: &ArrBase<S2, D>, mut f: F) -> usize
    where
        S: Data,
        S2: Data<Elem = T2>,
        T: Number,
        T2: Number,
        usize: NdIndex<D>,
        D: Dim1,
        F: FnMut(&T, &T2),
    {
        self.n_fold_valid_with(other, (), move |(), elt1, elt2| f(elt1, elt2))
            .0
    }

    /// Apply function `f` on each element,
    /// if the function return `true`,
    /// `count += 1`, then return count
    #[inline]
    pub fn count_by<F>(&self, mut f: F) -> usize
    where
        D: Dim1,
        usize: NdIndex<D>,
        S: Data,
        F: FnMut(&T) -> bool,
    {
        self.fold(0, move |acc, elt| acc + (f(elt) as usize))
    }

    pub fn iter(&self) -> Iter<'_, T, D>
    where
        D: Dim1,
        S: Data,
    {
        Iter::new(self)
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, T, D>
    where
        D: Dim1,
        S: DataMut,
    {
        IterMut::new(self)
    }

    /// Apply a window function on self and set values returned by `f`,
    ///
    /// the first argument of `f` is the new value in the window.
    ///
    /// the second argument of `f` is the value to be removed in the window.
    #[inline]
    pub fn apply_window_to<U, S2, F>(&self, out: &mut ArrBase<S2, D>, window: usize, mut f: F)
    where
        U: Copy,
        D: Dim1,
        S: Data,
        S2: DataMut<Elem = U>,
        usize: NdIndex<D>,
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
    pub fn apply_window_with_to<U, S2, T2, S3, F>(
        &self,
        other: &ArrBase<S2, D>,
        out: &mut ArrBase<S3, D>,
        window: usize,
        mut f: F,
    ) where
        U: Copy,
        D: Dim1,
        S: Data,
        S2: Data<Elem = T2>,
        S3: DataMut<Elem = U>,
        usize: NdIndex<D>,
        F: FnMut(&T, &T2, Option<&T>, Option<&T2>) -> U,
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
    pub fn stable_apply_window_to<S2, F>(&self, out: &mut ArrBase<S2, D>, window: usize, mut f: F)
    where
        T: Number,
        D: Dim1,
        S: Data,
        S2: DataMut<Elem = f64>,
        usize: NdIndex<D>,
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
        let _mean = self.mean_1d(false);
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
    pub fn stable_apply_window_with_to<S2, T2, S3, F>(
        &self,
        other: &ArrBase<S2, D>,
        out: &mut ArrBase<S3, D>,
        window: usize,
        mut f: F,
    ) where
        T: Number,
        T2: Number,
        D: Dim1,
        S: Data,
        S2: Data<Elem = T2>,
        S3: DataMut<Elem = f64>,
        usize: NdIndex<D>,
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
        let (_mean1, _mean2) = (self.mean_1d(false), other.mean_1d(false));
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

    pub fn sort_unstable_by<F>(&mut self, compare: F)
    where
        T: Copy + Clone,
        S: DataMut,
        usize: NdIndex<D>,
        D: Dim1,
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

pub trait WrapNdarray<S: RawData, D: Dimension> {
    fn wrap(self) -> ArrBase<S, D>;
}

impl<S: RawData, D: Dimension> WrapNdarray<S, D> for ArrayBase<S, D> {
    fn wrap(self) -> ArrBase<S, D> {
        ArrBase::new(self)
    }
}

use std::convert::From;
impl<S: RawData, D: Dimension> From<ArrayBase<S, D>> for ArrBase<S, D> {
    fn from(arr: ArrayBase<S, D>) -> Self {
        ArrBase::new(arr)
    }
}

impl<S: RawDataClone, D: Clone + Dimension> Clone for ArrBase<S, D> {
    fn clone(&self) -> Self {
        self.0.clone().wrap()
    }
}
