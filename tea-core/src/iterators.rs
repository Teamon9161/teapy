use super::prelude::{ArrBase, Dim1};
use datatype::Number;
use ndarray::{Axis, Data, DataMut};
use std::iter::{ExactSizeIterator, FusedIterator, IntoIterator, Iterator};
use std::marker::PhantomData;
use std::ops::Add;

impl<T, S, D> IntoIterator for ArrBase<S, D>
where
    S: Data<Elem = T>,
    // T: Copy,
    D: Dim1,
{
    type Item = T;
    type IntoIter = IntoIter<T, D>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(&self)
    }
}

impl<'a, S, D> IntoIterator for &'a ArrBase<S, D>
where
    S: Data,
    D: Dim1,
{
    type Item = &'a S::Elem;
    type IntoIter = Iter<'a, S::Elem, D>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, S, D> IntoIterator for &'a mut ArrBase<S, D>
where
    S: DataMut,
    D: Dim1,
{
    type Item = &'a mut S::Elem;
    type IntoIter = IterMut<'a, S::Elem, D>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

pub struct IntoIter<T, D> {
    ptr: *const T,
    end: *const T,
    len: usize,
    stride: usize,
    _dim: PhantomData<D>,
}

impl<T, D> IntoIter<T, D> {
    #[inline]
    pub fn new<S>(arr: &ArrBase<S, D>) -> Self
    where
        S: Data<Elem = T>,
        D: Dim1,
    {
        let ptr = arr.as_ptr();
        let len = arr.len();
        let stride = arr.stride_of(Axis(0)) as usize;
        let end = unsafe { ptr.add(len * stride) };
        IntoIter {
            ptr,
            end,
            len,
            stride,
            _dim: PhantomData,
        }
    }
}

impl<T, D> Iterator for IntoIter<T, D> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.ptr == self.end {
            None
        } else {
            let v = Some(unsafe { std::ptr::read(self.ptr) });
            // let v = Some(unsafe { *self.ptr });
            self.ptr = unsafe { self.ptr.add(self.stride) };
            v
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<T, D> ExactSizeIterator for IntoIter<T, D> {
    #[inline]
    fn len(&self) -> usize {
        (self.end as usize - self.ptr as usize) / (std::mem::size_of::<T>() * self.stride)
    }
}

impl<T, D> DoubleEndedIterator for IntoIter<T, D> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.ptr == self.end {
            None
        } else {
            self.end = unsafe { self.end.sub(self.stride) };
            Some(unsafe { std::ptr::read(self.end) })
        }
    }
}

impl<T, D> FusedIterator for IntoIter<T, D> {}

pub struct Iter<'a, T, D> {
    ptr: *const T,
    end: *const T,
    len: usize,
    stride: usize,
    _life: PhantomData<&'a T>,
    _dim: PhantomData<D>,
}

impl<'a, T, D> Iter<'a, T, D> {
    #[inline]
    pub fn new<S>(arr: &ArrBase<S, D>) -> Self
    where
        S: Data<Elem = T>,
        D: Dim1,
    {
        let ptr = arr.as_ptr();
        let len = arr.len();
        let stride = arr.stride_of(Axis(0)) as usize;
        let end = unsafe { ptr.add(len * stride) };
        Iter {
            ptr,
            end,
            len,
            stride,
            _life: PhantomData,
            _dim: PhantomData,
        }
    }

    /// Accumulate value using function f, also return the number of
    /// valid elements.
    fn n_fold_valid<U, F>(mut self, init: U, mut f: F) -> (usize, U)
    where
        T: Number,
        F: FnMut(U, &T) -> U,
    {
        let mut accum = init;
        let len = self.len();
        let mut i = 0;
        let mut n = 0; // valid num
        unsafe {
            while i < len {
                let v = &*self.ptr.add(i * self.stride);
                if v.notnan() {
                    accum = f(accum, v);
                    n += 1;
                }
                i += 1;
            }
        }
        self.ptr = self.end;
        (n, accum)
    }

    /// Count the number of valid values and accumulate valid values using function f
    #[inline]
    pub fn n_acc_valid<U, F>(self, init: U, mut f: F) -> (usize, U)
    where
        T: Number,
        U: Add<Output = U>,
        F: FnMut(&T) -> U,
    {
        self.n_fold_valid(init, move |acc, v| acc + f(v))
    }
}

impl<'a, T, D> Iterator for Iter<'a, T, D> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if self.ptr == self.end {
            None
        } else {
            let v = Some(unsafe { &*self.ptr });
            self.ptr = unsafe { self.ptr.add(self.stride) };
            v
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len
    }

    fn fold<U, F>(mut self, init: U, mut f: F) -> U
    where
        Self: Sized,
        F: FnMut(U, Self::Item) -> U,
    {
        let mut accum = init;
        let len = self.len();
        let mut i = 0;
        unsafe {
            while i < len {
                accum = f(accum, &*self.ptr.add(i * self.stride));
                i += 1;
            }
        }
        self.ptr = self.end;
        accum
    }
}

impl<'a, T, D> ExactSizeIterator for Iter<'a, T, D> {
    #[inline]
    fn len(&self) -> usize {
        (self.end as usize - self.ptr as usize) / (std::mem::size_of::<T>() * self.stride)
    }
}

impl<'a, T, D> DoubleEndedIterator for Iter<'a, T, D> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        if self.ptr == self.end {
            None
        } else {
            self.end = unsafe { self.end.sub(self.stride) };
            Some(unsafe { &*self.end })
        }
    }
}

impl<'a, T, D> FusedIterator for Iter<'a, T, D> {}

pub struct IterMut<'a, T, D> {
    ptr: *mut T,
    end: *mut T,
    len: usize,
    stride: usize,
    _life: PhantomData<&'a T>,
    _dim: PhantomData<D>,
}

impl<'a, T, D> IterMut<'a, T, D> {
    #[inline]
    pub fn new<S>(arr: &mut ArrBase<S, D>) -> IterMut<'a, T, D>
    where
        S: DataMut<Elem = T>,
        D: Dim1,
    {
        let ptr = arr.as_mut_ptr();
        let len = arr.len();
        let stride = arr.stride_of(Axis(0)) as usize;
        let end = unsafe { ptr.add(len * stride) };
        IterMut {
            ptr,
            end,
            len,
            stride,
            _life: PhantomData,
            _dim: PhantomData,
        }
    }
}

impl<'a, T, D> Iterator for IterMut<'a, T, D> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        if self.ptr == self.end {
            None
        } else {
            let v = Some(unsafe { &mut *self.ptr });
            self.ptr = unsafe { self.ptr.add(self.stride) };
            v
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len
    }

    fn fold<U, F>(mut self, init: U, mut f: F) -> U
    where
        Self: Sized,
        F: FnMut(U, Self::Item) -> U,
    {
        let mut accum = init;
        let len = self.len();
        let mut i = 0;
        unsafe {
            while i < len {
                accum = f(accum, &mut *self.ptr.add(i * self.stride));
                i += 1;
            }
        }
        self.ptr = self.end;
        accum
    }
}

impl<'a, T, D> ExactSizeIterator for IterMut<'a, T, D> {
    #[inline]
    fn len(&self) -> usize {
        (self.end as usize - self.ptr as usize) / (std::mem::size_of::<T>() * self.stride)
    }
}

impl<'a, T, D> DoubleEndedIterator for IterMut<'a, T, D> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut T> {
        if self.ptr == self.end {
            None
        } else {
            self.end = unsafe { self.end.sub(self.stride) };
            Some(unsafe { &mut *self.end })
        }
    }
}

impl<'a, T, D> FusedIterator for IterMut<'a, T, D> {}

pub struct GroupIter<'a, T1: 'a, T2: 'a, P> {
    key: &'a [T1], // must be sorted
    value: &'a [T2],
    predicate: P,
}

impl<'a, T1: 'a, T2: 'a, P> GroupIter<'a, T1, T2, P> {
    #[allow(dead_code)]
    #[inline]
    pub fn new(key: &'a [T1], value: &'a [T2], predicate: P) -> Self {
        assert_eq!(key.len(), value.len());
        GroupIter {
            key,
            value,
            predicate,
        }
    }
}

impl<'a, T1: 'a, T2: 'a, P> Iterator for GroupIter<'a, T1, T2, P>
where
    P: FnMut(&T1, &T1) -> bool,
{
    type Item = (&'a [T1], &'a [T2]);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.key.is_empty() {
            None
        } else {
            let mut len = 1;
            let mut iter = self.key.windows(2);
            while let Some([l, r]) = iter.next() {
                if (self.predicate)(l, r) {
                    len += 1
                } else {
                    break;
                }
            }
            let (key_head, key_tail) = self.key.split_at(len);
            let (value_head, value_tail) = self.value.split_at(len);
            self.key = key_tail;
            self.value = value_tail;
            Some((key_head, value_head))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.key.is_empty() {
            (0, Some(0))
        } else {
            (1, Some(self.key.len()))
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, T1: 'a, T2: 'a, P> DoubleEndedIterator for GroupIter<'a, T1, T2, P>
where
    P: FnMut(&T1, &T1) -> bool,
{
    // #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.key.is_empty() {
            None
        } else {
            let mut len = 1;
            let mut iter = self.key.windows(2);
            while let Some([l, r]) = iter.next_back() {
                if (self.predicate)(l, r) {
                    len += 1
                } else {
                    break;
                }
            }
            let split_idx = self.key.len() - len;
            let (key_head, key_tail) = self.key.split_at(split_idx);
            let (value_head, value_tail) = self.value.split_at(split_idx);
            self.key = key_head;
            self.value = value_head;
            Some((key_tail, value_tail))
        }
    }
}

impl<'a, T1: 'a, T2: 'a, P> FusedIterator for GroupIter<'a, T1, T2, P> where
    P: FnMut(&T1, &T1) -> bool
{
}
