use std::iter::IntoIterator;

/// The remain length of the iterator can be trusted
///
/// # Safety
///
/// the size hint of the iterator should be correct
pub unsafe trait TrustedLen {}

pub trait CollectTrusted<T> {
    fn collect_from_trusted<I>(i: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: TrustedLen;
}

impl<T> CollectTrusted<T> for Vec<T> {
    /// safety: upper bound on the remaining length of the iterator must be correct.
    fn collect_from_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        let len = iter
            .size_hint()
            .1
            .expect("The iterator must have an upper bound");
        let mut vec = Vec::<T>::with_capacity(len);
        let mut ptr = vec.as_mut_ptr();
        unsafe {
            for v in iter {
                std::ptr::write(ptr, v);
                ptr = ptr.add(1);
            }
            vec.set_len(len);
        }
        vec
    }
}

pub trait CollectTrustedToVec: Iterator + TrustedLen {
    fn collect_trusted(self) -> Vec<Self::Item>
    where
        Self: Sized,
    {
        CollectTrusted::<Self::Item>::collect_from_trusted(self)
    }
}

impl<T: Iterator + TrustedLen> CollectTrustedToVec for T {}
unsafe impl<K, V> TrustedLen for std::collections::hash_map::IntoIter<K, V> {}
unsafe impl<K, V> TrustedLen for std::collections::hash_map::IntoValues<K, V> {}
unsafe impl<T, D> TrustedLen for crate::arr::iterators::IntoIter<T, D> {}
unsafe impl<T1, T2> TrustedLen for std::iter::Map<T1, T2> {}
