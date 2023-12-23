use std::mem::MaybeUninit;

pub trait VecAssumeInit {
    type Elem;
    /// # Safety
    ///
    /// the elements in the vector must be initialized
    unsafe fn assume_init(self) -> Vec<Self::Elem>;

    /// An replacement of unstable API
    /// https://doc.rust-lang.org/std/mem/union.MaybeUninit.html#method.slice_assume_init_ref
    /// # Safety
    ///
    /// the elements in the vector must be initialized
    unsafe fn slice_assume_init_ref(&self) -> &[Self::Elem];

    /// An replacement of unstable API
    /// https://doc.rust-lang.org/std/mem/union.MaybeUninit.html#method.slice_assume_init_mut
    /// # Safety
    ///
    /// the elements in the vector must be initialized    
    unsafe fn slice_assume_init_mut(&mut self) -> &mut [Self::Elem];
}

impl<T> VecAssumeInit for Vec<MaybeUninit<T>> {
    type Elem = T;
    #[inline]
    unsafe fn assume_init(self) -> Vec<T> {
        // FIXME use Vec::into_raw_parts instead after stablized
        // https://doc.rust-lang.org/std/vec/struct.Vec.html#method.into_raw_parts
        let mut me = std::mem::ManuallyDrop::new(self);
        Vec::from_raw_parts(me.as_mut_ptr() as *mut T, me.len(), me.capacity())
    }

    #[inline]
    unsafe fn slice_assume_init_ref(&self) -> &[T] {
        std::slice::from_raw_parts(self.as_ptr() as *const T, self.len())
    }

    #[inline]
    unsafe fn slice_assume_init_mut(&mut self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.as_mut_ptr() as *mut T, self.len())
    }
}

/// Create a vector without initialization
///
/// Safety
/// ------
/// - Memory is not initialized. Do not read the memory before write.
///
#[inline]
pub fn vec_uninit<T: Sized>(n: usize) -> Vec<MaybeUninit<T>> {
    let mut v = Vec::with_capacity(n);
    unsafe {
        v.set_len(n);
    }
    v
}
