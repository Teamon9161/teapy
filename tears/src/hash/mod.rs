use ahash::RandomState;
use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
};

use crate::{Cast, DateTime};

pub type TpBuildHasher = BuildHasherDefault<TpHasher>;
pub type TpHashMap<K, V> = HashMap<K, V, TpBuildHasher>;
static HASHER: Lazy<RandomState> =
    Lazy::new(|| RandomState::with_seeds(2313, 12515, 12545345, 1245));

#[derive(Default)]
pub struct TpHasher {
    state: u64,
}

impl Hasher for TpHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        self.write_u64(HASHER.hash_one(bytes));
        // unimplemented!("hash arbitrary bytes is not supported")
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.state = i
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64)
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        // Safety: same number of bits
        unsafe { self.write_u32(std::mem::transmute::<i32, u32>(i)) }
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        // Safety: same number of bits
        unsafe { self.write_u64(std::mem::transmute::<i64, u64>(i)) }
    }
}

pub trait TpHash {
    fn hash(&self) -> u64;
}

macro_rules! impl_tphash {
    (uint $($ty: ty),*) => {
        $(
            impl TpHash for $ty {
                #[inline]
                fn hash(&self) -> u64 {
                    *self as u64
                }
            }
        )*
    };

    (int $($ty: ty),*) => {
        $(
            impl TpHash for $ty {
                #[inline]
                fn hash(&self) -> u64 {
                    unsafe {std::mem::transmute::<i64, u64>(self.clone().cast())}
                }
            }
        )*
    };

    (default $($ty: ty),*) => {
        $(
            impl TpHash for $ty {
                #[inline]
                fn hash(&self) -> u64 {
                    HASHER.hash_one(self)
                }
            }
        )*
    };
}

impl_tphash!(uint u8, u16, u32, u64, usize);
impl_tphash!(int i8, i16, i32, i64, isize, DateTime);
impl_tphash!(default String, &str, Vec<u64>, [u64]);

impl TpHash for f64 {
    #[inline]
    #[allow(clippy::transmute_float_to_int)]
    fn hash(&self) -> u64 {
        unsafe { std::mem::transmute::<f64, u64>((*self).cast()) }
    }
}

impl TpHash for f32 {
    #[inline]
    #[allow(clippy::transmute_float_to_int)]
    fn hash(&self) -> u64 {
        unsafe { std::mem::transmute::<f32, u32>((*self).cast()) as u64 }
    }
}

impl TpHash for bool {
    #[inline]
    #[allow(clippy::transmute_float_to_int)]
    fn hash(&self) -> u64 {
        *self as u64
    }
}
