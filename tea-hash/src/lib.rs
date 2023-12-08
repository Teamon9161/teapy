extern crate tea_dtype as datatype;

// use once_cell::sync::Lazy;
// use std::hash::{Hash, Hasher};

#[cfg(feature = "ahasher")]
use ahash::{AHasher, RandomState};
use datatype::Cast;
#[cfg(feature = "time")]
use datatype::DateTime;
#[cfg(feature = "ahasher")]
// pub static BUILD_HASHER: Lazy<RandomState> =
//     Lazy::new(|| RandomState::with_seeds(2313, 12515, 12545345, 1245));
pub static BUILD_HASHER: RandomState = RandomState::with_seeds(2313, 12515, 12545345, 1245);
#[cfg(feature = "ahasher")]
pub type TpHasher = AHasher;
#[cfg(feature = "ahasher")]
pub type TpHashMap<K, V> = ahash::AHashMap<K, V>;

#[cfg(feature = "gxhasher")]
use gxhash::GxHasher;
#[cfg(feature = "gxhasher")]
static HASHER: Lazy<GxHasher> = Lazy::new(|| GxHasher::with_seeds(2313));

// #[derive(Default)]
// pub struct TpHasher1 {
//     state: u64,
// }

// impl Hasher for TpHasher1 {
//     #[inline]
//     fn finish(&self) -> u64 {
//         self.state
//     }

//     #[inline]
//     fn write(&mut self, _bytes: &[u8]) {
//         // self.write_u64(BUILD_HASHER.hash_one(bytes));
//         unimplemented!("hash arbitrary bytes is not supported")
//     }

//     #[inline]
//     fn write_u64(&mut self, i: u64) {
//         self.state = i
//     }

//     #[inline]
//     fn write_u32(&mut self, i: u32) {
//         self.state = i as u64
//     }

//     #[inline]
//     fn write_i32(&mut self, i: i32) {
//         // Safety: same number of bits
//         unsafe { self.write_u32(std::mem::transmute::<i32, u32>(i)) }
//     }

//     #[inline]
//     fn write_i64(&mut self, i: i64) {
//         // Safety: same number of bits
//         unsafe { self.write_u64(std::mem::transmute::<i64, u64>(i)) }
//     }
// }

pub trait TpHash {
    fn tphash(&self) -> u64;
}

macro_rules! impl_tphash {
    (uint $($ty: ty),*) => {
        $(
            impl TpHash for $ty {
                #[inline]
                fn tphash(&self) -> u64 {
                    *self as u64
                }
            }
        )*
    };

    (int $($ty: ty),*) => {
        $(
            impl TpHash for $ty {
                #[inline]
                fn tphash(&self) -> u64 {
                    unsafe {std::mem::transmute::<i64, u64>(self.clone().cast())}
                }
            }
        )*
    };

    (default $($ty: ty),*) => {
        $(
            impl TpHash for $ty {
                #[inline]
                fn tphash(&self) -> u64 {
                    BUILD_HASHER.hash_one(self)
                }
            }
        )*
    };
}

impl_tphash!(uint u8, u16, u32, u64, usize);
impl_tphash!(int i8, i16, i32, i64, isize);
impl_tphash!(default String, &str, Vec<u64>, [u64]);
#[cfg(feature = "time")]
impl_tphash!(int DateTime);

impl TpHash for f64 {
    #[inline]
    #[allow(clippy::transmute_float_to_int)]
    fn tphash(&self) -> u64 {
        unsafe { std::mem::transmute::<f64, u64>(*self) }
    }
}

impl TpHash for f32 {
    #[inline]
    #[allow(clippy::transmute_float_to_int)]
    fn tphash(&self) -> u64 {
        unsafe { std::mem::transmute::<f32, u32>(*self) as u64 }
    }
}

impl TpHash for bool {
    #[inline]
    #[allow(clippy::transmute_float_to_int)]
    fn tphash(&self) -> u64 {
        *self as u64
    }
}
