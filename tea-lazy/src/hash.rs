// #[cfg(feature = "ahasher")]
// pub use ahash::AHashMap as TpHashMap;

// #[cfg(feature = "gxhasher")]
// pub use gxhash::GxHashMap as TpHashMap;

// #[cfg(feature = "gxhasher")]
// pub trait GxHasherExt {
//     fn with_capacity(capacity: usize) -> Self;
// }
// #[cfg(feature = "gxhasher")]
// impl<K, V> GxHasherExt for TpHashMap<K, V> {
//     fn with_capacity(capacity: usize) -> Self {
//         let hasher = gxhash::GxBuildHasher::default();
//         TpHashMap::with_capacity_and_hasher(capacity, hasher)
//     }
// }
