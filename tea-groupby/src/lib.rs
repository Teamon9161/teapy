#![feature(hash_raw_entry)]

#[macro_use]
extern crate tea_macros;

mod from_py;
#[cfg(feature = "lazy")]
mod groupby_agg;
#[cfg(feature = "lazy")]
mod impl_lazy;
#[cfg(feature = "lazy")]
mod join;
#[cfg(feature = "lazy")]
mod unique;

#[cfg(feature = "lazy")]
pub use groupby_agg::{AutoExprGroupbyAggExt, GroupbyAggExt};
#[cfg(feature = "lazy")]
pub use impl_lazy::ExprGroupByExt;
#[cfg(feature = "lazy")]
pub use join::{join_left, join_outer, ExprJoinExt, JoinType};
#[cfg(feature = "lazy")]
pub use unique::ExprUniqueExt;

use ndarray::{Data, Ix1};
// use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use std::collections::hash_map::Entry;
use std::hash::{BuildHasher, Hash, Hasher};
use tea_core::prelude::*;
use tea_core::utils::CollectTrustedToVec;
use tea_hash::{TpHash, TpHashMap, BUILD_HASHER};

// /// Get the partition size for parallel
// pub fn get_partition_size() -> usize {
//     let mut n_partitions = rayon::current_num_threads();
//     // set n_partitions to closes 2^n above the no of threads.
//     loop {
//         if n_partitions.is_power_of_two() {
//             break;
//         } else {
//             n_partitions += 1;
//         }
//     }
//     n_partitions
// }

// #[inline]
// /// For partition that are a power of 2 we can use a bitshift instead of a modulo.
// pub(crate) fn partition_here(h: u64, thread: u64, n_partition: u64) -> bool {
//     debug_assert!(n_partition.is_power_of_two());
//     // n % 2^i = n & (2^i - 1)
//     (h.wrapping_add(thread)) & n_partition.wrapping_sub(1) == 0
// }

// Faster than collecting from a flattened iterator.
pub fn flatten<T: Clone, R: AsRef<[T]>>(bufs: &[R], len: Option<usize>) -> Vec<T> {
    let len = len.unwrap_or_else(|| bufs.iter().map(|b| b.as_ref().len()).sum());
    let mut out = Vec::with_capacity(len);
    for b in bufs {
        out.extend_from_slice(b.as_ref());
    }
    out
}

// static GROUP_DICT_INIT_SIZE: usize = 512;
pub static GROUP_VEC_INIT_SIZE: usize = 16;

#[ext_trait]
impl<T, S: Data<Elem = T>> HashExt1d for ArrBase<S, Ix1> {
    /// Hash each element of the array.
    #[inline]
    fn hash_1d(self) -> Arr1<u64>
    where
        T: Hash,
    {
        self.map(|v| BUILD_HASHER.hash_one(v))
    }

    /// Hash each element of the array.
    #[inline]
    fn tphash_1d(self) -> Arr1<u64>
    where
        T: TpHash,
    {
        self.map(|v| v.tphash())
    }
}

pub fn prepare_groupby(
    keys: &[&ArrOk<'_>],
    // hasher: Option<TpBuildHasher>,
    _par: bool,
) -> (usize, Vec<Arr1<u64>>) {
    // let hasher = hasher.unwrap_or(TpBuildHasher::default());
    let hashed_keys = keys
        .iter()
        .map(|arr| {
            match_arrok!(
                hash
                arr,
                a,
                {
                    let a = a.view()
                        .to_dim1()
                        .expect("groupby key should be dim1");
                    a.tphash_1d()
                }
            )
        })
        .collect::<Vec<_>>();
    if keys.is_empty() {
        return (0, hashed_keys);
    }
    let len = hashed_keys[0].len();
    for key in &hashed_keys {
        if key.len() != len {
            panic!("All of the groupby keys should have the same shape")
        }
    }
    (len, hashed_keys)
}

// pub fn collect_hashmap_one_key(
//     len: usize,
//     // hasher: Build,
//     hashed_key: &Arr1<u64>,
//     size_hint: Option<usize>,
// ) -> TpHashMap<u64, (usize, Vec<usize>)> {
//     let init_size = if let Some(size) = size_hint {
//         size
//     } else {
//         (len / 2).min(1)
//     };
//     // let hasher = BUILD_HASHER.build_hasher();
//     let mut group_dict =
//         TpHashMap::<u64, (usize, Vec<usize>)>::with_capacity_and_hasher(init_size, Lazy::force(&BUILD_HASHER).clone());
//     for i in 0..len {
//         let hash = unsafe { *hashed_key.uget(i) };
//         let entry = group_dict.entry(hash);
//         match entry {
//             Entry::Vacant(entry) => {
//                 let mut group_idx_vec = Vec::with_capacity(GROUP_VEC_INIT_SIZE);
//                 group_idx_vec.push(i);
//                 entry.insert((i, group_idx_vec));
//             }
//             Entry::Occupied(mut entry) => {
//                 let v = entry.get_mut();
//                 v.1.push(i);
//             }
//         }
//         // let entry = group_dict.raw_entry_mut().from_key_hashed_nocheck(hash, &hash);
//         // match entry {
//         //     RawEntryMut::Vacant(entry) => {
//         //         let mut group_idx_vec = Vec::with_capacity(GROUP_VEC_INIT_SIZE);
//         //         group_idx_vec.push(i);
//         //         entry.insert_hashed_nocheck(hash, hash, (i, group_idx_vec));
//         //         // entry.insert((i, group_idx_vec));
//         //     }
//         //     RawEntryMut::Occupied(mut entry) => {
//         //         let v = entry.get_mut();
//         //         v.1.push(i);
//         //     }
//         // }
//     }
//     group_dict
// }
#[allow(clippy::clone_on_copy)]
pub fn collect_hashmap_keys(
    len: usize,
    // hasher: RandomState,
    hashed_keys: &[Arr1<u64>],
    size_hint: Option<usize>,
) -> TpHashMap<u64, (usize, Vec<usize>)> {
    let init_size = if let Some(size) = size_hint {
        size
    } else {
        (len / 2).min(1)
    };
    let mut hasher = BUILD_HASHER.build_hasher();
    let mut group_dict = TpHashMap::<u64, (usize, Vec<usize>)>::with_capacity_and_hasher(
        init_size,
        BUILD_HASHER.clone(),
    );
    for i in 0..len {
        let tuple_keys = hashed_keys
            .iter()
            .map(|keys| unsafe { *keys.uget(i) })
            .collect_trusted();
        tuple_keys.hash(&mut hasher);
        let entry = group_dict.entry(hasher.finish());
        match entry {
            Entry::Vacant(entry) => {
                let mut group_idx_vec = Vec::with_capacity(GROUP_VEC_INIT_SIZE);
                group_idx_vec.push(i);
                entry.insert((i, group_idx_vec));
            }
            Entry::Occupied(mut entry) => {
                let v = entry.get_mut();
                v.1.push(i);
            }
        }
    }
    group_dict
}

#[allow(suspicious_double_ref_op, clippy::clone_on_copy)]
pub fn groupby(keys: &[&ArrOk<'_>], sort: bool) -> Vec<(usize, Vec<usize>)> {
    let len = keys[0].len();
    for key in keys {
        if key.len() != len {
            panic!("All of the groupby keys should have the same shape")
        }
    }
    let init_size = (len / 2).min(1);
    let by_len = keys.len();
    let mut vec = if by_len == 1 {
        let key = keys[0];
        match_arrok!(hash key, key_arr, {
            let mut group_dict =
                TpHashMap::<_, (usize, Vec<usize>)>::with_capacity_and_hasher(init_size, BUILD_HASHER.clone());
            let arr = key_arr.view().to_dim1().unwrap();
            for i in 0..len {
                let value = unsafe { arr.uget(i) }.clone();
                let entry = group_dict.entry(value);
                match entry {
                    Entry::Vacant(entry) => {
                        let mut group_idx_vec = Vec::with_capacity(GROUP_VEC_INIT_SIZE);
                        group_idx_vec.push(i);
                        entry.insert((i, group_idx_vec));
                    }
                    Entry::Occupied(mut entry) => {
                        let v = entry.get_mut();
                        v.1.push(i);
                    }
                }
            }
            group_dict.into_values().collect_trusted()
        })
    } else if by_len == 2 {
        let key0 = keys[0];
        let key1 = keys[1];
        match_arrok!(hash key0, key0_arr, {
            match_arrok!(hash key1, key1_arr, {
                let mut group_dict =
                    TpHashMap::<_, (usize, Vec<usize>)>::with_capacity_and_hasher(init_size, BUILD_HASHER.clone());
                let arr0 = key0_arr.view().to_dim1().unwrap();
                let arr1 = key1_arr.view().to_dim1().unwrap();
                for i in 0..len {
                    let value0 = unsafe { arr0.uget(i) }.clone();
                    let value1 = unsafe { arr1.uget(i) }.clone();
                    let tuple_keys = (value0, value1);
                    let entry = group_dict.entry(tuple_keys);
                    match entry {
                        Entry::Vacant(entry) => {
                            let mut group_idx_vec = Vec::with_capacity(GROUP_VEC_INIT_SIZE);
                            group_idx_vec.push(i);
                            entry.insert((i, group_idx_vec));
                        }
                        Entry::Occupied(mut entry) => {
                            let v = entry.get_mut();
                            v.1.push(i);
                        }
                    }
                }
                group_dict.into_values().collect_trusted()
            })
        })
    } else if by_len == 3 {
        let key0 = keys[0];
        let key1 = keys[1];
        let key2 = keys[2];
        match_arrok!(hash key0, key0_arr, {
            match_arrok!(hash key1, key1_arr, {
                match_arrok!(hash key2, key2_arr, {
                    let mut group_dict =
                        TpHashMap::<_, (usize, Vec<usize>)>::with_capacity_and_hasher(init_size, BUILD_HASHER.clone());
                    let arr0 = key0_arr.view().to_dim1().unwrap();
                    let arr1 = key1_arr.view().to_dim1().unwrap();
                    let arr2 = key2_arr.view().to_dim1().unwrap();
                    for i in 0..len {
                        let value0 = unsafe { arr0.uget(i) }.clone();
                        let value1 = unsafe { arr1.uget(i) }.clone();
                        let value2 = unsafe { arr2.uget(i) }.clone();
                        let tuple_keys = (value0, value1, value2);
                        let entry = group_dict.entry(tuple_keys);
                        match entry {
                            Entry::Vacant(entry) => {
                                let mut group_idx_vec = Vec::with_capacity(GROUP_VEC_INIT_SIZE);
                                group_idx_vec.push(i);
                                entry.insert((i, group_idx_vec));
                            }
                            Entry::Occupied(mut entry) => {
                                let v = entry.get_mut();
                                v.1.push(i);
                            }
                        }
                    }
                    group_dict.into_values().collect_trusted()
                })
            })
        })
    } else {
        let (len, hashed_keys) = prepare_groupby(keys, false);
        let group_dict = collect_hashmap_keys(len, &hashed_keys, None);
        group_dict.into_values().collect_trusted()
    };
    if sort {
        vec.sort_unstable_by_key(|v| v.0);
    }
    vec
}

// /// Groupby this array, return a `vec` contains
// /// index of the first value in each group and a `vec` contains the
// /// index of values within each group.
// ///
// /// sort: whether to sort on index of the first value in each group
// pub fn groupby_par(keys: &[&ArrOk<'_>], sort: bool) -> Vec<(usize, Vec<usize>)> {
//     todo!()
//     // let n_partition = get_partition_size() as u64;
//     // let (len, hashed_keys) = prepare_groupby(keys, true);
//     // let hasher = RandomState::new();
//     // let thread_init_size = (len / n_partition as usize).min(1);
//     // let mut out = (0..n_partition)
//     //     .into_par_iter()
//     //     .map(|thread| {
//     //         let hasher = hasher.clone();
//     //         // fast path for only one key
//     //         if hashed_keys.len() == 1 {
//     //             let hashed_key = &hashed_keys[0];
//     //             let mut group_dict = AHashMap::<u64, (usize, Vec<usize>)>::with_capacity_and_hasher(
//     //                 thread_init_size,
//     //                 hasher,
//     //             );
//     //             for i in 0..len {
//     //                 let hash = unsafe { *hashed_key.uget(i) };
//     //                 if partition_here(hash, thread, n_partition) {
//     //                     let entry = group_dict.entry(hash);
//     //                     match entry {
//     //                         Entry::Vacant(entry) => {
//     //                             let mut group_idx_vec = Vec::with_capacity(GROUP_VEC_INIT_SIZE);
//     //                             group_idx_vec.push(i);
//     //                             entry.insert((i, group_idx_vec));
//     //                         }
//     //                         Entry::Occupied(mut entry) => {
//     //                             let v = entry.get_mut();
//     //                             v.1.push(i);
//     //                         }
//     //                     }
//     //                 }
//     //             }
//     //             if sort {
//     //                 let mut vec = group_dict.into_values().collect_trusted();
//     //                 vec.sort_unstable_by_key(|v| v.0);
//     //                 vec
//     //             } else {
//     //                 group_dict.into_values().collect_trusted()
//     //             }
//     //         } else {
//     //             let mut group_dict = TpHashMap::<u64, (usize, Vec<usize>)>::with_capacity_and_hasher(
//     //                 thread_init_size,
//     //                 hasher,
//     //             );
//     //             for i in 0..len {
//     //                 let tuple_keys = hashed_keys
//     //                     .iter()
//     //                     .map(|keys| unsafe { *keys.uget(i) })
//     //                     .collect_trusted();
//     //                 let hash = tuple_keys.hash();
//     //                 if partition_here(hash, thread, n_partition) {
//     //                     let entry = group_dict.entry(hash);
//     //                     match entry {
//     //                         Entry::Vacant(entry) => {
//     //                             let mut group_idx_vec = Vec::with_capacity(GROUP_VEC_INIT_SIZE);
//     //                             group_idx_vec.push(i);
//     //                             entry.insert((i, group_idx_vec));
//     //                         }
//     //                         Entry::Occupied(mut entry) => {
//     //                             let v = entry.get_mut();
//     //                             v.1.push(i);
//     //                         }
//     //                     }
//     //                 }
//     //             }
//     //             if sort {
//     //                 let mut vec = group_dict.into_values().collect_trusted();
//     //                 vec.sort_unstable_by_key(|v| v.0);
//     //                 vec
//     //             } else {
//     //                 group_dict.into_values().collect_trusted()
//     //             }
//     //         }
//     //     })
//     //     .collect::<Vec<_>>();
//     // assert!(
//     //     !out.is_empty(),
//     //     "The number of valid group must greater than one"
//     // );
//     // let mut out = if out.len() == 1 {
//     //     out.pop().unwrap()
//     // } else {
//     //     flatten(&out, None)
//     // };
//     // if sort {
//     //     out.sort_unstable_by_key(|v| v.0);
//     // }
//     // out
// }
