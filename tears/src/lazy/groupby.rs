use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use super::super::export::*;
use crate::{
    hash::{TpBuildHasher, TpHash, TpHashMap},
    match_all, match_arrok,
};
use std::collections::hash_map::Entry;

/// Get the partition size for parallel
pub fn get_partition_size() -> usize {
    let mut n_partitions = rayon::current_num_threads();
    // set n_partitions to closes 2^n above the no of threads.
    loop {
        if n_partitions.is_power_of_two() {
            break;
        } else {
            n_partitions += 1;
        }
    }
    n_partitions
}

#[inline]
/// For partition that are a power of 2 we can use a bitshift instead of a modulo.
pub(crate) fn partition_here(h: u64, thread: u64, n_partition: u64) -> bool {
    debug_assert!(n_partition.is_power_of_two());
    // n % 2^i = n & (2^i - 1)
    (h.wrapping_add(thread)) & n_partition.wrapping_sub(1) == 0
}

// Faster than collecting from a flattened iterator.
pub fn flatten<T: Clone, R: AsRef<[T]>>(bufs: &[R], len: Option<usize>) -> Vec<T> {
    let len = len.unwrap_or_else(|| bufs.iter().map(|b| b.as_ref().len()).sum());
    let mut out = Vec::with_capacity(len);
    for b in bufs {
        out.extend_from_slice(b.as_ref());
    }
    out
}

static GROUP_DICT_INIT_SIZE: usize = 512;
static GROUP_VEC_INIT_SIZE: usize = 32;

pub fn prepare_groupby(
    keys: &[&ArrOk<'_>],
    hasher: Option<TpBuildHasher>,
) -> (usize, TpBuildHasher, Vec<Arr1<u64>>) {
    let hasher = hasher.unwrap_or(TpBuildHasher::default());
    let hashed_keys = keys
        .iter()
        .map(|arr| {
            match_arrok!(
                hash
                arr,
                a,
                {
                    a.view()
                        .to_dim1()
                        .expect("groupby key should be dim1")
                        .tphash_1d()
                }
            )
        })
        .collect_trusted();
    if keys.is_empty() {
        return (0, hasher, hashed_keys);
    }
    let len = hashed_keys[0].len();
    for key in &hashed_keys {
        if key.len() != len {
            panic!("All of the groupby keys should have the same shape")
        }
    }
    (len, hasher, hashed_keys)
}

pub(super) fn collect_hashmap_one_key(
    len: usize,
    hasher: TpBuildHasher,
    hashed_key: &Arr1<u64>,
) -> TpHashMap<u64, (usize, Vec<usize>)> {
    let mut group_dict = TpHashMap::<u64, (usize, Vec<usize>)>::with_capacity_and_hasher(
        GROUP_DICT_INIT_SIZE,
        hasher,
    );
    for i in 0..len {
        let hash = unsafe { *hashed_key.uget(i) };
        let entry = group_dict.entry(hash);
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

pub(super) fn collect_hashmap_keys(
    len: usize,
    hasher: TpBuildHasher,
    hashed_keys: &[Arr1<u64>],
) -> TpHashMap<u64, (usize, Vec<usize>)> {
    let mut group_dict = TpHashMap::<u64, (usize, Vec<usize>)>::with_capacity_and_hasher(
        GROUP_DICT_INIT_SIZE,
        hasher,
    );
    for i in 0..len {
        let tuple_keys = hashed_keys
            .iter()
            .map(|keys| unsafe { *keys.uget(i) })
            .collect_trusted();
        let entry = group_dict.entry(tuple_keys.hash());
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

pub fn groupby(keys: &[&ArrOk<'_>], sort: bool) -> Vec<(usize, Vec<usize>)> {
    let (len, hasher, hashed_keys) = prepare_groupby(keys, None);
    // fast path for only one key
    let mut vec = if hashed_keys.len() == 1 {
        let group_dict = collect_hashmap_one_key(len, hasher, &hashed_keys[0]);
        group_dict.into_values().collect_trusted()
    } else {
        let group_dict = collect_hashmap_keys(len, hasher, &hashed_keys);
        group_dict.into_values().collect_trusted()
    };
    if sort {
        vec.sort_unstable_by_key(|v| v.0);
    }
    vec
    // unimplemented!()
}

// pub fn groupby_apply(&mut self, others: Vec<Expr<'a>>, by: Vec<Expr<'a>>)

/// Groupby this array, return a `vec` contains
/// index of the first value in each group and a `vec` contains the
/// index of values within each group.
///
/// sort: whether to sort on index of the first value in each group
pub fn groupby_par(keys: &[&ArrOk<'_>], sort: bool) -> Vec<(usize, Vec<usize>)> {
    let n_partition = get_partition_size() as u64;
    let (len, hasher, hashed_keys) = prepare_groupby(keys, None);
    let mut out = (0..n_partition)
        .into_par_iter()
        .map(|thread| {
            let hasher = hasher.clone();
            // fast path for only one key
            if hashed_keys.len() == 1 {
                let hashed_key = &hashed_keys[0];
                let mut group_dict =
                    TpHashMap::<u64, (usize, Vec<usize>)>::with_capacity_and_hasher(
                        GROUP_DICT_INIT_SIZE,
                        hasher,
                    );
                for i in 0..len {
                    let hash = unsafe { *hashed_key.uget(i) };
                    if partition_here(hash, thread, n_partition) {
                        let entry = group_dict.entry(hash);
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
                }
                if sort {
                    let mut vec = group_dict.into_values().collect_trusted();
                    vec.sort_unstable_by_key(|v| v.0);
                    vec
                } else {
                    group_dict.into_values().collect_trusted()
                }
            } else {
                let mut group_dict =
                    TpHashMap::<u64, (usize, Vec<usize>)>::with_capacity_and_hasher(
                        GROUP_DICT_INIT_SIZE,
                        hasher,
                    );
                for i in 0..len {
                    let tuple_keys = hashed_keys
                        .iter()
                        .map(|keys| unsafe { *keys.uget(i) })
                        .collect_trusted();
                    let hash = tuple_keys.hash();
                    if partition_here(hash, thread, n_partition) {
                        let entry = group_dict.entry(hash);
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
                }
                if sort {
                    let mut vec = group_dict.into_values().collect_trusted();
                    vec.sort_unstable_by_key(|v| v.0);
                    vec
                } else {
                    group_dict.into_values().collect_trusted()
                }
            }
        })
        .collect::<Vec<_>>();
    assert!(
        !out.is_empty(),
        "The number of valid group must greater than one"
    );
    let mut out = if out.len() == 1 {
        out.pop().unwrap()
    } else {
        flatten(&out, None)
    };
    if sort {
        out.sort_unstable_by_key(|v| v.0);
    }
    out
}
