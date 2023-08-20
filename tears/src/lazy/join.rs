// use crate::OptUsize;

use crate::hash::{TpBuildHasher, TpHash};

use super::super::{CollectTrustedToVec, Exprs, OptUsize};
use super::groupby::{collect_hashmap_keys, collect_hashmap_one_key, prepare_groupby};
use std::collections::hash_map::Entry;

pub enum JoinType {
    Left,
    Right,
    Inner,
    Outer,
}

#[allow(clippy::useless_conversion)]
pub fn join_left(left_keys: &[&Exprs], right_keys: &[&Exprs]) -> Vec<OptUsize> {
    assert_eq!(
        left_keys.len(),
        right_keys.len(),
        "the number of columns given as join key should be equal"
    );
    let hasher = TpBuildHasher::default();
    let (len, hasher, hashed_left_keys) = prepare_groupby(left_keys, Some(hasher));
    let (right_len, hasher, hashed_right_keys) = prepare_groupby(right_keys, Some(hasher));
    let key_len = hashed_left_keys.len();
    let mut output: Vec<OptUsize> = Vec::with_capacity(len);
    // fast path for only one key
    if key_len == 1 {
        let hashed_right_key = hashed_right_keys.get(0).unwrap();
        let mut group_dict_right = collect_hashmap_one_key(right_len, hasher, hashed_right_key);
        let hashed_left_key = &hashed_left_keys[0];
        for i in 0..len {
            let hash = unsafe { *hashed_left_key.uget(i) };
            let entry = group_dict_right.entry(hash);
            match entry {
                Entry::Vacant(_entry) => {
                    output.push(None.into());
                }
                Entry::Occupied(mut entry) => {
                    let v = entry.get_mut();
                    if v.1.len() > 1 {
                        output.push(v.1.pop().unwrap().into());
                    } else {
                        output.push(v.1[0].into());
                    }
                }
            }
        }
    } else {
        let mut group_dict_right = collect_hashmap_keys(right_len, hasher, &hashed_right_keys);
        for i in 0..len {
            let tuple_left_keys = hashed_left_keys
                .iter()
                .map(|keys| unsafe { *keys.uget(i) })
                .collect_trusted();
            let hash = tuple_left_keys.hash();
            let entry = group_dict_right.entry(hash);
            match entry {
                Entry::Vacant(_entry) => {
                    output.push(None.into());
                }
                Entry::Occupied(mut entry) => {
                    let v = entry.get_mut();
                    if v.1.len() > 1 {
                        output.push(v.1.pop().unwrap().into());
                    } else {
                        output.push(v.1[0].into());
                    }
                }
            }
        }
    }
    output
}
