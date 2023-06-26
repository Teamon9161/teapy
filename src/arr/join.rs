use super::{
    groupby::{collect_hashmap_keys, collect_hashmap_one_key, prepare_groupby},
    CollectTrustedToVec, Exprs,
};
use ahash::RandomState;
use std::collections::hash_map::RawEntryMut;

pub enum JoinType {
    Left,
    Right,
    Inner,
    Outer,
}

// static JOIN_DICT_INIT_SIZE: usize = 512;
// static JOIN_VEC_INIT_SIZE: usize = 32;

pub fn join_left(left_keys: Vec<&Exprs>, right_keys: Vec<&Exprs>) -> Vec<Option<usize>> {
    assert_eq!(
        left_keys.len(),
        right_keys.len(),
        "the number of columns given as join key should be equal"
    );
    let hasher = RandomState::new();
    let (len, hasher, hashed_left_keys) = prepare_groupby(&left_keys, Some(hasher));
    let (right_len, hasher, hashed_right_keys) = prepare_groupby(&right_keys, Some(hasher));
    let key_len = hashed_left_keys.len();
    let mut output = Vec::with_capacity(len);
    // fast path for only one key
    if key_len == 1 {
        let hashed_right_key = hashed_right_keys.get(0).unwrap();
        let mut group_dict_right = collect_hashmap_one_key(right_len, hasher, hashed_right_key);
        let hashed_left_key = &hashed_left_keys[0];
        for i in 0..len {
            let hash = unsafe { *hashed_left_key.uget(i) };
            let entry = group_dict_right
                .raw_entry_mut()
                .from_key_hashed_nocheck(hash, &hash);
            match entry {
                RawEntryMut::Vacant(_entry) => {
                    output.push(None);
                }
                RawEntryMut::Occupied(mut entry) => {
                    let v = entry.get_mut();
                    if v.1.len() > 1 {
                        output.push(Some(v.1.pop().unwrap()));
                    } else {
                        output.push(Some(v.1[0]));
                    }
                }
            }
        }
    } else {
        let mut group_dict_right =
            collect_hashmap_keys(right_len, hasher.clone(), &hashed_right_keys);
        for i in 0..len {
            let tuple_left_keys = hashed_left_keys
                .iter()
                .map(|keys| unsafe { *keys.uget(i) })
                .collect_trusted();
            let hash = hasher.hash_one(&tuple_left_keys);
            let entry = group_dict_right
                .raw_entry_mut()
                .from_key_hashed_nocheck(hash, &tuple_left_keys);
            match entry {
                RawEntryMut::Vacant(_entry) => {
                    output.push(None);
                }
                RawEntryMut::Occupied(mut entry) => {
                    let v = entry.get_mut();
                    if v.1.len() > 1 {
                        output.push(Some(v.1.pop().unwrap()));
                    } else {
                        output.push(Some(v.1[0]));
                    }
                }
            }
        }
    }
    output
}
