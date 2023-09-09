use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::hash::{TpBuildHasher, TpHash};
use crate::ArrOk;

use super::super::{Arr1, CollectTrustedToVec, OptUsize};
use super::groupby::{collect_hashmap_keys, collect_hashmap_one_key, prepare_groupby};
use super::Expr;
use std::collections::hash_map::Entry;

pub enum JoinType {
    Left,
    Right,
    Inner,
    Outer,
}

impl<'a> Expr<'a> {
    pub fn get_left_join_idx(
        &mut self,
        left_other: Option<Vec<Expr<'a>>>,
        right: Vec<Expr<'a>>,
    ) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let left_len = left_other.as_ref().map(|a| a.len()).unwrap_or(0);
            let left_other = left_other.clone();
            let right = right.clone();
            let all_keys = if let Some(left_other) = left_other {
                left_other
                    .into_par_iter()
                    .chain(right.into_par_iter())
                    .map(|a| a.into_arr(ctx.clone()).unwrap())
                    .collect::<Vec<_>>()
                // Some(left_other)
            } else {
                right
                    .into_par_iter()
                    .map(|a| a.into_arr(ctx.clone()).unwrap())
                    .collect::<Vec<_>>()
            };

            let arr = data.view_arr(ctx.as_ref())?;
            let left_keys = std::iter::once(arr)
                .chain(all_keys.iter().take(left_len))
                .collect::<Vec<_>>();
            let right_keys = all_keys.iter().skip(left_len).collect::<Vec<_>>();
            let idx = join_left(&left_keys, &right_keys);
            Ok((Arr1::from_vec(idx).to_dimd().into(), ctx))
        });
        self
    }
}

#[allow(clippy::useless_conversion)]
pub fn join_left<'a>(left_keys: &[&ArrOk<'a>], right_keys: &[&ArrOk<'a>]) -> Vec<OptUsize> {
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
