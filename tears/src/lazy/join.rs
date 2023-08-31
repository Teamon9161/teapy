use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::hash::{TpBuildHasher, TpHash};
use crate::ArbArray;

use super::super::{Arr1, CollectTrustedToVec, OptUsize};
use super::groupby::{collect_hashmap_keys, collect_hashmap_one_key, prepare_groupby};
use super::{Expr, ExprElement, Exprs, RefType};
use std::collections::hash_map::Entry;

pub enum JoinType {
    Left,
    Right,
    Inner,
    Outer,
}

impl<'a, T> Expr<'a, T>
where
    T: ExprElement + 'a,
{
    pub fn get_left_join_idx(
        self,
        left_other: Option<Vec<Exprs<'a>>>,
        mut right: Vec<Exprs<'a>>,
    ) -> Expr<'a, OptUsize> {
        self.chain_view_f_ct(
            move |(arr, ct)| {
                let left_other = if let Some(mut left_other) = left_other {
                    left_other
                        .par_iter_mut()
                        .chain(right.par_iter_mut())
                        .for_each(|e| {
                            _ = e.eval_inplace(ct.clone());
                        });
                    Some(left_other)
                } else {
                    right.par_iter_mut().for_each(|e| {
                        _ = e.eval_inplace(ct.clone());
                    });
                    None
                };
                // safety: we don't use arr_expr outside the closure
                let arr_expr = unsafe {
                    Expr::<'_, T>::new(
                        std::mem::transmute(Into::<ArbArray<'_, T>>::into(arr)),
                        None,
                    )
                };
                let arr_exprs: Exprs<'_> = arr_expr.into();
                if right.len() > 1 {
                    let left_other = left_other
                        .expect("left_other should be given when right has more than one key");
                    let left_keys = std::iter::once(&arr_exprs)
                        .chain(&left_other)
                        .collect::<Vec<_>>();
                    let right_keys = right.iter().collect::<Vec<_>>();
                    let idx = join_left(&left_keys, &right_keys);
                    Ok((Arr1::from_vec(idx).to_dimd().into(), ct))
                } else {
                    let left_keys = vec![&arr_exprs];
                    let right_keys = right.iter().collect::<Vec<_>>();
                    let idx = join_left(&left_keys, &right_keys);
                    Ok((Arr1::from_vec(idx).to_dimd().into(), ct))
                }
            },
            RefType::False,
        )
    }
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
