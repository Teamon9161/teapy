use ahash::AHashMap;
use ndarray::Axis;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::hash::TpHash;
use crate::{datatype::Cast, match_all, match_arrok, ArrOk, Context, Data, TpResult};

use super::super::{datatype::OptUsize, Arr1, CollectTrustedToVec};
use super::groupby::{collect_hashmap_keys, collect_hashmap_one_key, prepare_groupby};
use super::Expr;
use std::collections::hash_map::Entry;

pub enum JoinType {
    Left,
    Right,
    Inner,
    Outer,
}

fn collect_left_right_keys<'a, 'r>(
    data: &'r Data<'a>,
    ctx: Option<&'r Context<'a>>,
    left_other: &'r Option<Vec<Expr<'a>>>,
    right: &'r Vec<Expr<'a>>,
) -> TpResult<(Vec<&'r ArrOk<'a>>, Vec<&'r ArrOk<'a>>)> {
    let left_len = left_other.as_ref().map(|a| a.len()).unwrap_or(0);
    let all_keys = if let Some(left_other) = left_other.as_ref() {
        left_other
            .par_iter()
            .chain(right.par_iter())
            .map(|a| a.view_arr(ctx).unwrap())
            .collect::<Vec<_>>()
        // Some(left_other)
    } else {
        right
            .par_iter()
            .map(|a| a.view_arr(ctx).unwrap())
            .collect::<Vec<_>>()
    };

    let arr = data.view_arr(ctx)?;
    let left_keys = std::iter::once(&arr)
        .chain(all_keys.iter().take(left_len))
        .cloned()
        .collect::<Vec<_>>();
    let right_keys = all_keys.into_iter().skip(left_len).collect::<Vec<_>>();
    Ok((left_keys, right_keys))
}

impl<'a> Expr<'a> {
    pub fn get_left_join_idx(
        &mut self,
        left_other: Option<Vec<Expr<'a>>>,
        right: Vec<Expr<'a>>,
    ) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let (left_keys, right_keys) =
                collect_left_right_keys(&data, ctx.as_ref(), &left_other, &right)?;
            let idx = join_left(&left_keys, &right_keys);
            Ok((Arr1::from_vec(idx).to_dimd().into(), ctx))
        });
        self
    }

    #[allow(clippy::clone_on_copy)]
    pub fn get_outer_join_idx(
        &mut self,
        left_other: Option<Vec<Expr<'a>>>,
        right: Vec<Expr<'a>>,
        sort: bool,
        rev: bool,
    ) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let (left_keys, right_keys) =
                collect_left_right_keys(&data, ctx.as_ref(), &left_other, &right)?;
            let (key_idx, outer_dict) = join_outer(&left_keys, &right_keys);
            let (left_idx, right_idx) = key_idx.iter().fold(
                (Vec::new(), Vec::new()),
                |mut acc, (_idx, _is_left, hash)| {
                    acc.0.push(outer_dict.get(hash).unwrap()[0]);
                    acc.1.push(outer_dict.get(hash).unwrap()[1]);
                    acc
                },
            );
            let key_len = left_keys.len();
            let mut outer_keys = Vec::<ArrOk<'a>>::with_capacity(key_len);
            for i in 0..key_len {
                outer_keys.push(match_arrok!(castable left_keys[i], larr, {
                    match_arrok!(castable right_keys[i], rarr, {
                        let arr_left = larr.view().to_dim1().unwrap();
                        let arr_right = rarr.view().to_dim1().unwrap();
                        let a = key_idx.iter().map(|(idx, is_left, _hash)| {
                            unsafe {
                                if *is_left {
                                    arr_left.uget(*idx).clone()
                                } else {
                                    arr_right.uget(*idx).clone().cast()
                                }
                            }
                        }).collect::<Vec<_>>();
                        Arr1::from_vec(a).to_dimd().into()
                    })
                }));
            }
            let mut output = Vec::<ArrOk>::with_capacity(key_len + 2);
            if sort {
                let sort_idx = ArrOk::get_sort_idx(&outer_keys.iter().collect::<Vec<_>>(), rev)?;
                let left_idx = Arr1::from_vec(left_idx)
                    .to_dimd()
                    .select_unchecked(Axis(0), &sort_idx)
                    .0
                    .into_raw_vec();
                let right_idx = Arr1::from_vec(right_idx)
                    .to_dimd()
                    .select_unchecked(Axis(0), &sort_idx)
                    .0
                    .into_raw_vec();
                let slc: ArrOk = Arr1::from_vec(sort_idx).to_dimd().into();
                for key in outer_keys {
                    output.push(key.select(&slc, 0, false)?)
                }
                output.push(Arr1::from_vec(left_idx).to_dimd().into());
                output.push(Arr1::from_vec(right_idx).to_dimd().into());
            } else {
                for key in outer_keys {
                    output.push(key)
                }
                output.push(Arr1::from_vec(left_idx).to_dimd().into());
                output.push(Arr1::from_vec(right_idx).to_dimd().into());
            }
            Ok((output.into(), ctx))
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
    // let hasher = TpBuildHasher::default();
    let hasher = ahash::RandomState::new();
    let (len, hashed_left_keys) = prepare_groupby(left_keys, false);
    let (right_len, hashed_right_keys) = prepare_groupby(right_keys, false);
    let key_len = hashed_left_keys.len();
    let mut output: Vec<OptUsize> = Vec::with_capacity(len);
    // fast path for only one key
    if key_len == 1 {
        let hashed_right_key = hashed_right_keys.get(0).unwrap();
        let mut group_dict_right =
            collect_hashmap_one_key(right_len, hasher, hashed_right_key, Some(right_len));
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
        let mut group_dict_right =
            collect_hashmap_keys(right_len, hasher, &hashed_right_keys, Some(right_len));
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

#[allow(clippy::useless_conversion, clippy::type_complexity)]
pub fn join_outer<'a>(
    left_keys: &[&ArrOk<'a>],
    right_keys: &[&ArrOk<'a>],
) -> (Vec<(usize, bool, u64)>, AHashMap<u64, [OptUsize; 2]>) {
    assert_eq!(
        left_keys.len(),
        right_keys.len(),
        "the number of columns given as join key should be equal"
    );
    // let hasher = TpBuildHasher::default();
    let hasher = ahash::RandomState::new();
    let (len, hashed_left_keys) = prepare_groupby(left_keys, false);
    let (right_len, hashed_right_keys) = prepare_groupby(right_keys, false);
    let outer_capatiy = len.max(right_len);
    let mut outer_dict =
        AHashMap::<u64, [OptUsize; 2]>::with_capacity_and_hasher(outer_capatiy, hasher);
    // the first element is the index of the key and the right table indicates the idx is left or right
    let mut key_idx = Vec::<(usize, bool, u64)>::with_capacity(outer_capatiy);
    // fast path for only one key
    if hashed_left_keys.len() == 1 {
        let hashed_left_key = &hashed_left_keys[0];
        let hashed_right_key = hashed_right_keys.get(0).unwrap();
        for i in 0..len {
            let hash = unsafe { *hashed_left_key.uget(i) };
            let entry = outer_dict.entry(hash);
            match entry {
                Entry::Vacant(entry) => {
                    entry.insert([i.into(), None.into()]);
                    key_idx.push((i, true, hash));
                }
                Entry::Occupied(mut _entry) => {
                    // do not join if the key is duplicated
                }
            }
        }
        for i in 0..right_len {
            let hash = unsafe { *hashed_right_key.uget(i) };
            let entry = outer_dict.entry(hash);
            match entry {
                Entry::Vacant(entry) => {
                    entry.insert([None.into(), i.into()]);
                    key_idx.push((i, false, hash));
                }
                Entry::Occupied(mut entry) => {
                    let v = entry.get_mut();
                    v[1] = i.into();
                    // the key is duplicated, we don't need to push the key_idx
                }
            }
        }
    } else {
        for i in 0..len {
            let tuple_left_keys = hashed_left_keys
                .iter()
                .map(|keys| unsafe { *keys.uget(i) })
                .collect_trusted();
            let hash = tuple_left_keys.hash();
            let entry = outer_dict.entry(hash);
            match entry {
                Entry::Vacant(entry) => {
                    entry.insert([i.into(), None.into()]);
                    key_idx.push((i, true, hash));
                }
                Entry::Occupied(mut _entry) => {
                    // do not join if the key is duplicated
                }
            }
        }
        for i in 0..right_len {
            let tuple_right_keys = hashed_right_keys
                .iter()
                .map(|keys| unsafe { *keys.uget(i) })
                .collect_trusted();
            let hash = tuple_right_keys.hash();
            let entry = outer_dict.entry(hash);
            match entry {
                Entry::Vacant(entry) => {
                    entry.insert([None.into(), i.into()]);
                    key_idx.push((i, false, hash));
                }
                Entry::Occupied(mut entry) => {
                    let v = entry.get_mut();
                    v[1] = i.into();
                    // the key is duplicated, we don't need to push the key_idx
                }
            }
        }
    }
    (key_idx, outer_dict)
}
