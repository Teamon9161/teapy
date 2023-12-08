use ndarray::Axis;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

// use once_cell::sync::Lazy;
use crate::{collect_hashmap_keys, prepare_groupby, GROUP_VEC_INIT_SIZE};
use tea_core::prelude::*;
use tea_core::utils::CollectTrustedToVec;
use tea_hash::{TpHash, TpHashMap, BUILD_HASHER};

use std::collections::hash_map::Entry;

use tea_ext::ArrOkExt;
use tea_lazy::{Context, Data, Expr};

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

#[ext_trait]
impl<'a> ExprJoinExt for Expr<'a> {
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
            let key_len = left_keys.len();
            let (mut outer_keys, left_idx, right_idx) = join_outer(&left_keys, &right_keys);
            let output = if sort {
                let mut output = Vec::<ArrOk>::with_capacity(key_len + 2);
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
                output
            } else {
                outer_keys.push(Arr1::from_vec(left_idx).to_dimd().into());
                outer_keys.push(Arr1::from_vec(right_idx).to_dimd().into());
                outer_keys
            };
            Ok((output.into(), ctx))
        });
        self
    }
}

#[allow(suspicious_double_ref_op, clippy::clone_on_copy)]
pub fn join_left<'a>(left_keys: &[&ArrOk<'a>], right_keys: &[&ArrOk<'a>]) -> Vec<OptUsize> {
    assert_eq!(
        left_keys.len(),
        right_keys.len(),
        "the number of columns given as join key should be equal"
    );
    let key_len = left_keys.len();
    if key_len == 0 {
        panic!("the number of columns given as join key should be greater than 0")
    }
    let len = left_keys[0].len();
    let right_len = right_keys[0].len();
    let init_size = (right_len / 2).min(1);
    // check the length of left keys and right keys are equal
    for key in left_keys.iter().skip(1) {
        if key.len() != len {
            panic!(
                "the length of left keys should be equal, but the length of left key is different"
            )
        }
    }
    for key in right_keys.iter().skip(1) {
        if key.len() != right_len {
            panic!("the length of right keys should be equal, but the length of right key is different")
        }
    }
    let mut output: Vec<OptUsize> = Vec::with_capacity(len);
    // fast path for only one key
    if key_len == 1 {
        match_arrok!(hash left_keys[0], lk_arr, {
            match_arrok!(hash right_keys[0], rk_arr, {
                if lk_arr.dtype() != rk_arr.dtype() {
                    panic!("the dtype of left key and right key should be equal")
                }
                let lk_arr = lk_arr.cast_ref_with(rk_arr).view().to_dim1().unwrap();
                let rk_arr = rk_arr.view().to_dim1().unwrap();
                // collect right keys as a hashmap
                let mut group_dict_right =
                    TpHashMap::<_, (usize, Vec<usize>)>::with_capacity_and_hasher(init_size, BUILD_HASHER.clone());
                for i in 0..right_len {
                    let value = unsafe { rk_arr.uget(i) }.clone();
                    let entry = group_dict_right.entry(value);
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
                for i in 0..len {
                    let left_key = unsafe { lk_arr.uget(i) }.clone();
                    let entry = group_dict_right.entry(left_key);
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
            })
        });
    } else {
        let (len, hashed_left_keys) = prepare_groupby(left_keys, false);
        let (right_len, hashed_right_keys) = prepare_groupby(right_keys, false);
        let mut group_dict_right =
            collect_hashmap_keys(right_len, &hashed_right_keys, Some(right_len));
        for i in 0..len {
            let tuple_left_keys = hashed_left_keys
                .iter()
                .map(|keys| unsafe { *keys.uget(i) })
                .collect_trusted();
            let hash = tuple_left_keys.tphash();
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

// #[allow(clippy::useless_conversion, clippy::type_complexity)]
#[allow(suspicious_double_ref_op, clippy::clone_on_copy)]
/// return outer_keys, left_idx and right_idx to select from left and right table
pub fn join_outer<'a>(
    left_keys: &[&ArrOk<'a>],
    right_keys: &[&ArrOk<'a>],
) -> (Vec<ArrOk<'a>>, Vec<OptUsize>, Vec<OptUsize>) {
    assert_eq!(
        left_keys.len(),
        right_keys.len(),
        "the number of columns given as join key should be equal"
    );
    let key_len = left_keys.len();
    if key_len == 0 {
        panic!("the number of columns given as join key should be greater than 0")
    }
    let len = left_keys[0].len();
    let right_len = right_keys[0].len();
    for key in left_keys.iter().skip(1) {
        if key.len() != len {
            panic!(
                "the length of left keys should be equal, but the length of left key is different"
            )
        }
    }
    for key in right_keys.iter().skip(1) {
        if key.len() != right_len {
            panic!("the length of right keys should be equal, but the length of right key is different")
        }
    }
    let outer_capatiy = len.max(right_len);
    let mut outer_keys = Vec::<ArrOk<'a>>::with_capacity(key_len);

    // fast path for only one key
    let (left_idx, right_idx): (Vec<_>, Vec<_>) = if key_len == 1 {
        match_arrok!(hash left_keys[0], lk_arr, {
            match_arrok!(hash right_keys[0], rk_arr, {
                if lk_arr.dtype() != rk_arr.dtype() {
                    panic!("the dtype of left key and right key should be equal")
                }
                // the first element is the index of the key and the right table indicates the idx is left or right
                let mut key_idx = Vec::<(usize, bool, _)>::with_capacity(outer_capatiy);
                let mut outer_dict =
                    TpHashMap::<_, (OptUsize, OptUsize)>::with_capacity_and_hasher(outer_capatiy, BUILD_HASHER.clone());
                let lk_arr = lk_arr.cast_ref_with(rk_arr).view().to_dim1().unwrap();
                let rk_arr = rk_arr.view().to_dim1().unwrap();

                for i in 0..len {
                    let left_value = unsafe { lk_arr.uget(i) }.clone();
                    let entry = outer_dict.entry(left_value.clone());
                    match entry {
                        Entry::Vacant(entry) => {
                            entry.insert((i.into(), None.into()));
                            key_idx.push((i, true, left_value));
                        }
                        Entry::Occupied(mut _entry) => {
                            // do not join if the key is duplicated
                        }
                    }
                }
                for i in 0..right_len {
                    let right_value = unsafe { rk_arr.uget(i) }.clone();
                    let entry = outer_dict.entry(right_value.clone());
                    match entry {
                        Entry::Vacant(entry) => {
                            entry.insert((None.into(), i.into()));
                            key_idx.push((i, false, right_value));
                        }
                        Entry::Occupied(mut entry) => {
                            let v = entry.get_mut();
                            // the key is duplicated, we don't need to push the key_idx
                            v.1 = i.into();
                        }
                    }
                }
                let outer_key = key_idx.iter().map(|(idx, is_left, _value)| {
                    unsafe {
                        if *is_left {
                            lk_arr.uget(*idx).clone()
                        } else {
                            rk_arr.uget(*idx).clone()
                        }
                    }
                }).collect_trusted();
                outer_keys.push(Arr1::from_vec(outer_key).to_dimd().into());
                key_idx.iter().map(|(_idx, _is_left, value)| {
                    outer_dict.get(value).unwrap().clone()
                }).unzip()
            })
        })
    } else {
        let (len, hashed_left_keys) = prepare_groupby(left_keys, false);
        let (right_len, hashed_right_keys) = prepare_groupby(right_keys, false);
        let mut key_idx = Vec::<(usize, bool, _)>::with_capacity(outer_capatiy);
        let mut outer_dict = TpHashMap::<_, (OptUsize, OptUsize)>::with_capacity_and_hasher(
            outer_capatiy,
            BUILD_HASHER.clone(),
        );
        for i in 0..len {
            let tuple_left_keys = hashed_left_keys
                .iter()
                .map(|keys| unsafe { *keys.uget(i) })
                .collect_trusted();
            let hash = tuple_left_keys.tphash();
            let entry = outer_dict.entry(hash);
            match entry {
                Entry::Vacant(entry) => {
                    entry.insert((i.into(), None.into()));
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
            let hash = tuple_right_keys.tphash();
            let entry = outer_dict.entry(hash);
            match entry {
                Entry::Vacant(entry) => {
                    entry.insert((None.into(), i.into()));
                    key_idx.push((i, false, hash));
                }
                Entry::Occupied(mut entry) => {
                    let v = entry.get_mut();
                    v.1 = i.into();
                    // the key is duplicated, we don't need to push the key_idx
                }
            }
        }
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
                    }).collect_trusted();
                    Arr1::from_vec(a).to_dimd().into()
                })
            }));
        }
        key_idx
            .iter()
            .map(|(_idx, _is_left, value)| outer_dict.get(value).unwrap().clone())
            .unzip()
    };
    (outer_keys, left_idx, right_idx)
}

// #[cfg(test)]
// mod test {
//     use super::*;
//     use tea_core::prelude::*;
//     #[test]
//     fn test_join_left() {
//         let left: ArrOk = Arr1::from_vec(vec!["a", "b", "a", "d"]).to_dimd().into();
//         let right: ArrOk = Arr1::from_vec(vec!["b", "b", "c", "e"]).to_dimd().into();
//         let idx = join_left(&[&left], &[&right]);
//     }
// }
