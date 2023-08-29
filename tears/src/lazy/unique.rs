use super::expr::{Expr, ExprElement, RefType};
use super::exprs::Exprs;
use crate::hash::{TpHash, TpHashMap};
use crate::{Arr1, CollectTrustedToVec};
use rayon::prelude::*;
use std::collections::hash_map::Entry;

impl<'a, T> Expr<'a, T>
where
    T: ExprElement + 'a,
{
    pub fn sorted_unique(self) -> Self
    where
        T: PartialEq + Clone,
    {
        self.chain_view_f(
            move |arr| {
                let arr = arr.to_dim1()?;
                let out = arr.sorted_unique_1d();
                Ok(out.to_dimd().into())
            },
            RefType::False,
        )
    }

    pub fn get_sorted_unique_idx(self, keep: String) -> Expr<'a, usize>
    where
        T: PartialEq,
    {
        self.chain_view_f(
            move |arr| {
                let arr = arr.to_dim1()?;
                let out = arr.get_sorted_unique_idx_1d(keep);
                Ok(out.to_dimd().into())
            },
            RefType::False,
        )
    }

    pub fn get_unique_idx(self, others: Option<Vec<Exprs<'a>>>, keep: String) -> Expr<'a, usize>
    where
        T: TpHash,
    {
        self.chain_view_f_ct(
            move |(arr, ct)| {
                let others = others
                    .map(|mut x| {
                        x.par_iter_mut()
                            .try_for_each(|e| e.eval_inplace(ct.clone()).map(|_| {}))
                            .unwrap();
                        x
                    })
                    .unwrap_or(vec![]);
                let others_ref = others.iter().collect::<Vec<_>>();
                let (len, hasher, hashed_keys) = super::prepare_groupby(&others_ref, None);
                let arr_key = arr.to_dim1()?.tphash_1d();
                let mut out_idx = Vec::with_capacity(len);
                if &keep == "first" {
                    if hashed_keys.is_empty() {
                        let mut map = TpHashMap::<u64, u8>::with_capacity_and_hasher(len, hasher);
                        let len = arr_key.len();
                        for i in 0..len {
                            let entry = map.entry(unsafe { *arr_key.uget(i) });
                            if let Entry::Vacant(entry) = entry {
                                entry.insert(1);
                                out_idx.push(i);
                            }
                        }
                    } else {
                        let mut map =
                            TpHashMap::<Vec<u64>, u8>::with_capacity_and_hasher(len, hasher);
                        for i in 0..len {
                            let tuple_keys = hashed_keys
                                .iter()
                                .chain(std::iter::once(&arr_key))
                                .map(|keys| unsafe { *keys.uget(i) })
                                .collect_trusted();
                            let entry = map.entry(tuple_keys);
                            if let Entry::Vacant(entry) = entry {
                                entry.insert(1);
                                out_idx.push(i);
                            }
                        }
                    }
                } else if &keep == "last" {
                    if hashed_keys.is_empty() {
                        let mut map =
                            TpHashMap::<u64, usize>::with_capacity_and_hasher(len, hasher);
                        let hashed_key = &arr_key;
                        let len = hashed_key.len();
                        for i in 0..len {
                            let hash = unsafe { *hashed_key.uget(i) };
                            let entry = map.entry(hash);
                            match entry {
                                Entry::Vacant(entry) => {
                                    entry.insert(i);
                                }
                                Entry::Occupied(mut entry) => {
                                    let v = entry.get_mut();
                                    *v = i;
                                }
                            }
                        }
                        out_idx = map.into_values().collect_trusted();
                        out_idx.sort_unstable()
                    } else {
                        let mut map =
                            TpHashMap::<Vec<u64>, usize>::with_capacity_and_hasher(len, hasher);
                        for i in 0..len {
                            let tuple_keys = hashed_keys
                                .iter()
                                .chain(std::iter::once(&arr_key))
                                .map(|keys| unsafe { *keys.uget(i) })
                                .collect_trusted();
                            let entry = map.entry(tuple_keys);
                            match entry {
                                Entry::Vacant(entry) => {
                                    entry.insert(i);
                                }
                                Entry::Occupied(mut entry) => {
                                    let v = entry.get_mut();
                                    *v = i;
                                }
                            }
                        }
                        out_idx = map.into_values().collect_trusted();
                        out_idx.sort_unstable()
                    }
                } else {
                    return Err("keep must be either first or last".into());
                }
                let arr = Arr1::from_vec(out_idx).to_dimd();
                Ok((arr.into(), ct))
            },
            RefType::False,
        )
    }
}
