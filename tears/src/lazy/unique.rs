use crate::hash::TpHashMap;
use crate::{match_all, match_arrok};
use crate::{Arr1, ArrOk, CollectTrustedToVec, Expr};
use rayon::prelude::*;
use std::collections::hash_map::Entry;

impl<'a> Expr<'a> {
    #[allow(unreachable_patterns)]
    pub fn sorted_unique(&mut self) -> &mut Self {
        self.chain_f(move |data| {
            let arr = data.into_arr(None)?;
            match_arrok!(arr, a, {
                Ok(a.view().to_dim1()?.sorted_unique_1d().to_dimd().into())
            })
        });
        self
    }

    #[allow(unreachable_patterns)]
    pub fn get_sorted_unique_idx(&mut self, keep: String) -> &mut Self {
        self.chain_f(move |arr| {
            let arr = arr.into_arr(None)?;
            match_arrok!(arr, a, {
                Ok(a.view()
                    .to_dim1()?
                    .get_sorted_unique_idx_1d(keep.clone())
                    .to_dimd()
                    .into())
            })
        });
        self
    }

    pub fn get_unique_idx(&mut self, others: Option<Vec<Expr<'a>>>, keep: String) -> &mut Self {
        self.chain_f_ctx(move |(arr, ctx)| {
            let others = others
                .as_ref()
                .map(|vecs| {
                    vecs.into_par_iter()
                        .map(|e| e.view_arr(ctx.as_ref()).unwrap().deref())
                        .collect::<Vec<_>>()
                })
                .unwrap_or(vec![]);
            let others_ref = others.iter().collect::<Vec<_>>();
            let (len, hasher, hashed_keys) = super::prepare_groupby(&others_ref, None);
            let arr = arr.into_arr(ctx.clone())?;
            let arr_key = match_arrok!(hash arr, a, {a.view().to_dim1()?.tphash_1d()});
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
                    let mut map = TpHashMap::<Vec<u64>, u8>::with_capacity_and_hasher(len, hasher);
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
                    let mut map = TpHashMap::<u64, usize>::with_capacity_and_hasher(len, hasher);
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
            Ok((arr.into(), ctx))
        });
        self
    }
}
