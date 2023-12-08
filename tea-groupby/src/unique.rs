use tea_core::prelude::*;
use tea_core::utils::CollectTrustedToVec;

use tea_ext::map::*;
use tea_hash::{TpHashMap, BUILD_HASHER};
use tea_lazy::Expr;
// use once_cell::sync::Lazy;
use crate::HashExt1d;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::hash_map::Entry;

#[ext_trait]
impl<'a> ExprUniqueExt for Expr<'a> {
    #[allow(unreachable_patterns)]
    pub fn sorted_unique(&mut self) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.into_arr(ctx.clone())?;
            match_arrok!(arr, a, {
                Ok((a.view().to_dim1()?.sorted_unique_1d().to_dimd().into(), ctx))
            })
        });
        self
    }

    #[allow(unreachable_patterns)]
    pub fn get_sorted_unique_idx(&mut self, keep: String) -> &mut Self {
        self.chain_f_ctx(move |(arr, ctx)| {
            let arr = arr.into_arr(ctx.clone())?;
            match_arrok!(arr, a, {
                Ok((
                    a.view()
                        .to_dim1()?
                        .get_sorted_unique_idx_1d(keep.clone())
                        .to_dimd()
                        .into(),
                    ctx,
                ))
            })
        });
        self
    }

    #[allow(suspicious_double_ref_op, clippy::clone_on_copy)]
    pub fn get_unique_idx(&mut self, others: Option<Vec<Expr<'a>>>, keep: String) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let others = others
                .as_ref()
                .map(|vecs| {
                    vecs.into_par_iter()
                        .map(|e| e.view_arr(ctx.as_ref()).unwrap().deref())
                        .collect::<Vec<_>>()
                })
                .unwrap_or(vec![]);
            let others_ref = others.iter().collect::<Vec<_>>();
            let arr = data.view_arr(ctx.as_ref()).unwrap();
            let len = arr.len();
            let out_idx = if others_ref.is_empty() {
                if &keep == "first" {
                    let mut out_idx = Vec::with_capacity(len);
                    let arr: ArrOk = if arr.is_float() {
                        match_arrok!(float arr, a, {a.view().to_dim1()?.tphash_1d().to_dimd().into()})
                    } else {
                        arr.deref()
                    };
                    match_arrok!(hash arr, a, {
                        let a = a.view().to_dim1()?;
                        if a.dtype().is_float() {
                            let a = a.tphash_1d();
                            let mut map = TpHashMap::<_, u8>::with_capacity_and_hasher(len, BUILD_HASHER.clone());
                            for i in 0..len {
                                let entry = map.entry(unsafe { a.uget(i)}.clone());
                                if let Entry::Vacant(entry) = entry {
                                    entry.insert(1);
                                    out_idx.push(i);
                                }
                            }
                        } else {
                            let mut map = TpHashMap::<_, u8>::with_capacity_and_hasher(len, BUILD_HASHER.clone());
                            for i in 0..len {
                                let entry = map.entry(unsafe { a.uget(i)}.clone());
                                if let Entry::Vacant(entry) = entry {
                                    entry.insert(1);
                                    out_idx.push(i);
                                }
                            }
                        }
                    });
                    out_idx
                } else if &keep == "last" {
                    match_arrok!(hash arr, a, {
                        let a = a.view().to_dim1()?;
                        let mut map = TpHashMap::<_, usize>::with_capacity_and_hasher(len, BUILD_HASHER.clone());
                        for i in 0..len {
                            let entry = map.entry(unsafe { a.uget(i)}.clone());
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
                        let mut out_idx = map.into_values().collect_trusted();
                        out_idx.sort_unstable();
                        out_idx
                    })
                } else {
                    return Err("keep must be either first or last".into());
                }
            } else {
                let (len, hashed_keys) = super::prepare_groupby(&others_ref, false);
                let arr = data.view_arr(ctx.as_ref())?;
                let arr_key = match_arrok!(tphash arr, a, {a.view().to_dim1()?.tphash_1d()});
                let mut out_idx = Vec::with_capacity(len);
                if &keep == "first" {
                    let mut map = TpHashMap::<Vec<u64>, u8>::with_capacity_and_hasher(len, BUILD_HASHER.clone());
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
                } else if &keep == "last" {
                    let mut map =
                        TpHashMap::<Vec<u64>, usize>::with_capacity_and_hasher(len, BUILD_HASHER.clone());
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
                } else {
                    return Err("keep must be either first or last".into());
                }
                out_idx
            };
            let arr = Arr1::from_vec(out_idx).to_dimd();
            Ok((arr.into(), ctx))
        });
        self
    }
}
