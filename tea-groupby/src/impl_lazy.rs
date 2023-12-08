use crate::groupby;
use ndarray::Axis;
use rayon::prelude::*;
use std::sync::Arc;
use tea_core::prelude::*;
use tea_core::utils::CollectTrustedToVec;
use tea_hash::TpHashMap;
use tea_lazy::{DataDict, Expr};

#[ext_trait]
impl<'a> ExprGroupByExt for Expr<'a> {
    pub fn get_group_by_idx(&mut self, others: Vec<Expr<'a>>, sort: bool, par: bool) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?;
            let others_ref = others
                .par_iter()
                .map(|e| e.view_arr(ctx.as_ref()).unwrap())
                .collect::<Vec<_>>();
            let keys = std::iter::once(arr)
                .chain(others_ref.into_iter())
                .collect::<Vec<_>>();
            let group_idx = if par {
                // groupby_par(&keys, sort)
                unimplemented!()
            } else {
                groupby(&keys, sort)
            };
            let output = group_idx.into_iter().map(|v| v.1).collect_trusted();
            let output = Arr1::from_vec(output).to_dimd();
            Ok((output.into(), ctx))
        });
        self
    }

    #[allow(unreachable_patterns)]
    pub fn apply_with_vecusize(
        &mut self,
        agg_expr: Expr<'a>,
        idxs: Expr<'a>,
        others: Vec<Expr<'a>>,
    ) -> &mut Self {
        let name = self.name().unwrap();
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?.deref();
            let others_ref = others
                .par_iter()
                .map(|e| e.view_arr(ctx.as_ref()).unwrap().deref())
                .collect::<Vec<_>>();
            let others_name = others.iter().map(|e| e.name().unwrap()).collect_trusted();
            let idxs = idxs.view_arr(ctx.as_ref())?.deref().cast_vecusize();
            let idxs_arr = idxs.view().to_dim1()?;
            let columns = std::iter::once(name.clone())
                .chain(others_name)
                .collect::<Vec<_>>();
            let mut map: Option<Arc<TpHashMap<String, usize>>> = None;
            let agg_expr = agg_expr.flatten();
            let out: ArrOk<'a> = match_arrok!(arr, arr, {
                let arr = arr.view().to_dim1()?;
                let out = idxs_arr
                    .iter() // do not use into_iter here, as we need to keep the idxs_arr alive until we use to_owned
                    .map(|idx| {
                        let current_arr = arr.select_unchecked(Axis(0), idx);
                        let current_others: Vec<ArrOk> = others_ref
                            .iter()
                            .map(|arr| {
                                match_arrok!(arr, o, {
                                    o.view()
                                        .to_dim1()
                                        .unwrap()
                                        .select_unchecked(Axis(0), idx)
                                        .to_dimd()
                                        .into()
                                })
                            })
                            .collect_trusted();
                        let exprs: Vec<Expr<'_>> = std::iter::once(current_arr.to_dimd().into())
                            .chain(current_others.into_iter().map(|a| a.into()))
                            .collect::<Vec<_>>();
                        let current_ctx = if map.is_some() {
                            DataDict {
                                data: exprs,
                                map: map.clone().unwrap(),
                            }
                        } else {
                            let dd = DataDict::new(exprs, Some(columns.clone()));
                            map = Some(dd.map.clone());
                            dd
                        };
                        let out_e = agg_expr.context_clone();
                        // this is safe as we don't return a view on the current context
                        // into_owned is important here to guarantee the above
                        let current_ctx: Arc<DataDict> =
                            Arc::new(unsafe { std::mem::transmute(current_ctx) });
                        let o = out_e
                            .view_arr(Some(&current_ctx))
                            .unwrap()
                            .deref()
                            .into_owned();
                        o
                    })
                    .collect_trusted();
                ArrOk::same_dtype_concat_1d(out)
            });
            Ok((out.into(), ctx.clone()))
        });
        self
    }
}
