use ndarray::s;
use std::sync::Arc;
use tea_core::prelude::*;
use tea_core::utils::CollectTrustedToVec;
use tea_ext::agg::*;
use tea_hash::TpHashMap;
use tea_lazy::{Data, DataDict, Expr};

#[ext_trait]
impl<'a> GroupbyAggExt for Expr<'a> {
    /// This func should return an array indicates the start of each group
    /// and a label array indicates the label of each group
    pub fn get_group_by_time_idx<TD>(&mut self, duration: TD, closed: String) -> &mut Self
    where
        TD: Into<TimeDelta>,
    {
        let duration: TimeDelta = duration.into();
        self.chain_f_ctx(move |(data, ctx)| {
            let closed = closed.clone();
            let arr = data.view_arr(ctx.as_ref())?.deref().cast_datetime_default();
            let ts = arr.view().to_dim1()?;
            if ts.is_empty() {
                let label = Arr1::from_vec(Vec::<DateTime>::with_capacity(0))
                    .to_dimd()
                    .into();
                let start_vec = Arr1::from_vec(Vec::<usize>::with_capacity(0))
                    .to_dimd()
                    .into();
                return Ok((Data::ArrVec(vec![label, start_vec]), ctx));
            }

            let mut label = vec![];
            let mut start_vec = vec![];
            match closed.to_lowercase().as_str() {
                "left" => {
                    let mut start = ts.first_1d().duration_trunc(duration.clone());
                    label.push(start);
                    start_vec.push(0);
                    for i in 0..ts.len() {
                        let t = unsafe { *ts.uget(i) };
                        if t < start + duration.clone() {
                            continue;
                        } else {
                            start_vec.push(i);
                            start = t.duration_trunc(duration.clone());
                            label.push(start);
                        }
                    }
                }
                "right" => {
                    let mut start = ts.first_1d().duration_trunc(duration.clone());
                    if start == *ts.get(0).unwrap() {
                        start = start - duration.clone();
                    }
                    label.push(start);
                    start_vec.push(0);
                    for i in 0..ts.len() {
                        let t = unsafe { *ts.uget(i) };
                        if (t <= start + duration.clone()) | (t == start) {
                            continue;
                        } else {
                            start_vec.push(i);
                            start = t.duration_trunc(duration.clone());
                            if start == t {
                                start = start - duration.clone();
                            }
                            label.push(start);
                        }
                    }
                }
                _ => unimplemented!(),
            }
            start_vec.push(ts.len()); // the end of the array, this element is not the start of the group
            let label: ArrOk<'a> = Arr1::from_vec(label).to_dimd().into();
            let start_vec: ArrOk<'a> = Arr1::from_vec(start_vec).to_dimd().into();
            Ok((Data::ArrVec(vec![label, start_vec]), ctx))
        });
        self
    }

    #[allow(unreachable_patterns)]
    pub fn group_by_startidx(
        &mut self,
        agg_expr: Expr<'a>,
        start_idx: Expr<'a>,
        others: Vec<Expr<'a>>,
    ) -> &mut Self {
        let name = self.name().unwrap();
        self.chain_f_ctx(move |(data, ctx)| {
            let arr = data.view_arr(ctx.as_ref())?.deref();
            let others_ref = others
                .iter()
                .map(|e| e.view_arr(ctx.as_ref()).unwrap().deref())
                .collect_trusted();
            let others_name = others.iter().map(|e| e.name().unwrap()).collect_trusted();
            let group_start = if let Ok(mut start_idx) = start_idx.view_arr_vec(ctx.as_ref()) {
                start_idx.pop().unwrap().deref().cast_usize()
            } else {
                start_idx.view_arr(ctx.as_ref())?.deref().cast_usize()
            };
            let group_start_view = group_start.view().to_dim1()?;
            let columns = std::iter::once(name.clone())
                .chain(others_name)
                .collect::<Vec<_>>();
            let mut map: Option<Arc<TpHashMap<String, usize>>> = None;
            let mut agg_expr = agg_expr.flatten();
            agg_expr.simplify();
            let init_data = agg_expr.get_chain_base();
            let nodes = agg_expr.collect_chain_nodes(vec![]);
            let out: ArrOk<'a> = match_arrok!(arr, arr, {
                let arr = arr.view().to_dim1()?;
                let out = group_start_view
                    .as_slice()
                    .unwrap()
                    .windows(2)
                    .map(|v| {
                        let (start, next_start) = (v[0], v[1]);
                        let exprs: Vec<Expr<'_>> = if others_ref.is_empty() {
                            vec![arr.slice(s![start..next_start]).to_dimd().into()]
                        } else {
                            let slice = s![start..next_start];
                            std::iter::once(arr.slice(slice).to_dimd().into())
                                .chain(others_ref.iter().map(|a| a.slice(slice).into()))
                                .collect::<Vec<_>>()
                        };
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
                        // let out_e = agg_expr.context_clone();
                        // this is safe as we don't return a view on the current context
                        // into_owned is important here to guarantee the above
                        let current_ctx: Arc<DataDict> = Arc::new(unsafe {
                            std::mem::transmute::<DataDict<'_>, DataDict<'a>>(current_ctx)
                        });
                        let mut data = init_data.clone();
                        let mut ctx = Some(current_ctx);
                        for f in &nodes {
                            (data, ctx) = f((data, ctx)).unwrap();
                        }
                        data.into_arr(ctx).unwrap()
                        // let o = out_e
                        //     .view_arr(Some(&current_ctx))
                        //     .unwrap()
                        //     .deref()
                        //     .into_owned();
                        // o
                    })
                    .collect_trusted();
                ArrOk::same_dtype_concat_1d(out)
            });
            Ok((out.into(), ctx.clone()))
        });
        self
    }

    #[lazy_only(lazy = "group_by_startidx_agg", type = "numeric")]
    fn group_by_startidx_max(&mut self, group_idx: Self) {}

    #[lazy_only(lazy = "group_by_startidx_agg", type = "numeric")]
    fn group_by_startidx_min(&mut self, group_idx: Self) {}

    #[lazy_only(lazy = "group_by_startidx_agg", type = "numeric")]
    fn group_by_startidx_umax(&mut self, group_idx: Self) {}

    #[lazy_only(lazy = "group_by_startidx_agg", type = "numeric")]
    fn group_by_startidx_umin(&mut self, group_idx: Self) {}

    #[lazy_only(lazy = "group_by_startidx_agg", type = "numeric")]
    fn group_by_startidx_median(&mut self, group_idx: Self) {}

    #[lazy_only(lazy = "group_by_startidx_agg", type = "numeric")]
    fn group_by_startidx_quantile(&mut self, group_idx: Self, q: f64, method: QuantileMethod) {}

    #[lazy_only(lazy = "group_by_startidx_agg", type = "nostr")]
    fn group_by_startidx_first(&mut self, group_idx: Self) {}

    #[lazy_only(lazy = "group_by_startidx_agg", type = "nostr")]
    fn group_by_startidx_valid_first(&mut self, group_idx: Self) {}

    #[lazy_only(lazy = "group_by_startidx_agg", type = "nostr")]
    fn group_by_startidx_last(&mut self, group_idx: Self) {}

    #[lazy_only(lazy = "group_by_startidx_agg", type = "nostr")]
    fn group_by_startidx_valid_last(&mut self, group_idx: Self) {}

    #[lazy_only(lazy = "group_by_startidx_agg", type = "numeric")]
    fn group_by_startidx_sum(&mut self, group_idx: Self, stable: bool) {}

    #[lazy_only(lazy = "group_by_startidx_agg", type = "numeric")]
    fn group_by_startidx_mean(&mut self, group_idx: Self, min_periods: usize, stable: bool) {}

    #[lazy_only(lazy = "group_by_startidx_agg", type = "numeric")]
    fn group_by_startidx_var(&mut self, group_idx: Self, min_periods: usize, stable: bool) {}

    #[lazy_only(lazy = "group_by_startidx_agg", type = "numeric")]
    fn group_by_startidx_std(&mut self, group_idx: Self, min_periods: usize, stable: bool) {}

    #[lazy_only(lazy = "group_by_startidx_agg2", type = "numeric", type2 = "numeric")]
    fn group_by_startidx_cov(
        &mut self,
        other: Self,
        group_idx: Self,
        min_periods: usize,
        stable: bool,
    ) {
    }

    #[lazy_only(lazy = "group_by_startidx_agg2", type = "numeric", type2 = "numeric")]
    fn group_by_startidx_corr(
        &mut self,
        other: Self,
        group_idx: Self,
        method: CorrMethod,
        min_periods: usize,
        stable: bool,
    ) {
    }
}
