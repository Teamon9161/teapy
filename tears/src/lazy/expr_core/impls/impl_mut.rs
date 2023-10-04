use ndarray::SliceInfoElem;

use super::export::*;
use super::utils::adjust_slice;
use crate::ArbArray;

impl<'a> Expr<'a> {
    #[allow(unreachable_patterns)]
    pub fn put_mask(&mut self, mask: Expr<'a>, value: Expr<'a>, axis: i32, par: bool) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let mut mask = mask.clone();
            let mut value = value.clone();
            mask.cast_bool().eval_inplace(ctx.clone())?;
            value.eval_inplace(ctx.clone())?;
            let mask_arr = mask.view_arr(ctx.as_ref())?;
            let mask_arr = match_arrok!(mask_arr, a, { a }, Bool);
            let value_arr = value.into_arr(ctx.clone())?;
            let mut arr = data.into_arr(ctx.clone())?;
            match_arrok!(&mut arr, a, {
                let value: ArbArray<_> = value_arr.cast();
                a.view_mut()
                    .put_mask(&mask_arr.view(), &value.view(), axis, par)?
            });
            Ok((arr.into(), ctx))
        });
        self
    }

    pub fn set_item_by_slice(&mut self, slc: Vec<SliceInfoElem>, value: Self) -> &mut Self {
        self.chain_f_ctx(move |(data, ctx)| {
            let value = value.view_arr(ctx.as_ref())?;
            let mut arr = data.into_arr(ctx.clone())?;
            let slc_info = adjust_slice(slc.clone(), arr.shape(), arr.ndim());
            match_arrok!(castable &mut arr, arr, {
                let mut arr_view_mut = arr.view_mut();
                let mut arr_mut = arr_view_mut.slice_mut(slc_info).wrap().to_dimd();
                match_arrok!(castable value, v, {
                    let v = v.deref().cast();
                    arr_mut.assign(&v.view());
                })
            });
            Ok((arr.into(), ctx))
        });
        self
    }
}
