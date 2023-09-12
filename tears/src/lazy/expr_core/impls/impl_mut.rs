use super::export::*;
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
                a.viewmut()
                    .put_mask(&mask_arr.view(), &value.view(), axis, par)?
            });
            // use ArrOk::*;
            // match &mut arr {
            //     F64(a) => a.viewmut().put_mask(&mask_arr.view(), &value_arr.cast_f64().view(), axis, par)?,
            //     F32(a) => a.viewmut().put_mask(&mask_arr.view(), &value_arr.cast_f32().view(), axis, par)?,
            //     I64(a) => a.viewmut().put_mask(&mask_arr.view(), &value_arr.cast_i64().view(), axis, par)?,
            //     I32(a) => a.viewmut().put_mask(&mask_arr.view(), &value_arr.cast_i32().view(), axis, par)?,
            //     Usize(a) => a.viewmut().put_mask(&mask_arr.view(), &value_arr.cast_usize().view(), axis, par)?,
            //     DateTime(a) => a.viewmut().put_mask(&mask_arr.view(), &value_arr.cast_datetime_default().view(), axis, par)?,
            //     TimeDelta(a) => a.viewmut().put_mask(&mask_arr.view(), &value_arr.cast_timedelta().view(), axis, par)?,
            //     Bool(a) => a.viewmut().put_mask(&mask_arr.view(), &value_arr.cast_bool().view(), axis, par)?,
            //     String(a) => a.viewmut().put_mask(&mask_arr.view(), &value_arr.cast_string().view(), axis, par)?,
            //     Str(a) => a.viewmut().put_mask(&mask_arr.view(), &value_arr.cast_str().view(), axis, par)?,
            //     OptUsize(a) => a.viewmut().put_mask(&mask_arr.view(), &value_arr.cast_optusize().view(), axis, par)?,
            //     #[cfg(feature="option_dtype")] OptI64(a) => a.viewmut().put_mask(&mask_arr.view(), &value_arr.cast_opti64().view(), axis, par)?,
            //     #[cfg(feature="option_dtype")] OptI32(a) => a.viewmut().put_mask(&mask_arr.view(), &value_arr.cast_opti32().view(), axis, par)?,
            //     #[cfg(feature="option_dtype")] OptF64(a) => a.viewmut().put_mask(&mask_arr.view(), &value_arr.cast_optf64().view(), axis, par)?,
            //     #[cfg(feature="option_dtype")] OptF32(a) => a.viewmut().put_mask(&mask_arr.view(), &value_arr.cast_optf32().view(), axis, par)?,
            //     _ => unimplemented!("put_mask not implemented for {:?}", arr.dtype()),
            // }
            Ok((arr.into(), ctx))
        });
        self
    }
}
