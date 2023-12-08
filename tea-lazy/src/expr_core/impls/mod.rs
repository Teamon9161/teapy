mod impl_cast;
mod impl_dtype_judge;
mod utils;
pub use utils::adjust_slice;
#[cfg(feature = "ops")]
mod impl_ops;

use super::super::Expr;
use tea_core::prelude::*;
use tea_core::utils::CollectTrustedToVec;

impl<'a> Expr<'a> {
    #[allow(unreachable_patterns)]
    pub fn split_vec_base(self, len: usize) -> Vec<Expr<'a>> {
        // todo: improve performance
        // currently we need clone the result,
        // how to take the ownership of the result and split into
        // several Exprs without clone?
        let mut out = (0..len).map(|_| self.clone()).collect_trusted();
        out.iter_mut().enumerate().for_each(|(i, e)| {
            e.chain_f_ctx(move |(data, ctx)| {
                let arr = data.view_arr_vec(ctx.as_ref())?.remove(i);
                Ok((match_arrok!(arr, a, { a.view().to_owned().into() }), ctx))
            });
        });
        out
    }
}
