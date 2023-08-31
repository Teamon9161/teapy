use crate::lazy::{Expr, ExprElement};

impl<'a, T> Expr<'a, T>
where
    T: ExprElement + Clone + 'a,
{
    pub fn put_mask(self, mask: Expr<'a, bool>, value: Expr<'a, T>, axis: i32, par: bool) -> Self {
        self.chain_view_mut_f_ct(move |(arr, ct)| {
            let (out1, ct) = mask.eval(ct)?;
            let (out2, ct) = value.eval(ct)?;
            arr.put_mask(&out1.view_arr(), &out2.view_arr(), axis, par)?;
            Ok(ct)
        })
    }
}
