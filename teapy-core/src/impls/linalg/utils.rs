use crate::{ArrBase, WrapNdarray};
use ndarray::{ArrayBase, Data, DataMut, DataOwned, Dimension};

pub fn replicate<A, Sv, So, D>(a: &ArrBase<Sv, D>) -> ArrBase<So, D>
where
    A: Copy,
    Sv: Data<Elem = A>,
    So: DataOwned<Elem = A> + DataMut,
    D: Dimension,
{
    unsafe {
        let ret = ArrayBase::<So, D>::build_uninit(a.dim(), |view| {
            a.assign_to(view);
        });
        ret.assume_init().wrap()
    }
}
