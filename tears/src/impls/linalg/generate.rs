use super::super::super::{ArrBase, ArrayBase, Data, DataMut, DataOwned, Ix2, WrapNdarray};
use super::replicate;

/// Hermite conjugate matrix
pub fn conjugate<Si, So>(a: &ArrBase<Si, Ix2>) -> ArrBase<So, Ix2>
where
    Si: Data<Elem = f64>,
    So: DataOwned<Elem = f64> + DataMut,
{
    let a: ArrayBase<So, Ix2> = replicate(&a.t().wrap()).0;
    // for val in a.iter_mut() {
    //     *val = val.conj();
    // }
    a.wrap()
}
