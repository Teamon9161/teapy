mod feature;
mod cmp;
mod corr;
mod norm;
mod reg;
#[cfg(feature = "lazy")]
mod impl_lazy;

pub use cmp::CmpTs;
pub use feature::FeatureTs;
pub use corr::CorrTs;
pub use norm::NormTs;
pub use reg::{RegTs, Reg2Ts};
#[cfg(feature = "lazy")]
pub use impl_lazy::ExprRollingExt;