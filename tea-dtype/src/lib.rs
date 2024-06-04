mod object;
#[cfg(feature = "time")]
mod time;

pub use num::{One, Zero};
pub use object::Object;
pub use tevec::prelude::{BoolType, Cast, DataType, GetDataType, IsNone, Number as TvNumber};

#[cfg(feature = "time")]
pub use tevec::prelude::{DateTime, TimeDelta, TimeUnit};
#[cfg(feature = "time")]
pub use time::{DateTimeToPy, DateTimeToRs};

/// just for old code compatibility
pub trait Number: TvNumber {
    #[inline(always)]
    fn nan() -> Self {
        Self::none()
    }

    #[inline(always)]
    fn notnan(self) -> bool {
        self.not_none()
    }

    #[inline(always)]
    fn isnan(self) -> bool {
        self.is_none()
    }
}

impl<T: TvNumber> Number for T {}
