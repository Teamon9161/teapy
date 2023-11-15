use crate::TimeDelta;
use chrono::Duration;

impl Default for TimeDelta {
    fn default() -> Self {
        TimeDelta::nat()
    }
}

impl From<Duration> for TimeDelta {
    fn from(duration: Duration) -> Self {
        Self {
            months: 0,
            inner: duration,
        }
    }
}

impl From<i64> for TimeDelta {
    fn from(dt: i64) -> Self {
        if dt == i64::MIN {
            return TimeDelta::nat();
        }
        Duration::microseconds(dt).into()
    }
}

// impl Cast<i64> for TimeDelta {
//     fn cast(self) -> i64 {
//         let months = self.months;
//         if months != 0 {
//             panic!("not support cast TimeDelta to i64 when months is not zero")
//         } else {
//             self.inner.num_microseconds().unwrap_or(i64::MIN)
//         }
//     }
// }
