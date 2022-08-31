#[cxx::bridge]
pub mod ffi {
    extern "C++" {
        include!("teapy/include/ts_func.h");
        pub unsafe fn ts_sma_1d(arr: *const f64, out: *mut f64, len: i32, window: i32, min_periods: i32, o_step: i32);
    }
}
pub use self::ffi::*;