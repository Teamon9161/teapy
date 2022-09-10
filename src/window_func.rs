use crate::arrview::{ArrView1, ArrViewMut1};
use crate::datatype::Number;
use crate::func_type::auto_define;

auto_define!(
    direct
    ts_sma, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_ewm, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_wma, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_sum, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_std, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_var, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_skew, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_kurt, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_prod, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_prod_mean, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_reg, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_tsf, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_reg_slope, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_reg_intercept, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_meanstdnorm, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_stable, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_minmaxnorm, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_min, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_max, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_argmin, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_argmax, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_rank, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define!(
    direct
    ts_rank_pct, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define2!(
    direct
    ts_cov, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);

auto_define2!(
    direct
    ts_corr, f64,
    -a (
        window: usize,
        min_periods: usize,
        stable: bool,
    ),
);
