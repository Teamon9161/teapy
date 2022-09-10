use crate::arrview::{ArrView1, ArrViewMut1};
use crate::datatype::Number;
use crate::func_type::{auto_define, auto_define2};

auto_define!(
    agg
    min, f64,
    -a (_stable: bool),
    -noargs
);

auto_define!(
    agg
    max, f64,
    -a (_stable: bool),
    -noargs
);

auto_define!(
    agg
    sum, f64,
    -a (stable: bool),
);

auto_define!(
    agg
    mean, f64,
    -a (stable: bool),
);

auto_define!(
    agg
    var, f64,
    -a (stable: bool),
    -c (false, stable),
);

auto_define!(
    agg
    std, f64,
    -a (stable: bool),
    -c (false, stable)
);

auto_define!(
    agg
    skew, f64,
    -a (stable: bool),
    -c (stable),
);

auto_define!(
    agg
    kurt, f64,
    -a (stable: bool),
    -c (stable),
);

auto_define!(
    agg
    count_nan, usize,
    -a (_stable: bool),
    -noargs
);

auto_define!(
    agg
    count_notnan, usize,
    -a (_stable: bool),
    -noargs
);

auto_define!(
    direct
    argsort, i32,
    -a (_stable: bool),
    -noargs
);

auto_define!(
    direct
    rank, f64,
    -a (pct: bool),
);

auto_define2!(
    agg
    cov, f64,
    -a (stable: bool),
);

auto_define2!(
    agg
    corr, f64,
    -a (stable: bool),
);
