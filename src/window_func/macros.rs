macro_rules! IsNan {
    ($v:expr) => {
        ($v) != ($v)
    };
}

macro_rules! NotNan {
    ($v:expr) => {
        ($v) == ($v)
    };
}
