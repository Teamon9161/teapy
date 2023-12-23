#![feature(test)]

extern crate test;

use test::Bencher;

use tea_core::prelude::*;
use tea_ext::{AutoExprAggExt, AutoExprRollingExt, RollingExt};
use tea_lazy::{ColumnSelector, Data, Expr};

#[bench]
fn bench_rolling_apply_with_start(b: &mut Bencher) {
    let length = 100000;
    let window = 200;
    let v: ArrOk = Arr1::from_vec((1..=length).collect::<Vec<_>>())
        .to_dimd()
        .into();
    let mut v: Expr = v.into();
    v.rename("v".to_string());
    let start = std::iter::repeat(0)
        .take(window)
        .chain(0..(length - window))
        .collect::<Vec<_>>();
    let start: ArrOk = Arr1::from_vec(start).to_dimd().into();
    let start: Expr = start.into();
    let agg_expr: Data = ColumnSelector::Index(0).into();
    let mut agg_expr: Expr = agg_expr.into();
    agg_expr.mean(1, false, 0, false);
    // let mut v1 = v.clone();
    // v1.rolling_apply_with_start(agg_expr.clone(), start.clone(), vec![], false);
    // let _ = v1.eval_inplace(None).unwrap();
    // dbg!("{:?}", v1.view_arr(None).unwrap());
    b.iter(|| {
        let mut v1 = v.clone();
        v1.rolling_apply_with_start(agg_expr.clone(), start.clone(), vec![], false);
        let _ = v1.eval_inplace(None).unwrap();
    })
}

// #[bench]
// fn bench_rolling_apply_with_start1(b: &mut Bencher) {
//     let length = 100000;
//     let window = 200;
//     let v: ArrOk = Arr1::from_vec((1..=length).collect::<Vec<_>>()).to_dimd().into();
//     let mut v: Expr = v.into();
//     v.rename("v".to_string());
//     let start = std::iter::repeat(0).take(window).chain(0..(length - window)).collect::<Vec<_>>();
//     let start: ArrOk = Arr1::from_vec(start).to_dimd().into();
//     let start: Expr = start.into();
//     let agg_expr: Data = ColumnSelector::Index(0).into();
//     let mut agg_expr: Expr = agg_expr.into();
//     agg_expr.mean(1, false, 0, false);
//     let mut v1 = v.clone();
//     v1.rolling_apply_with_start1(agg_expr.clone(), start.clone(), vec![], false);
//     let _ = v1.eval_inplace(None).unwrap();
//     dbg!("{:?}", v1.view_arr(None).unwrap());
//     b.iter(|| {
//         let mut v1 = v.clone();
//         v1.rolling_apply_with_start1(agg_expr.clone(), start.clone(), vec![], false);
//         let _ = v1.eval_inplace(None).unwrap();
//     })
// }

#[bench]
fn bench_rolling_apply_mean(b: &mut Bencher) {
    let length = 100000;
    let window = 200;
    let v: ArrOk = Arr1::from_vec((1..=length).collect::<Vec<_>>())
        .to_dimd()
        .into();
    let mut v: Expr = v.into();
    v.rename("v".to_string());
    let start = std::iter::repeat(0)
        .take(window)
        .chain(0..(length - window))
        .collect::<Vec<_>>();
    let start: ArrOk = Arr1::from_vec(start).to_dimd().into();
    let start: Expr = start.into();
    let mut v1 = v.clone();
    v1.rolling_select_mean(start.clone(), 1, false);
    let _ = v1.eval_inplace(None).unwrap();
    dbg!("{:?}", v1.view_arr(None).unwrap());
    b.iter(|| {
        let mut v1 = v.clone();
        v1.rolling_select_mean(start.clone(), 1, false);
        let _ = v1.eval_inplace(None).unwrap();
    })
}
