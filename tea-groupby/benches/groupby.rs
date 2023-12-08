#![feature(test)]

extern crate test;

use test::Bencher;

use tea_core::prelude::*;
use tea_groupby::{groupby, groupby1, groupby2};

fn produce_data() -> (ArrOk<'static>, ArrOk<'static>) {
    let length = 1000;
    let sy1 = vec![
        String::from("a"),
        String::from("b"),
        String::from("c"),
        String::from("d"),
    ];
    let len1 = sy1.len();
    let sy2 = vec![1, 2, 3, 4, 5];
    let len2 = sy2.len();
    let data1: ArrD<_> = Arr1::from_vec(
        std::iter::repeat(sy1)
            .take(length / len1)
            .flatten()
            .collect::<Vec<_>>(),
    )
    .to_dimd();
    let arr1: ArrOk = data1.into();
    let data2: ArrD<_> = Arr1::from_vec(
        std::iter::repeat(sy2)
            .take(length / len2)
            .flatten()
            .collect::<Vec<_>>(),
    )
    .to_dimd();
    let arr2: ArrOk = data2.into();
    (arr1, arr2)
}

#[bench]
fn bench_groupby(b: &mut Bencher) {
    let (arr1, arr2) = produce_data();
    b.iter(|| {
        groupby(&[&arr1, &arr2], true);
    })
}
