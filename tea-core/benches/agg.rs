#![feature(test)]

extern crate test;
use test::Bencher;

use tea_core::prelude::*;
use tevec::prelude::*;

const LENGTH: i32 = 10_000_000;

#[bench]
fn bench_sum(b: &mut Bencher) {
    let arr = Arr1::from_iter(0..LENGTH);
    b.iter(|| arr.sum_1d(false));
}

#[bench]
fn bench_sum2(b: &mut Bencher) {
    let arr = Arr1::from_iter(0..LENGTH).0;
    // let arr = Vec::from_iter(0..LENGTH);
    let slc = arr.try_as_slice().unwrap();
    b.iter(|| slc.to_iter().vsum().unwrap());
}

#[bench]
fn bench_sum3(b: &mut Bencher) {
    let arr = Arr1::from_iter(0..LENGTH).0;
    b.iter(|| arr.to_iter().vsum().unwrap());
}
