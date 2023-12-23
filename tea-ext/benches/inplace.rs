// #![feature(test)]

// extern crate test;

// use test::Bencher;

// use tea_core::prelude::*;
// use tea_ext::InplaceExt;

// // #[bench]
// fn bench_shift_left(b: &mut Bencher) {
//     let mut v = Arr1::from_vec(vec![1.0; 100000]);
//     b.iter(|| {
//         v.shift_1d(2, None);
//     })
// }

// // #[bench]
// fn bench_shift_right(b: &mut Bencher) {
//     let mut v = Arr1::from_vec(vec![1.0; 100000]);
//     b.iter(|| {
//         v.shift_1d(-2, None);
//     })
// }
