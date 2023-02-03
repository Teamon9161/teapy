use std::collections::VecDeque;

use crate::arr::{Arr1, Expr, WrapNdarray};
use crate::pylazy::PyExpr;
use ahash::{HashMap, HashMapExt};
use ndarray::{Array1, Axis};
use pyo3::{pyfunction, PyResult};

#[pyfunction]
#[pyo3(signature=(factor, price, select_num, c_rate=0.0006, hold_time=1, init_cash=10000.))]
pub fn calc_digital_ret(
    factor: PyExpr,
    price: PyExpr,
    select_num: (usize, usize),
    c_rate: f64,
    hold_time: usize,
    init_cash: f64,
) -> PyResult<PyExpr> {
    let factor = factor.cast_f64()?;
    let out_expr: Expr<f64> = price.cast_f64()?.chain_view_f(move |price_arr| {
        let price = price_arr.to_dim2().unwrap();
        let factor_expr = factor.eval();
        let factor = factor_expr.view_arr().to_dim2().unwrap();
        let time_n = factor.shape()[1];
        let mut cash_vec = Vec::with_capacity(time_n);
        // symbol_index => (hold_num, hold_cost)
        let mut hold_amt = HashMap::<i32, (f64, f64)>::with_capacity(select_num.0 + select_num.1);
        let mut select_symbol = VecDeque::<(Arr1<i32>, Arr1<i32>)>::with_capacity(hold_time);
        let mut cash = init_cash;
        // let mut unrealized_cash = init_cash;
        for i in 0..time_n {
            let price_c = price.index_axis(Axis(1), i).wrap(); // current price
            let factor_c = factor.index_axis(Axis(1), i).wrap();
            // the selected symbol index in long position
            let mut select_long = Array1::zeros(select_num.0).wrap();
            factor_c.arg_partition_1d(select_long.view_mut(), select_num.0 - 1, false, false);
            // the selected symbol index in short position
            let mut select_short = Array1::zeros(select_num.0).wrap();
            factor_c.arg_partition_1d(select_short.view_mut(), select_num.1 - 1, false, true);
            let (select_long_num, select_short_num) =
                (select_long.shape()[0], select_short.shape()[0]);
            // calculate profit
            let mut profit = 0.;
            for (hold_symbol, (hold_num, cost)) in &hold_amt {
                let current_symbol_price = unsafe { *price_c.uget(*hold_symbol as usize) };
                profit += *hold_num * (current_symbol_price - *cost)
            }
            // adjust select_symbol
            if select_symbol.len() == hold_time {
                select_symbol.pop_front();
            }
            select_symbol.push_back((select_long, select_short));
            let realized_cash = if i >= hold_time {
                unsafe { *cash_vec.get_unchecked(i - hold_time) }
            } else {
                init_cash
            };
            // calculate target symbol amount
            let order_cash_long = realized_cash / 2. / select_long_num as f64 / select_num.0 as f64;
            let order_cash_short =
                realized_cash / 2. / select_short_num as f64 / select_num.1 as f64;
            let mut target_amt =
                HashMap::<i32, (f64, f64)>::with_capacity(select_num.0 + select_num.1);
            for (long_symbol_list, short_symbol_list) in &select_symbol {
                for symbol in long_symbol_list {
                    let symbol_price = unsafe { *price_c.uget(*symbol as usize) };
                    let hold_num_one_part = order_cash_long / symbol_price;
                    if let Some((hold_num, _cost)) = target_amt.get_mut(symbol) {
                        *hold_num += hold_num_one_part;
                    } else {
                        target_amt.insert(*symbol, (hold_num_one_part, symbol_price));
                    }
                }
                for symbol in short_symbol_list {
                    let symbol_price = unsafe { *price_c.uget(*symbol as usize) };
                    let hold_num_one_part = -order_cash_short / symbol_price;
                    if let Some((hold_num, _cost)) = target_amt.get_mut(symbol) {
                        *hold_num += hold_num_one_part;
                    } else {
                        target_amt.insert(*symbol, (hold_num_one_part, symbol_price));
                    }
                }
            }
            // calculate commission charge
            for (target_symbol, (target_num, cost)) in &target_amt {
                if let Some((hold_num, _hold_cost)) = hold_amt.remove(target_symbol) {
                    profit -= (*target_num - hold_num).abs() * *cost * c_rate
                } else {
                    // open position
                    profit -= target_num.abs() * *cost * c_rate
                }
            }
            // close position
            for (hold_num, cost) in hold_amt.values() {
                profit -= hold_num.abs() * *cost * c_rate
            }
            cash += profit;
            cash_vec.push(cash);
            hold_amt = target_amt;
        }
        let out = Arr1::from_vec(cash_vec).to_dimd().unwrap();
        out.into()
    });
    Ok(out_expr.into())
}
