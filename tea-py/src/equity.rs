use crate::pylazy::{parse_expr_nocopy, PyExpr};
use ndarray::Zip;
use pyo3::{pyfunction, FromPyObject, PyAny, PyResult};
use tea_core::prelude::*;

pub enum CommisionType {
    Percent,
    Absolute,
}

impl<'source> FromPyObject<'source> for CommisionType {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("percent").to_lowercase();
        let out = match s.as_str() {
            "percent" => CommisionType::Percent,
            "absolute" => CommisionType::Absolute,
            _ => panic!("不支持的手续费类型: {s}, commision_type必须是'percent'或'absolute'"),
        };
        Ok(out)
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::redundant_clone)]
#[pyo3(signature=(pos, opening_cost, closing_cost, init_cash=1_000_000, multiplier=1, leverage=1., slippage=0., ticksize=0., c_rate=3e-4, blowup=false, commision_type=CommisionType::Percent, contract_change_signal=None))]
pub unsafe fn calc_ret_single(
    pos: &PyAny,
    opening_cost: &PyAny,
    closing_cost: &PyAny,
    init_cash: i64,
    multiplier: i32,
    leverage: f64,
    slippage: f64,
    ticksize: f64,
    c_rate: f64,
    blowup: bool,
    commision_type: CommisionType,
    contract_change_signal: Option<&PyAny>,
) -> PyResult<PyExpr> {
    let pos = parse_expr_nocopy(pos)?;
    let opening_cost = parse_expr_nocopy(opening_cost)?;
    let closing_cost = parse_expr_nocopy(closing_cost)?;
    let (obj1, obj2) = (opening_cost.obj(), closing_cost.obj());
    // let (opening_cost, closing_cost) = (opening_cost.cast_f64()?, closing_cost.cast_f64()?);
    let (contract_signal, obj3) = if let Some(contract_signal_obj) = contract_change_signal {
        let contract_signal = parse_expr_nocopy(contract_signal_obj)?;
        let obj = contract_signal.obj();
        (Some(contract_signal), obj)
    } else {
        (None, None)
    };
    let mut out = pos.clone();
    out.e.cast_f64().chain_f_ctx(move |(data, ctx)| {
        let arr = data.view_arr(ctx.as_ref())?;
        let pos_arr = match_arrok!(arr, a, { a.view().to_dim1()? }, F64); // 当期仓位的1d array
        let opening_cost = opening_cost.e.view_arr(ctx.as_ref())?.deref().cast_f64(); // 开仓成本的1d array
        let opening_cost_arr = opening_cost.view().to_dim1()?; // 开仓成本的1d array
        let closing_cost = closing_cost.e.view_arr(ctx.as_ref())?.deref().cast_f64(); // 平仓价格的1d array
        let closing_cost_arr = closing_cost.view().to_dim1()?; // 平仓价格的1d array
        if pos_arr.is_empty() {
            return Ok((Arr1::from_vec(Vec::<f64>::new()).to_dimd().into(), ctx));
        }
        // 账户变动信息
        let mut cash = init_cash.f64();
        let mut last_pos = 0_f64; // pos_arr[0];
        let mut last_lot_num = 0.;
        let mut last_close = closing_cost_arr[0];
        if let Some(contract_signal) = contract_signal.as_ref() {
            let contract_signal = contract_signal
                .e
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_bool();
            let contract_signal_arr = contract_signal.view().to_dim1()?;
            Ok((
                Zip::from(&pos_arr.0)
                    .and(&opening_cost_arr.0)
                    .and(&closing_cost_arr.0)
                    .and(&contract_signal_arr.0)
                    .map_collect(|&pos, &opening_cost, &closing_cost, &contract_signal| {
                        if blowup && cash <= 0. {
                            return 0.;
                        }
                        if (last_lot_num != 0.) && (!contract_signal) {
                            // 换月的时候不计算跳开的损益
                            cash += last_lot_num
                                * (opening_cost - last_close)
                                * multiplier.f64()
                                * last_pos.signum();
                        }
                        // 因为采用pos来判断加减仓，所以杠杆leverage必须是个常量，不应修改leverage的类型
                        if (pos != last_pos) || contract_signal {
                            // 仓位出现变化，计算新的理论持仓手数
                            let l = ((cash * leverage * pos.abs())
                                / (multiplier.f64() * opening_cost))
                                .floor();
                            let (lot_num, lot_num_change) = if !contract_signal {
                                (
                                    l,
                                    (l * pos.signum() - last_lot_num * last_pos.signum()).abs(),
                                )
                            } else {
                                (l, l.abs() * 2.)
                            };
                            // 扣除手续费变动
                            if let CommisionType::Percent = commision_type {
                                cash -= lot_num_change
                                    * multiplier.f64()
                                    * (opening_cost * c_rate + slippage * ticksize);
                            } else {
                                cash -= lot_num_change
                                    * (c_rate + multiplier.f64() * slippage * ticksize);
                            };
                            // 更新上期持仓手数和持仓头寸
                            last_lot_num = lot_num;
                            last_pos = pos;
                        }
                        // 计算当期损益
                        if last_lot_num != 0. {
                            cash += last_lot_num
                                * last_pos.signum()
                                * (closing_cost - opening_cost)
                                * multiplier.f64();
                        }
                        last_close = closing_cost; // 更新上期收盘价

                        cash
                        // closing_cost - opening_cost
                    })
                    .wrap()
                    .to_dimd()
                    .into(),
                ctx,
            ))
        } else {
            // 不考虑合约换月信号的情况
            Ok((
                Zip::from(&pos_arr.0)
                    .and(&opening_cost_arr.0)
                    .and(&closing_cost_arr.0)
                    .map_collect(|&pos, &opening_cost, &closing_cost| {
                        if blowup && cash <= 0. {
                            return 0.;
                        }
                        if last_lot_num != 0. {
                            cash += last_lot_num
                                * (opening_cost - last_close)
                                * multiplier.f64()
                                * last_pos.signum();
                        }
                        // 因为采用pos来判断加减仓，所以杠杆leverage必须是个常量，不应修改leverage的类型
                        if pos != last_pos {
                            // 仓位出现变化
                            // 计算新的理论持仓手数
                            let lot_num = ((cash * leverage * pos.abs())
                                / (multiplier.f64() * opening_cost))
                                .floor();
                            // 扣除手续费变动
                            if let CommisionType::Percent = commision_type {
                                cash -= (lot_num * pos.signum() - last_lot_num * last_pos.signum())
                                    .abs()
                                    * multiplier.f64()
                                    * (opening_cost * c_rate + slippage * ticksize);
                            } else {
                                cash -= (lot_num * pos.signum() - last_lot_num * last_pos.signum())
                                    .abs()
                                    * (c_rate + multiplier.f64() * slippage * ticksize);
                            };
                            // 更新上期持仓手数和持仓头寸
                            last_lot_num = lot_num;
                            last_pos = pos;
                        }
                        // 计算当期损益
                        if last_lot_num != 0. {
                            cash += last_lot_num
                                * (closing_cost - opening_cost)
                                * multiplier.f64()
                                * last_pos.signum();
                        }
                        last_close = closing_cost; // 更新上期收盘价

                        cash
                        // cash
                    })
                    .wrap()
                    .to_dimd()
                    .into(),
                ctx,
            ))
        }
    });
    Ok(out.add_obj_into(obj1).add_obj_into(obj2).add_obj_into(obj3))
}

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::redundant_clone)]
#[pyo3(signature=(pos, opening_cost, closing_cost, spread, init_cash=1_000_000, multiplier=1, leverage=1., c_rate=3e-4, blowup=false, commision_type=CommisionType::Percent, contract_change_signal=None))]
pub unsafe fn calc_ret_single_with_spread(
    pos: &PyAny,
    opening_cost: &PyAny,
    closing_cost: &PyAny,
    spread: &PyAny,
    init_cash: i64,
    multiplier: i32,
    leverage: f64,
    c_rate: f64,
    blowup: bool,
    commision_type: CommisionType,
    contract_change_signal: Option<&PyAny>,
) -> PyResult<PyExpr> {
    let pos = parse_expr_nocopy(pos)?;
    let opening_cost = parse_expr_nocopy(opening_cost)?;
    let closing_cost = parse_expr_nocopy(closing_cost)?;
    let spread = parse_expr_nocopy(spread)?;
    let (obj1, obj2, obj3) = (opening_cost.obj(), closing_cost.obj(), spread.obj());
    // let (opening_cost, closing_cost) = (opening_cost.cast_f64()?, closing_cost.cast_f64()?);
    // let spread = spread.cast_f64()?;
    let mut out = pos.clone();
    let (contract_signal, obj4) = if let Some(contract_signal_obj) = contract_change_signal {
        let contract_signal = parse_expr_nocopy(contract_signal_obj)?;
        let obj = contract_signal.obj();
        (Some(contract_signal), obj)
    } else {
        (None, None)
    };
    out.e.cast_f64().chain_f_ctx(move |(data, ctx)| {
        let arr = data.view_arr(ctx.as_ref())?;
        let pos_arr = match_arrok!(arr, a, { a.view().to_dim1()? }, F64); // 当期仓位的1d array
        let opening_cost = opening_cost.e.view_arr(ctx.as_ref())?.deref().cast_f64(); // 开仓成本的1d array
        let opening_cost_arr = opening_cost.view().to_dim1()?; // 开仓成本的1d array
        let closing_cost = closing_cost.e.view_arr(ctx.as_ref())?.deref().cast_f64(); // 平仓价格的1d array
        let closing_cost_arr = closing_cost.view().to_dim1()?; // 平仓价格的1d array
        let spread = spread.e.view_arr(ctx.as_ref())?.deref().cast_f64();
        let spread_arr = spread.view().to_dim1()?;
        if pos_arr.is_empty() {
            return Ok((Arr1::from_vec(Vec::<f64>::new()).to_dimd().into(), ctx));
        }
        // 账户变动信息
        let mut cash = init_cash.f64();
        let mut last_pos = 0_f64; // pos_arr[0];
        let mut last_lot_num = 0.;
        let mut last_close = closing_cost_arr[0];
        if let Some(contract_signal) = contract_signal.as_ref() {
            let contract_signal = contract_signal
                .e
                .view_arr(ctx.as_ref())?
                .deref()
                .cast_bool();
            let contract_signal_arr = contract_signal.view().to_dim1()?;
            Ok((
                Zip::from(&pos_arr.0)
                    .and(&opening_cost_arr.0)
                    .and(&closing_cost_arr.0)
                    .and(&spread_arr.0)
                    .and(&contract_signal_arr.0)
                    .map_collect(
                        |&pos, &opening_cost, &closing_cost, &spread, &contract_signal| {
                            if blowup && cash <= 0. {
                                return 0.;
                            }
                            if (last_lot_num != 0.) && (!contract_signal) {
                                // 换月的时候不计算跳开的损益
                                cash += last_lot_num
                                    * (opening_cost - last_close)
                                    * multiplier.f64()
                                    * last_pos.signum();
                            }
                            // 因为采用pos来判断加减仓，所以杠杆leverage必须是个常量，不应修改leverage的类型
                            if (pos != last_pos) || contract_signal {
                                // 仓位出现变化，计算新的理论持仓手数
                                let l = ((cash * leverage * pos.abs())
                                    / (multiplier.f64() * opening_cost))
                                    .floor();
                                let (lot_num, lot_num_change) = if !contract_signal {
                                    (
                                        l,
                                        (l * pos.signum() - last_lot_num * last_pos.signum()).abs(),
                                    )
                                } else {
                                    (l, l.abs() * 2.)
                                };
                                // 扣除手续费变动
                                if let CommisionType::Percent = commision_type {
                                    cash -= lot_num_change
                                        * multiplier.f64()
                                        * (opening_cost * c_rate + spread);
                                } else {
                                    cash -= lot_num_change * (c_rate + multiplier.f64() * spread);
                                };
                                // 更新上期持仓手数和持仓头寸
                                last_lot_num = lot_num;
                                last_pos = pos;
                            }
                            // 计算当期损益
                            if last_lot_num != 0. {
                                cash += last_lot_num
                                    * last_pos.signum()
                                    * (closing_cost - opening_cost)
                                    * multiplier.f64();
                            }
                            last_close = closing_cost; // 更新上期收盘价

                            cash
                            // closing_cost - opening_cost
                        },
                    )
                    .wrap()
                    .to_dimd()
                    .into(),
                ctx,
            ))
        } else {
            // 不考虑合约换月信号的情况
            Ok((
                Zip::from(&pos_arr.0)
                    .and(&opening_cost_arr.0)
                    .and(&closing_cost_arr.0)
                    .and(&spread_arr.0)
                    .map_collect(|&pos, &opening_cost, &closing_cost, &spread| {
                        if blowup && cash <= 0. {
                            return 0.;
                        }
                        if last_lot_num != 0. {
                            cash += last_lot_num
                                * (opening_cost - last_close)
                                * multiplier.f64()
                                * last_pos.signum();
                        }
                        // 因为采用pos来判断加减仓，所以杠杆leverage必须是个常量，不应修改leverage的类型
                        if pos != last_pos {
                            // 仓位出现变化
                            // 计算新的理论持仓手数
                            let lot_num = ((cash * leverage * pos.abs())
                                / (multiplier.f64() * opening_cost))
                                .floor();
                            // 扣除手续费变动
                            if let CommisionType::Percent = commision_type {
                                cash -= (lot_num * pos.signum() - last_lot_num * last_pos.signum())
                                    .abs()
                                    * multiplier.f64()
                                    * (opening_cost * c_rate + spread);
                            } else {
                                cash -= (lot_num * pos.signum() - last_lot_num * last_pos.signum())
                                    .abs()
                                    * (c_rate + multiplier.f64() * spread);
                            };
                            // 更新上期持仓手数和持仓头寸
                            last_lot_num = lot_num;
                            last_pos = pos;
                        }
                        // 计算当期损益
                        if last_lot_num != 0. {
                            cash += last_lot_num
                                * (closing_cost - opening_cost)
                                * multiplier.f64()
                                * last_pos.signum();
                        }
                        last_close = closing_cost; // 更新上期收盘价

                        cash
                        // cash
                    })
                    .wrap()
                    .to_dimd()
                    .into(),
                ctx,
            ))
        }
    });
    Ok(out
        .add_obj_into(obj1)
        .add_obj_into(obj2)
        .add_obj_into(obj3)
        .add_obj_into(obj4))
}

// #[pyfunction]
// #[pyo3(signature=(factor, price, select_num, c_rate=0.0006, hold_time=1, init_cash=10000.))]
// pub fn calc_digital_ret(
//     factor: PyExpr,
//     price: PyExpr,
//     select_num: (usize, usize),
//     c_rate: f64,
//     hold_time: usize,
//     init_cash: f64,
// ) -> PyResult<PyExpr> {
//     let factor = factor.cast_f64()?;
//     let out_expr: Expr<f64> = price.cast_f64()?.chain_view_f(
//         move |price_arr| {
//             let price = price_arr.to_dim2()?;
//             let factor_expr = factor.eval(None)?.0;
//             let factor = factor_expr.view_arr().to_dim2()?;
//             let time_n = factor.shape()[1];
//             let mut cash_vec = Vec::with_capacity(time_n);
//             // symbol_index => (hold_num, hold_cost)
//             let mut hold_amt =
//                 HashMap::<i32, (f64, f64)>::with_capacity(select_num.0 + select_num.1);
//             let mut select_symbol = VecDeque::<(Arr1<i32>, Arr1<i32>)>::with_capacity(hold_time);
//             let mut cash = init_cash;
//             // let mut unrealized_cash = init_cash;
//             for i in 0..time_n {
//                 let price_c = price.index_axis(Axis(1), i).wrap(); // current price
//                 let factor_c = factor.index_axis(Axis(1), i).wrap();
//                 // the selected symbol index in long position
//                 let mut select_long = Array1::zeros(select_num.0).wrap();
//                 factor_c.arg_partition_1d(select_long.view_mut(), select_num.0 - 1, false, false);
//                 // the selected symbol index in short position
//                 let mut select_short = Array1::zeros(select_num.0).wrap();
//                 factor_c.arg_partition_1d(select_short.view_mut(), select_num.1 - 1, false, true);
//                 let (select_long_num, select_short_num) =
//                     (select_long.shape()[0], select_short.shape()[0]);
//                 // calculate profit
//                 let mut profit = 0.;
//                 for (hold_symbol, (hold_num, cost)) in &hold_amt {
//                     let current_symbol_price = unsafe { *price_c.uget(*hold_symbol as usize) };
//                     profit += *hold_num * (current_symbol_price - *cost)
//                 }
//                 // adjust select_symbol
//                 if select_symbol.len() == hold_time {
//                     select_symbol.pop_front();
//                 }
//                 select_symbol.push_back((select_long, select_short));
//                 let realized_cash = if i >= hold_time {
//                     unsafe { *cash_vec.get_unchecked(i - hold_time) }
//                 } else {
//                     init_cash
//                 };
//                 // calculate target symbol amount
//                 let order_cash_long =
//                     realized_cash / 2. / select_long_num as f64 / select_num.0 as f64;
//                 let order_cash_short =
//                     realized_cash / 2. / select_short_num as f64 / select_num.1 as f64;
//                 let mut target_amt =
//                     HashMap::<i32, (f64, f64)>::with_capacity(select_num.0 + select_num.1);
//                 for (long_symbol_list, short_symbol_list) in &select_symbol {
//                     for symbol in long_symbol_list {
//                         let symbol_price = unsafe { *price_c.uget(*symbol as usize) };
//                         let hold_num_one_part = order_cash_long / symbol_price;
//                         if let Some((hold_num, _cost)) = target_amt.get_mut(symbol) {
//                             *hold_num += hold_num_one_part;
//                         } else {
//                             target_amt.insert(*symbol, (hold_num_one_part, symbol_price));
//                         }
//                     }
//                     for symbol in short_symbol_list {
//                         let symbol_price = unsafe { *price_c.uget(*symbol as usize) };
//                         let hold_num_one_part = -order_cash_short / symbol_price;
//                         if let Some((hold_num, _cost)) = target_amt.get_mut(symbol) {
//                             *hold_num += hold_num_one_part;
//                         } else {
//                             target_amt.insert(*symbol, (hold_num_one_part, symbol_price));
//                         }
//                     }
//                 }
//                 // calculate commission charge
//                 for (target_symbol, (target_num, cost)) in &target_amt {
//                     if let Some((hold_num, _hold_cost)) = hold_amt.remove(target_symbol) {
//                         profit -= (*target_num - hold_num).abs() * *cost * c_rate
//                     } else {
//                         // open position
//                         profit -= target_num.abs() * *cost * c_rate
//                     }
//                 }
//                 // close position
//                 for (hold_num, cost) in hold_amt.values() {
//                     profit -= hold_num.abs() * *cost * c_rate
//                 }
//                 cash += profit;
//                 cash_vec.push(cash);
//                 hold_amt = target_amt;
//             }
//             let out = Arr1::from_vec(cash_vec).to_dimd();
//             Ok(out.into())
//         },
//         RefType::False,
//     );
//     Ok(out_expr.into())
// }
