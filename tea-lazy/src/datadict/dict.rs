use rayon::prelude::*;
use regex::Regex;
use std::fmt::Debug;
use std::iter::zip;
use std::sync::Arc;
// use serde::Serialize;

use crate::{Context, Expr};
use core::{
    error::{StrError, TpResult},
    utils::CollectTrustedToVec,
};

#[cfg(feature = "agg")]
use crate::CorrMethod;

use super::get_set::{GetMutOutput, GetOutput, SetInput};
use super::selector::ColumnSelector;

use crate::TpHashMap;

// static DATADICT_INIT_SIZE: usize = 10;

#[derive(Clone, Default)]
pub struct DataDict<'a> {
    pub data: Vec<Expr<'a>>,
    pub map: Arc<TpHashMap<String, usize>>,
}

impl Debug for DataDict<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.data.iter().map(|e| (e.name().unwrap(), e)))
            .finish()
    }
}

#[allow(clippy::missing_safety_doc)]
impl<'a> DataDict<'a> {
    pub fn new(mut data: Vec<Expr<'a>>, columns: Option<Vec<String>>) -> Self {
        if let Some(columns) = columns {
            assert_eq!(data.len(), columns.len());
            let mut column_map = TpHashMap::<String, usize>::with_capacity(data.len());
            for (col_name, i) in zip(columns, 0..data.len()) {
                column_map.insert(col_name.clone(), i);
                let expr = unsafe { data.get_unchecked_mut(i) };
                expr.rename(col_name);
            }
            DataDict {
                data,
                map: Arc::new(column_map),
            }
        } else {
            let mut column_map = TpHashMap::<String, usize>::with_capacity(data.len());
            let mut name_auto = 0;
            for i in 0..data.len() {
                // we must check the name of PyExpr
                let expr = unsafe { data.get_unchecked_mut(i) };
                if let Some(name) = expr.name() {
                    column_map.insert(name, i);
                } else {
                    // the expr doen't have a name, we must ensure the name of
                    // expr are the same with the name in column_map.
                    column_map.insert(format!("column_{name_auto}"), i);
                    expr.rename(format!("column_{name_auto}"));
                    name_auto += 1;
                }
            }
            DataDict {
                data,
                map: Arc::new(column_map),
            }
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn reproduce_map(&mut self) {
        let mut map = TpHashMap::<String, usize>::with_capacity(self.len());
        for (i, e) in self.data.iter().enumerate() {
            map.insert(e.name().unwrap(), i);
        }
        self.map = Arc::new(map);
    }

    #[inline]
    pub fn columns_owned(&self) -> Vec<String> {
        self.data
            .iter()
            .map(|e| e.name().unwrap())
            .collect_trusted()
    }

    #[inline]
    pub fn columns(&self) -> Vec<&str> {
        // safety: we can't evaluate the exprs because we don't reference mut.
        self.data
            .iter()
            .map(|e| e.ref_name().unwrap())
            .collect_trusted()
    }

    #[inline(always)]
    pub fn into_data(self) -> Vec<Expr<'a>> {
        self.data
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get_new_col_name(&self) -> String {
        let mut name_auto = 0;
        loop {
            let name = format!("column_{name_auto}");
            if self.map.contains_key(&name) {
                name_auto += 1;
            } else {
                return name;
            }
        }
    }

    /// If the name of expression has changed after evaluating, we should update the column_map.
    pub fn update_column_map(&mut self, ori_name: String, new_name: String) -> TpResult<()> {
        if ori_name != new_name {
            if let Some(map) = Arc::get_mut(&mut self.map) {
                let i = map.remove(&ori_name).unwrap();
                map.insert(new_name, i);
            } else {
                let mut map = (*self.map).clone();
                let i = map.remove(&ori_name).unwrap();
                map.insert(new_name, i);
                self.map = Arc::new(map);
            };
            Ok(())
        } else {
            Ok(())
        }
    }

    /// Copy the column map
    #[inline(always)]
    pub fn map_to_own(&mut self) {
        if Arc::get_mut(&mut self.map).is_none() {
            self.map = Arc::new((*self.map).clone());
        }
    }

    #[inline]
    pub fn set_columns(&mut self, columns: Vec<String>) {
        assert_eq!(columns.len(), self.len());
        let mut map = TpHashMap::<String, usize>::with_capacity(columns.len());
        for (col_name, i) in zip(columns, 0..self.len()) {
            map.insert(col_name.clone(), i);
            let expr = unsafe { self.data.get_unchecked_mut(i) };
            expr.rename(col_name);
        }
        self.map = Arc::new(map);
    }

    pub fn get_selector_out_name(&self, col: ColumnSelector) -> Vec<String> {
        // this logic should be the same with set
        match col {
            ColumnSelector::All => self.columns_owned(),
            ColumnSelector::Index(col_idx) => {
                let col_idx = self.valid_idx(col_idx).ok();
                if let Some(col_idx) = col_idx {
                    vec![self.data[col_idx].name().unwrap()]
                } else {
                    vec![self.get_new_col_name()]
                }
            }
            ColumnSelector::Name(col_name) => {
                if col_name.starts_with('^') & col_name.ends_with('$') {
                    let re = Regex::new(col_name).unwrap();
                    return self.get_selector_out_name(ColumnSelector::Regex(re));
                } else {
                    vec![col_name.to_string()]
                }
            }
            ColumnSelector::NameOwned(name) => self.get_selector_out_name(name.as_str().into()),
            ColumnSelector::Regex(re) => self
                .columns()
                .into_iter()
                .filter_map(|col_name| {
                    if re.is_match(col_name) {
                        Some(col_name.to_string())
                    } else {
                        None
                    }
                })
                .collect(),
            ColumnSelector::VecIndex(idx_vec) => idx_vec
                .into_iter()
                .flat_map(|idx| self.get_selector_out_name(idx.into()))
                .collect::<Vec<_>>(),
            ColumnSelector::VecName(name_vec) => name_vec
                .into_iter()
                .flat_map(|name| self.get_selector_out_name(ColumnSelector::Name(name)))
                .collect::<Vec<_>>(),
            ColumnSelector::VecNameOwned(name_vec) => self.get_selector_out_name(
                name_vec.iter().map(|s| s.as_str()).collect_trusted().into(),
            ),
        }
    }

    /// drop some columns inplace, return the name of the dropped columns
    #[inline]
    pub fn drop_inplace(&mut self, col: ColumnSelector) -> TpResult<Vec<String>> {
        let drop_cols = self.get_selector_out_name(col);
        self.data = self
            .data
            .drain_filter(|e| !drop_cols.contains(&e.name().unwrap()))
            .collect::<Vec<_>>();
        self.reproduce_map();
        Ok(drop_cols)
    }

    /// Adjust when idx < 0
    #[inline]
    fn valid_idx(&self, col_idx: i32) -> TpResult<usize> {
        let mut col_idx = col_idx;
        if col_idx < 0 {
            col_idx += self.len() as i32;
            if col_idx < 0 {
                return Err("Column doesn't exist!".into());
            }
        }
        Ok(col_idx as usize)
    }

    pub fn get<'b>(&'b self, col: ColumnSelector<'b>) -> TpResult<GetOutput<'a, 'b>> {
        match col {
            ColumnSelector::Index(col_idx) => {
                let col_idx = self.valid_idx(col_idx)?;
                Ok(self
                    .data
                    .get(col_idx)
                    .unwrap_or_else(|| panic!("Select index: {:?} of ot bound", &col_idx))
                    .into())
            }
            ColumnSelector::NameOwned(col_name) => {
                // self.get(col_name.clone().as_str().into())
                if col_name.starts_with('^') & col_name.ends_with('$') {
                    let re = Regex::new(col_name.as_str())
                        .map_err(|_| StrError("Invalid regex!".into()))?;
                    return self.get(ColumnSelector::Regex(re));
                }
                let col_idx = *self.map.get(&col_name).ok_or_else(|| -> StrError {
                    format!("Column {col_name} doesn't exist!").into()
                })?;
                Ok(self
                    .data
                    .get(col_idx)
                    .expect("Select index of ot bound")
                    .into())
            }
            ColumnSelector::Name(col_name) => {
                if col_name.starts_with('^') & col_name.ends_with('$') {
                    let re = Regex::new(col_name).map_err(|_| StrError("Invalid regex!".into()))?;
                    return self.get(ColumnSelector::Regex(re));
                }
                let col_idx = *self.map.get(col_name).ok_or_else(|| -> StrError {
                    format!("Column {col_name} doesn't exist!").into()
                })?;
                Ok(self
                    .data
                    .get(col_idx)
                    .expect("Select index of ot bound")
                    .into())
            }
            ColumnSelector::All => Ok(self.data.iter().collect::<Vec<_>>().into()),
            ColumnSelector::Regex(re) => {
                let out: Vec<&Expr> = self
                    .data
                    .iter()
                    .filter(|e| re.is_match(e.name().unwrap().as_str()))
                    .collect();
                Ok(out.into())
            }
            ColumnSelector::VecIndex(idx_vec) => {
                let out = idx_vec
                    .into_iter()
                    .map(|idx| self.get(idx.into()).unwrap().into_expr().unwrap())
                    .collect_trusted();
                Ok(out.into())
            }
            ColumnSelector::VecName(name_vec) => {
                let out = name_vec
                    .into_iter()
                    .flat_map(|name| self.get(name.into()).unwrap().into_exprs())
                    .collect::<Vec<_>>();
                Ok(out.into())
            }
            ColumnSelector::VecNameOwned(name_vec) => {
                let out = name_vec
                    .into_iter()
                    .flat_map(|name| self.get(name.into()).unwrap().into_exprs())
                    .collect::<Vec<_>>();
                Ok(out.into())
            }
        }
    }

    pub fn get_mut<'b>(&'b mut self, col: ColumnSelector<'b>) -> TpResult<GetMutOutput<'a, 'b>> {
        match col {
            ColumnSelector::Index(col_idx) => {
                let col_idx = self.valid_idx(col_idx)?;
                Ok(unsafe { self.data.get_unchecked_mut(col_idx).into() })
            }
            ColumnSelector::NameOwned(col_name) => {
                // self.get_mut(col_name.as_str().into())
                if col_name.starts_with('^') & col_name.ends_with('$') {
                    let re = Regex::new(col_name.as_str())
                        .map_err(|_| StrError("Invalid regex!".into()))?;
                    return self.get_mut(ColumnSelector::Regex(re));
                }
                let col_idx = *self.map.get(&col_name).ok_or_else(|| -> StrError {
                    format!("Column {col_name} doesn't exist!").into()
                })?;
                Ok(unsafe { self.data.get_unchecked_mut(col_idx).into() })
            }
            ColumnSelector::Name(col_name) => {
                if col_name.starts_with('^') & col_name.ends_with('$') {
                    let re = Regex::new(col_name).map_err(|_| StrError("Invalid regex!".into()))?;
                    return self.get_mut(ColumnSelector::Regex(re));
                }
                let col_idx = *self.map.get(col_name).ok_or_else(|| -> StrError {
                    format!("Column {col_name} doesn't exist!").into()
                })?;
                Ok(unsafe { self.data.get_unchecked_mut(col_idx).into() })
            }
            ColumnSelector::All => Ok(self.data.iter_mut().collect::<Vec<_>>().into()),
            ColumnSelector::Regex(re) => {
                let out: Vec<&mut Expr> = self
                    .data
                    .iter_mut()
                    .filter(|e| re.is_match(e.name().unwrap().as_str()))
                    .collect();
                Ok(out.into())
            }
            ColumnSelector::VecIndex(vi) => {
                let vi = vi
                    .into_iter()
                    .map(|idx| self.valid_idx(idx).unwrap())
                    .collect::<Vec<_>>();
                let out: Vec<&mut Expr> = self
                    .data
                    .iter_mut()
                    .enumerate()
                    .filter(|(i, _e)| vi.contains(i))
                    .map(|(_i, e)| e)
                    .collect();
                Ok(out.into())
                // unimplemented!("get_mut with VecIndex is not implemented yet!")
            }
            ColumnSelector::VecName(vn) => {
                // may be slow, todo: improve the performance
                let out: Vec<&mut Expr> = self
                    .data
                    .iter_mut()
                    .filter(|e| {
                        let name = e.name().unwrap();
                        vn.iter().any(|n| {
                            if n.starts_with('^') & n.ends_with('$') {
                                let re = Regex::new(n).unwrap();
                                re.is_match(&name)
                            } else {
                                name == *n
                            }
                        })
                    })
                    .collect();
                Ok(out.into())
                // unimplemented!("get_mut with VecName is not implemented yet!")
            }
            ColumnSelector::VecNameOwned(vn) => {
                // may be slow, todo: improve the performance
                let out: Vec<&mut Expr> = self
                    .data
                    .iter_mut()
                    .filter(|e| {
                        let name = e.name().unwrap();
                        vn.iter().any(|n| {
                            if n.starts_with('^') & n.ends_with('$') {
                                let re = Regex::new(n).unwrap();
                                re.is_match(&name)
                            } else {
                                name == *n
                            }
                        })
                    })
                    .collect();
                Ok(out.into())
                // unimplemented!("get_mut with VecName is not implemented yet!")
            }
        }
    }

    /// Set a new column or replace an existed column using col name
    pub fn set(&mut self, col: ColumnSelector<'_>, expr: SetInput<'a>) -> TpResult<()> {
        match col {
            ColumnSelector::Name(name) => {
                if name.starts_with('^') & name.ends_with('$') {
                    let re = Regex::new(name).map_err(|_| StrError("Invalid regex!".into()))?;
                    return self.set(ColumnSelector::Regex(re), expr);
                }
                let mut expr = expr.into_expr()?;
                expr.rename(name.to_string());
                self.insert_inplace(expr)
            }
            ColumnSelector::VecName(name_vec) => {
                let exprs = expr.into_exprs();
                let name_vec = name_vec
                    .into_iter()
                    .flat_map(|n| self.get_selector_out_name(n.into()))
                    .collect::<Vec<_>>();
                if name_vec.len() != exprs.len() {
                    return Err("The number of expressions to set doesn't match the number of selected columns!".into());
                }
                for (col_name, expr) in zip(name_vec, exprs) {
                    self.set(col_name.into(), expr.into())?
                }
                Ok(())
            }
            ColumnSelector::VecNameOwned(name_vec) => {
                let exprs = expr.into_exprs();
                let name_vec = name_vec
                    .into_iter()
                    .flat_map(|n| self.get_selector_out_name(n.into()))
                    .collect::<Vec<_>>();
                if name_vec.len() != exprs.len() {
                    return Err("The number of expressions to set doesn't match the number of selected columns!".into());
                }
                for (col_name, expr) in zip(name_vec, exprs) {
                    self.set(col_name.into(), expr.into())?
                }
                Ok(())
            }
            ColumnSelector::Index(idx) => {
                let mut expr = expr.into_expr()?;
                let idx = self.valid_idx(idx)?.min(self.len());
                if idx < self.len() {
                    // replace a column
                    // We use the expr's name by default if the value expression has a name.
                    if expr.name().is_none() {
                        let ori_name = self.data[idx].name().unwrap();
                        expr.rename(ori_name)
                    }
                } else {
                    // insert a new column
                    if expr.name().is_none() {
                        let new_name = self.get_new_col_name();
                        expr.rename(new_name)
                    }
                }
                self.insert_inplace(expr)
            }
            ColumnSelector::VecIndex(vec_idx) => {
                let exprs = expr.into_exprs();
                if vec_idx.len() != exprs.len() {
                    return Err("The number of expressions to set doesn't match the number of selected columns!".into());
                }
                for (idx, expr) in zip(vec_idx, exprs) {
                    self.set(idx.into(), expr.into())?
                }
                Ok(())
            }
            ColumnSelector::NameOwned(name) => self.set(name.as_str().into(), expr),
            ColumnSelector::Regex(re) => {
                let exprs = expr.into_exprs();
                let set_col_vec = self.get_selector_out_name(ColumnSelector::Regex(re));
                if exprs.len() != set_col_vec.len() {
                    return Err("The number of expressions to set doesn't match the number of selected columns!".into());
                }
                for (col_name, expr) in zip(set_col_vec, exprs) {
                    self.set(col_name.into(), expr.into())?
                }
                Ok(())
            }
            ColumnSelector::All => {
                let exprs = expr.into_exprs();
                if exprs.len() != self.len() {
                    return Err(
                        "The number of expressions to set doesn't match the number of columns!"
                            .into(),
                    );
                }
                let columns = self.columns_owned();
                for (col_name, expr) in zip(columns, exprs) {
                    self.set(col_name.into(), expr.into())?
                }
                Ok(())
            }
        }
    }

    #[inline]
    pub fn eval(self, col: ColumnSelector, context: bool) -> TpResult<Self> {
        let mut df = self.clone();
        df.eval_inplace(col, context)?;
        Ok(df)
    }

    pub fn eval_inplace(&mut self, col: ColumnSelector, context: bool) -> TpResult<()> {
        // is there a good way to avoid clone at all cases? Currently we can not get a immutable reference
        // of self and a mutable reference of self at the same time.
        let context: Option<Context<'a>> = if context { self.clone().into() } else { None };
        let expr = self.get_mut(col)?;
        match expr {
            GetMutOutput::Expr(e) => {
                // we should update_column_map if the name of the expr has changed after evaluation.
                let ori_name = e.name().unwrap();
                e.eval_inplace_freeze(context)?;
                let new_name = e.name().unwrap();
                self.update_column_map(ori_name, new_name)?
            }
            GetMutOutput::Exprs(mut es) => {
                let ori_name_vec = es.iter().map(|e| e.name().unwrap()).collect_trusted();
                es.par_iter_mut()
                    .try_for_each(|e| e.eval_inplace_freeze(context.clone()).map(|_| {}))?;
                let new_name_vec = es.iter().map(|e| e.name().unwrap()).collect_trusted();
                zip(ori_name_vec, new_name_vec).try_for_each(|(ori_name, new_name)| {
                    self.update_column_map(ori_name, new_name)
                })?
            }
        }
        Ok(())
    }

    /// Insert a new expr or update the old value,
    /// the column name will be the name of the value expression.
    /// The caller must ensure that the name of the value is not None;
    pub fn insert_inplace(&mut self, expr: Expr<'a>) -> TpResult<()> {
        let new_name = expr
            .name()
            .ok_or(StrError("The name of the value expression is None!".into()))?;
        let len = self.len();
        if let Some(map) = Arc::get_mut(&mut self.map) {
            let col_idx = *map.get(&new_name).unwrap_or(&len);
            if col_idx == len {
                // insert a new column
                map.insert(new_name, len);
                self.data.push(expr);
            } else {
                // update a existed column
                let col = self.data.get_mut(col_idx).unwrap();
                let col_name_to_drop = col.name().unwrap(); // record the old name
                *col = expr;
                let res = map.insert(new_name, col_idx);
                if res.is_none() {
                    // the name is different, so we must drop the old name
                    map.remove(&col_name_to_drop);
                }
            }
        } else {
            let mut map = (*self.map).clone();
            let col_idx = *map.get(&new_name).unwrap_or(&len);
            if col_idx == len {
                // insert a new column
                map.insert(new_name, len);
                self.data.push(expr);
            } else {
                // update a existed column
                let col = self.data.get_mut(col_idx).unwrap();
                let col_name_to_drop = col.name().unwrap(); // record the old name
                *col = expr;
                let res = map.insert(new_name, col_idx);
                if res.is_none() {
                    // the name is different, so we must drop the old name
                    map.remove(&col_name_to_drop);
                }
            }
            self.map = Arc::new(map);
        }
        Ok(())
    }
}
