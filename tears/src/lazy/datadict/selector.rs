use pyo3::PyAny;
use regex::Regex;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub enum ColumnSelector<'a> {
    Index(i32),
    VecIndex(Vec<i32>),
    VecName(Vec<&'a str>),
    Name(&'a str),
    NameOwned(String),
    Regex(Regex),
    All,
}

impl ColumnSelector<'_> {
    pub fn name(&self) -> Option<String> {
        use ColumnSelector::*;
        match self {
            Index(_) => None,
            VecIndex(_) => None,
            VecName(_) => None,
            Name(name) => Some(name.to_string()),
            NameOwned(name) => Some(name.clone()),
            Regex(_) => None,
            All => None,
        }
    }
}

impl From<i32> for ColumnSelector<'_> {
    fn from(idx: i32) -> Self {
        ColumnSelector::Index(idx)
    }
}

impl From<usize> for ColumnSelector<'_> {
    fn from(idx: usize) -> Self {
        ColumnSelector::Index(idx as i32)
    }
}

impl From<Vec<i32>> for ColumnSelector<'_> {
    fn from(idx: Vec<i32>) -> Self {
        ColumnSelector::VecIndex(idx)
    }
}

impl From<Vec<usize>> for ColumnSelector<'_> {
    fn from(idx: Vec<usize>) -> Self {
        ColumnSelector::VecIndex(idx.into_iter().map(|x| x as i32).collect())
    }
}

impl<'a> From<Vec<&'a str>> for ColumnSelector<'a> {
    fn from(name: Vec<&'a str>) -> Self {
        ColumnSelector::VecName(name)
    }
}

impl<'a> From<&'a Vec<String>> for ColumnSelector<'a> {
    fn from(name: &'a Vec<String>) -> Self {
        ColumnSelector::VecName(name.iter().map(|x| x.as_str()).collect())
    }
}

impl<'a> From<&'a str> for ColumnSelector<'a> {
    fn from(name: &'a str) -> Self {
        ColumnSelector::Name(name)
    }
}

impl From<String> for ColumnSelector<'_> {
    fn from(name: String) -> Self {
        ColumnSelector::NameOwned(name)
    }
}

impl<'py> From<&'py PyAny> for ColumnSelector<'py> {
    fn from(select: &'py PyAny) -> Self {
        if let Ok(select) = select.extract::<&'py str>() {
            select.into()
        } else if let Ok(select) = select.extract::<i32>() {
            select.into()
        } else if let Ok(select) = select.extract::<Vec<&'py str>>() {
            select.into()
        } else if let Ok(select) = select.extract::<Vec<i32>>() {
            select.into()
        } else {
            panic!("Invalid column selector")
        }
    }
}

impl<'py> From<Option<&'py PyAny>> for ColumnSelector<'py> {
    fn from(select: Option<&'py PyAny>) -> Self {
        if let Some(select) = select {
            if select.is_none() {
                ColumnSelector::All
            } else {
                select.into()
            }
        } else {
            ColumnSelector::All
        }
    }
}
