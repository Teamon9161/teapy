use super::super::expr_inner::ExprInner;
use super::export::*;
use std::path::Path;

#[derive(Clone)]
pub enum SingleCol<'a> {
    Idx(i32),
    Name(&'a str),
    NameOwned(String),
}

impl From<i32> for SingleCol<'_> {
    fn from(v: i32) -> Self {
        SingleCol::Idx(v)
    }
}

impl<'a> From<&'a str> for SingleCol<'a> {
    fn from(v: &'a str) -> Self {
        SingleCol::Name(v)
    }
}

impl From<String> for SingleCol<'_> {
    fn from(v: String) -> Self {
        SingleCol::NameOwned(v)
    }
}

impl<'a> From<SingleCol<'a>> for crate::arrow_io::ColSelect<'a> {
    fn from(v: SingleCol<'a>) -> Self {
        match v {
            SingleCol::Idx(idx) => crate::arrow_io::ColSelect::Idx(vec![idx]),
            SingleCol::Name(name) => crate::arrow_io::ColSelect::Name(vec![name]),
            SingleCol::NameOwned(name) => crate::arrow_io::ColSelect::NameOwned(vec![name]),
        }
    }
}

#[cfg(feature = "arw")]
impl<'a> Expr<'a> {
    pub fn read_ipc<P>(p: P, col: SingleCol<'a>) -> Self
    where
        P: AsRef<Path> + Send + Sync + Clone + 'a,
    {
        let mut e: ExprInner<'a> = 0i32.into();
        let name = if let SingleCol::Name(s) = col {
            Some(s.to_owned())
        } else if let SingleCol::NameOwned(s) = &col {
            Some(s.clone())
        } else {
            None
        };
        e.set_name(name);
        e.chain_f_ctx(move |(_data, _ctx)| {
            let (_schema, mut arr) = crate::arrow_io::read_ipc(p.clone(), col.clone().into())?;
            let arr = arr.pop().unwrap();
            Ok((arr.into(), None))
        });
        e.into()
    }
}
