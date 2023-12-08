use tea_core::error::TpResult;
use tea_core::utils::CollectTrustedToVec;
// use tea_lazy::SingleCol;
#[cfg(feature = "arw")]
use arrow::datatypes::Schema;

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

pub enum ColSelect<'a> {
    Idx(Vec<i32>),
    Name(Vec<&'a str>),
    NameOwned(Vec<String>),
    Null,
}

impl<T> From<Option<T>> for ColSelect<'_> {
    fn from(v: Option<T>) -> Self {
        if v.is_none() {
            ColSelect::Null
        } else {
            unreachable!("should not convert Some<T> to ColSelect")
        }
    }
}

impl<'a> ColSelect<'a> {
    #[cfg(feature = "arw")]
    pub fn into_proj(self, schema: &Schema) -> TpResult<Option<Vec<usize>>> {
        use super::arrow_io::utils::columns_to_projection;
        match self {
            ColSelect::Idx(idx) => {
                let len = schema.fields.len();
                let out = idx
                    .into_iter()
                    .map(|v| {
                        if v >= 0 {
                            v as usize
                        } else {
                            (len as i32 + v) as usize
                        }
                    })
                    .collect_trusted();
                Ok(Some(out))
            }
            ColSelect::Name(names) => {
                let out = columns_to_projection(&names, schema)?;
                Ok(Some(out))
            }
            ColSelect::NameOwned(names) => {
                let names_ref = names.iter().map(|s| s.as_str()).collect_trusted();
                let out = columns_to_projection(&names_ref, schema)?;
                Ok(Some(out))
            }
            ColSelect::Null => Ok(None),
        }
    }
}

impl<'a> From<SingleCol<'a>> for ColSelect<'a> {
    fn from(v: SingleCol<'a>) -> Self {
        match v {
            SingleCol::Idx(idx) => ColSelect::Idx(vec![idx]),
            SingleCol::Name(name) => ColSelect::Name(vec![name]),
            SingleCol::NameOwned(name) => ColSelect::NameOwned(vec![name]),
        }
    }
}
