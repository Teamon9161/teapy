#[cfg(feature = "arw")]
use arrow::datatypes::Schema;

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
                let out = super::utils::columns_to_projection(&names, schema)?;
                Ok(Some(out))
            }
            ColSelect::NameOwned(names) => {
                let names_ref = names.iter().map(|s| s.as_str()).collect_trusted();
                let out = super::utils::columns_to_projection(&names_ref, schema)?;
                Ok(Some(out))
            }
            ColSelect::Null => Ok(None),
        }
    }
}
