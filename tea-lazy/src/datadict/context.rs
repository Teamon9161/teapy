use super::DataDict;
use std::sync::Arc;

pub type Context<'a> = Arc<DataDict<'a>>;

impl<'a> From<DataDict<'a>> for Option<Context<'a>> {
    #[inline(always)]
    fn from(dd: DataDict<'a>) -> Self {
        Some(Arc::new(dd))
    }
}
