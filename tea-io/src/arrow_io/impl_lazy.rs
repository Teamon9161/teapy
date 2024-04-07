use crate::{ColSelect, SingleCol};
use std::path::Path;
use tea_core::error::TpResult;
use tea_core::utils::CollectTrustedToVec;
use tea_lazy::{DataDict, Expr};

#[ext_trait]
impl<'a> ExprIPCExt for Expr<'a> {
    pub fn read_ipc<P>(p: P, col: SingleCol<'a>) -> Self
    where
        P: AsRef<Path> + Send + Sync + Clone + 'a,
    {
        let mut e: Expr<'a> = 0_i32.into();
        let name = if let SingleCol::Name(s) = col {
            Some(s.to_owned())
        } else if let SingleCol::NameOwned(s) = &col {
            Some(s.clone())
        } else {
            None
        };
        e.set_name(name);
        e.chain_f_ctx(move |(_data, _ctx)| {
            let (_schema, mut arr) = super::read_ipc(p.clone(), col.clone().into())?;
            let arr = arr.pop().unwrap();
            Ok((arr.into(), None))
        });
        e
    }
}

pub fn scan_ipc_lazy<'a, P>(path: P, columns: ColSelect<'_>) -> TpResult<Vec<Expr<'a>>>
where
    P: AsRef<Path> + Send + Sync + Clone + 'a,
{
    let mut schema = super::read_ipc_schema(path.clone())?;
    let proj = columns.into_proj(&schema)?;
    if let Some(proj) = proj {
        schema = schema.filter(|i, _f| proj.contains(&i));
    }
    let out = schema
        .fields
        .into_iter()
        .map(|f| Expr::read_ipc(path.clone(), f.name.into()))
        .collect_trusted();
    Ok(out)
}

#[ext_trait]
impl<'a> DataDictIPCExt for DataDict<'a> {
    pub fn read_ipc<P: AsRef<Path>>(path: P, columns: ColSelect<'_>) -> TpResult<DataDict<'a>> {
        let (schema, arr_vec) = super::read_ipc(path, columns)?;
        let data: Vec<Expr<'a>> = schema
            .fields
            .into_iter()
            .zip(arr_vec)
            .map(|(s, a)| Expr::new_from_arr(a, Some(s.name)))
            .collect();
        Ok(DataDict::new(data, None))
    }

    pub fn scan_ipc<P>(path: P, columns: ColSelect<'_>) -> TpResult<DataDict<'a>>
    where
        P: AsRef<Path> + Send + Sync + Clone + 'a,
    {
        let out = scan_ipc_lazy(path, columns)?;
        Ok(Self::new(out, None))
    }
}
