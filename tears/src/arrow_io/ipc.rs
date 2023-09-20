// use arrow::array::Array;
use arrow::datatypes::Schema;
use arrow::mmap::{mmap_dictionaries_unchecked, mmap_unchecked};
use arrow::{error::Error, io::ipc::read};
use memmap::Mmap;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::{fs::File, path::Path, sync::Arc};

use crate::{ArrOk, CollectTrustedToVec, TpResult};

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

#[inline]
pub fn get_ipc_schema<P: AsRef<Path>>(path: P) -> TpResult<Schema> {
    let mut file = File::open(path)?;
    let metadata = read::read_file_metadata(&mut file)?;
    Ok(metadata.schema)
}

// #[inline]
// pub fn filter_arr_by_proj(arrs: Vec<Box<dyn Array>>, schema: Schema, proj: &[usize]) -> (Vec<Box<dyn Array>>, Schema) {
//     let schema = schema.filter(|i, _f| proj.contains(&i));
//     let arrs = arrs.into_iter().enumerate().filter(|(i, _)| proj.contains(i)).map(|(_, a)| a).collect::<Vec<_>>();
//     (arrs, schema)
// }

pub fn read_ipc<'a, P: AsRef<Path>>(
    path: P,
    columns: ColSelect<'_>,
) -> TpResult<(Schema, Vec<ArrOk<'a>>)> {
    let mut file = File::open(path)?;
    let mmap = Arc::new(unsafe { Mmap::map(&file)? });

    // read the files' metadata. At this point, we can distribute the read whatever we like.
    let metadata = read::read_file_metadata(&mut file)?;
    let blocks_num = metadata.blocks.len();
    let mut schema = metadata.schema.clone();
    let proj = columns.into_proj(&schema)?;

    let dictionaries = unsafe { mmap_dictionaries_unchecked(&metadata, mmap.clone())? };
    let chunk = unsafe { mmap_unchecked(&metadata, &dictionaries, mmap.clone(), 0) };
    let chunk = match chunk {
        Ok(chunk) => chunk,
        Err(e) => {
            if let Error::NotYetImplemented(e) = &e {
                if e == "mmap can only be done on uncompressed IPC files" {
                    // unimplemented!("mmap can only be done on uncompressed IPC files")
                    let mut reader = read::FileReader::new(file, metadata, proj, None);
                    let schema = reader.schema().clone();
                    let out: Vec<ArrOk<'a>> = if blocks_num > 1 {
                        let mut arrs = reader
                            .map(|batch| batch.unwrap().into_arrays())
                            .collect::<Vec<_>>();
                        let arrs = (0..arrs.get(0).unwrap().len())
                            .map(|_| {
                                let mut out = Vec::with_capacity(arrs.len());
                                // out.push(arrs.pop().unwrap());
                                arrs.iter_mut().for_each(|a| out.push(a.pop().unwrap()));
                                out
                            })
                            .collect::<Vec<_>>();
                        arrs.into_par_iter()
                            .rev()
                            .map(|a| {
                                let a = a
                                    .into_par_iter()
                                    .map(|arr| ArrOk::from_arrow(arr))
                                    .collect();
                                ArrOk::same_dtype_concat_1d(a)
                            })
                            .collect()
                    } else {
                        let chunk = reader.next().unwrap().unwrap();
                        chunk
                            .into_arrays()
                            .into_par_iter()
                            .map(|a| ArrOk::from_arrow(a))
                            .collect()
                    };
                    return Ok((schema, out));
                } else {
                    return Err(e.to_owned().into());
                }
            } else {
                return Err(e.into());
            }
        }
    };

    let arrs = chunk.into_arrays();
    let mut arrs = if let Some(proj) = &proj {
        schema = schema.filter(|i, _f| proj.contains(&i));
        arrs.into_iter()
            .enumerate()
            .filter(|(i, _)| proj.contains(i))
            .map(|(_, a)| a)
            .collect::<Vec<_>>()
    } else {
        arrs
    };
    let mut other_arrs = (1..blocks_num)
        .map(|i| {
            let chunk =
                unsafe { mmap_unchecked(&metadata, &dictionaries, mmap.clone(), i).unwrap() };
            let arrs = chunk.into_arrays();
            if let Some(proj) = &proj {
                arrs.into_iter()
                    .enumerate()
                    .filter(|(i, _)| proj.contains(i))
                    .map(|(_, a)| a)
                    .collect::<Vec<_>>()
            } else {
                arrs
            }
        })
        .collect::<Vec<_>>();

    // let out: Vec<ArrOk<'a>> = arrs.into_par_iter().map(|a| {
    //     ArrOk::from_arrow(a)
    // }).collect();
    let out: Vec<ArrOk<'a>> = if !other_arrs.is_empty() {
        let arrs = (0..arrs.len())
            .map(|_| {
                let mut out = Vec::with_capacity(other_arrs.len() + 1);
                out.push(arrs.pop().unwrap());
                other_arrs
                    .iter_mut()
                    .for_each(|a| out.push(a.pop().unwrap()));
                out
            })
            .collect::<Vec<_>>();
        arrs.into_par_iter()
            .rev()
            .map(|a| {
                let a = a
                    .into_par_iter()
                    .map(|arr| ArrOk::from_arrow(arr))
                    .collect();
                ArrOk::same_dtype_concat_1d(a)
            })
            .collect()
    } else {
        arrs.into_par_iter().map(|a| ArrOk::from_arrow(a)).collect()
    };
    Ok((schema, out))
}
