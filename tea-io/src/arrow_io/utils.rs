use arrow::datatypes::Schema;
use tea_core::prelude::{tbail, TResult};

pub(crate) fn columns_to_projection(columns: &[&str], schema: &Schema) -> TResult<Vec<usize>> {
    use tea_hash::TpHashMap;

    let mut prj = Vec::with_capacity(columns.len());
    if columns.len() > 100 {
        let mut column_names = TpHashMap::with_capacity(schema.fields.len());
        schema.fields.iter().enumerate().for_each(|(i, c)| {
            column_names.insert(c.name.as_str(), i);
        });

        for column in columns {
            let Some(&i) = column_names.get(column) else {
                tbail!("unable to find column {:?}", column);
            };
            prj.push(i);
        }
    } else {
        for column in columns {
            for (i, f) in schema.fields.iter().enumerate() {
                if f.name == *column {
                    prj.push(i);
                    break;
                }
            }
        }
    }
    Ok(prj)
}
