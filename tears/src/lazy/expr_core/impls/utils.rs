use std::marker::PhantomData;

use ndarray::{Dim, IxDyn, IxDynImpl, SliceInfo, SliceInfoElem};

pub fn adjust_slice(
    mut slc: Vec<SliceInfoElem>,
    shape: &[usize],
    ndim: usize,
) -> SliceInfo<Vec<SliceInfoElem>, Dim<IxDynImpl>, Dim<IxDynImpl>> {
    // adjust the slice info in order to make it compatible with ndarray
    let mut axis = 0;
    slc.iter_mut().for_each(|elem| {
        match elem {
            SliceInfoElem::Slice { start, end, step } => {
                let len = shape[axis] as isize;
                // adjust when step is smaller than zero so the result when step < 0 is the same with numpy slice
                if *step < 0 {
                    if *start < 0 {
                        *start += len;
                    }
                    if end.is_some() {
                        let mut _end = *end.as_ref().unwrap();
                        if _end < 0 {
                            _end += len;
                        } else if _end >= len {
                            _end = len;
                        }
                        if *start > _end {
                            if *start == len {
                                *end = Some(len);
                            } else {
                                *end = Some(*start + 1)
                            };
                            if _end >= len - 2 {
                                *start = len - 1;
                            } else {
                                *start = _end + 1;
                            }
                        } else {
                            // if start < end and step < 0, we shouldn't slice anything
                            (*start, *end) = (0, Some(0));
                        }
                    }
                } else if let Some(_end) = end {
                    if *_end >= len {
                        *end = Some(len)
                    }
                }
                axis += 1;
            }
            SliceInfoElem::Index(_) => axis += 1,
            _ => {}
        }
    });
    let slc_len = slc.len();
    if slc_len < ndim {
        for _idx in slc_len..ndim {
            slc.push(SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1,
            })
        }
    }
    unsafe { SliceInfo::new_unchecked(slc, PhantomData::<IxDyn>, PhantomData::<IxDyn>) }
}
