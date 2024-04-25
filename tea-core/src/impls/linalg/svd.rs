use super::{into_matrix, MatrixLayout};
use crate::TpResult;
use crate::{
    prelude::{Arr1, ArrD},
    utils::{vec_uninit, VecAssumeInit},
};
use lapack_sys::dgesvd_;
use libc::c_char;

impl ArrD<f64> {
    pub fn svd_into(
        self,
        full: bool,
        calc_uvt: bool,
    ) -> TpResult<(Option<Self>, Self, Option<Self>)> {
        let mut arr = self.to_dim2()?;
        let layout = arr.layout()?;
        let mut_arr = arr
            .as_slice_memory_order_mut()
            .expect("Array should be contiguous when svd");
        let (m, n) = (layout.lda(), layout.len());
        let k = m.min(n);
        let (_u_col, vt_row) = if calc_uvt {
            if full {
                (m, n)
            } else {
                (k, k)
            }
        } else {
            (m, n)
        };

        let (jobu, jobvt) = if calc_uvt {
            if full {
                (b'A', b'A')
            } else if m == k {
                (b'S', b'O')
            } else {
                (b'O', b'S')
            }
        } else {
            (b'N', b'N')
        };
        let (mut u, mut vt) = match (&jobu, &jobvt) {
            (b'A', b'A') => (
                Some(vec_uninit::<f64>(m as usize * m as usize)),
                Some(vec_uninit::<f64>(n as usize * n as usize)),
            ),
            (b'S', b'O') => (Some(vec_uninit(m as usize * k as usize)), None),
            (b'O', b'S') => (None, Some(vec_uninit(n as usize * k as usize))),
            (b'N', b'N') => (None, None),
            _ => unreachable!(),
        };
        let mut s = vec_uninit::<f64>(k as usize);
        // eval work size
        let mut info = 0;
        let mut work_size = [0.];
        unsafe {
            dgesvd_(
                &jobu as *const u8 as *const c_char,
                &jobvt as *const u8 as *const c_char,
                &m,
                &n,
                std::ptr::null_mut(),       // A
                &m,                         // lda
                s.as_mut_ptr() as *mut f64, // S
                u.as_mut()
                    .map(|x| x.as_mut_ptr())
                    .unwrap_or([].as_mut_ptr()) as *mut f64, // u
                &m,                         // ldu
                vt.as_mut()
                    .map(|x| x.as_mut_ptr())
                    .unwrap_or([].as_mut_ptr()) as *mut f64, // vt
                &vt_row,                    // ldvt
                work_size.as_mut_ptr(),
                &(-1), // lwork
                &mut info,
            );
        }
        if info != 0 {
            panic!("SVD error: info = {info}");
        }
        let lwork = work_size[0] as i32;
        let mut work = vec_uninit::<f64>(lwork.try_into().unwrap());
        info = 0;
        unsafe {
            dgesvd_(
                &jobu as *const u8 as *const c_char,
                &jobvt as *const u8 as *const c_char,
                &m,
                &n,
                mut_arr.as_mut_ptr(),
                &m,                         // lda
                s.as_mut_ptr() as *mut f64, // S
                u.as_mut()
                    .map(|x| x.as_mut_ptr())
                    .unwrap_or([].as_mut_ptr()) as *mut f64, // u
                &m,                         // ldu
                vt.as_mut()
                    .map(|x| x.as_mut_ptr())
                    .unwrap_or([].as_mut_ptr()) as *mut f64, // vt
                &vt_row,                    // ldvt
                work.as_mut_ptr() as *mut f64,
                &(lwork), // lwork
                &mut info,
            );
        }
        if info != 0 {
            panic!("SVD error: info = {info}");
        }
        let s = unsafe { s.assume_init() };
        let s = Arr1::from_vec(s).to_dimd();
        if !calc_uvt {
            Ok((None, s, None))
        } else {
            // let mut res_vec = Vec::<ArbArray<'a, f64>>::with_capacity(3);
            let (u, vt) = if !full {
                if m == k {
                    (
                        u.map(|x| unsafe { x.assume_init() }),
                        Some(arr.0.into_raw_vec()),
                    )
                } else {
                    (
                        Some(arr.0.into_raw_vec()),
                        vt.map(|x| unsafe { x.assume_init() }),
                    )
                }
            } else {
                (
                    u.map(|x| unsafe { x.assume_init() }),
                    vt.map(|x| unsafe { x.assume_init() }),
                )
            };
            let (u, vt) = match &layout {
                MatrixLayout::C { .. } => (vt, u),
                _ => (u, vt),
            };
            let (m, n) = layout.size();
            let (u_col, vt_row) = if full { (m, n) } else { (k, k) };
            let u: Self = into_matrix(layout.resized(m, u_col), u.unwrap())?.to_dimd();
            let vt: Self = into_matrix(layout.resized(vt_row, n), vt.unwrap())?.to_dimd();
            Ok((Some(u), s, Some(vt)))
        }
    }
}
