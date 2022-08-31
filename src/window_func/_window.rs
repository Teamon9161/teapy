use super::prelude::*;

pub struct Window<T: Number> {
    f: fn(&mut Window<T>, v: T), // 步进函数，滚动到下一个窗口
    data: Vec<T>,
    window: usize,
    min_periods: usize,
    sum_flag: bool,
    std_flag: bool,
    valid_window: usize,
    sum: T,
    sum2: T,
    std: T,
    head_idx: usize, // 记录最早窗口的位置索引
}

impl<T: Number> Window<T> {
    pub fn new(data: Vec<T>, window: usize, min_periods: usize, sum_flag: bool, std_flag: bool) -> Self 
    where usize: AsPrimitive<T>
    {   
        assert!(window >= min_periods, "滚动window不能小于最小期数");
        assert!(window >= 1, "滚动window不能小于1");
        assert!(data.len() == window, "初始化窗口时，数据长度和window长度不相等");
        let init_value = 0_usize.as_();
        let mut w = Window {
            f: Window::push0,
            data,
            window, 
            min_periods,
            sum_flag,
            std_flag,
            valid_window: window,
            sum: init_value,
            sum2: init_value,
            std: init_value,
            head_idx: 0_usize,
        };
        // 计算初始窗口的合法期数，和、平方和、标准差等
        match (sum_flag, std_flag) {
            (_, true) => {
                todo!()
            },
            (true, false) => { // 只需求可用窗口以及窗口中的和
                w.f = Window::push1;
                for i in 0..window {
                    unsafe { // 安全性：i必定比data的长度要小
                        let v = *w.data.get_unchecked(i);
                        if IsNan!(v) {w.valid_window -= 1} else {w.sum += v};
                    }
                }
            },
            (false, false) => {
                for i in 0..window {
                    unsafe { // 安全性：i必定比data的长度要小
                        let v = *w.data.get_unchecked(i);
                        if IsNan!(v) {w.valid_window -= 1};
                    }
                }
            }
        }
        w    
    }

    #[inline(always)]
    fn next(&mut self) {
        self.head_idx += 1;
        if self.head_idx >= self.window {self.head_idx = 0;}
    }

    #[inline(always)]
    fn push0(&mut self, v: T) { // 最低计算要求的滚动，只更新可用窗口
        unsafe { // 安全性：window的长度必定大于等于1
            if IsNan!(*self.data.get_unchecked(self.head_idx)) {
                self.valid_window += 1; // 有一个nan数据过期，可用窗口加一
            }
        }
        if IsNan!(v) {
            self.valid_window -= 1;
        } 
        unsafe { *self.data.get_unchecked_mut(self.head_idx) = v }
        self.next();
    }

    #[inline(always)]
    fn push1(&mut self, v: T) { // 更新可用窗口和窗口的和
        let v_first = unsafe { *self.data.get_unchecked(self.head_idx)}; // 安全性：window的长度必定大于等于1
        if IsNan!(v_first) { self.valid_window += 1;} 
        else { self.sum -= v_first;}
        if IsNan!(v) {self.valid_window -= 1;}
        else { self.sum += v;}
        unsafe { *self.data.get_unchecked_mut(self.head_idx) = v }
        self.next();
    }

    #[inline(always)]
    fn push2(&mut self, v: T) { // 更新可用窗口、窗口的和、窗口的平方和
        let v_first = unsafe { *self.data.get_unchecked(self.head_idx)}; // 安全性：window的长度必定大于等于1
        if IsNan!(v_first) { self.valid_window += 1;} 
        else { self.sum -= v_first; self.sum2 -= v_first * v_first }
        if IsNan!(v) { self.valid_window -= 1; }
        else { self.sum += v; self.sum2 += v * v }
        unsafe { *self.data.get_unchecked_mut(self.head_idx) = v }
        self.next();
    }

    #[inline(always)]
    pub fn f(&mut self, v: T) {
        (self.f)(self, v)
    }

    #[inline(always)]
    pub fn mean(& self) -> f64 { // 窗口均值
        if self.valid_window >= self.min_periods {
            AsPrimitive::<f64>::as_(self.sum) / AsPrimitive::<f64>::as_(self.valid_window)
        }
        else {
            f64::NAN
        }
    }

    #[inline(always)]
    pub fn sum(&self) -> f64 { if self.valid_window >= self.min_periods {self.sum.as_()} else {f64::NAN} } // 窗口和

    #[inline(always)]
    pub fn std(&self) -> f64 {
        let mean2 = if self.valid_window >= self.min_periods {
            let mut v = AsPrimitive::<f64>::as_(self.sum2) / AsPrimitive::<f64>::as_(self.valid_window);
            v -= self.mean().powi(2);
            v
        }
        else { f64::NAN };
        let v_window = AsPrimitive::<f64>::as_(self.valid_window);
        (mean2 * v_window / (v_window - 1f64)).sqrt()
    }

}