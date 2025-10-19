#[derive(Clone, Debug)]
pub struct ImageF32 {
    pub w: usize,
    pub h: usize,
    pub stride: usize, // number of f32 elements between rows
    pub data: Vec<f32>,
}

impl ImageF32 {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            stride: w,
            data: vec![0.0; w * h],
        }
    }
    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.stride + x
    }
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[self.idx(x, y)]
    }
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        let i = self.idx(x, y);
        self.data[i] = v;
    }
}

impl crate::image::traits::ImageView for ImageF32 {
    type Pixel = f32;

    #[inline]
    fn width(&self) -> usize {
        self.w
    }
    #[inline]
    fn height(&self) -> usize {
        self.h
    }
    #[inline]
    fn stride(&self) -> usize {
        self.stride
    }
    #[inline]
    fn row(&self, y: usize) -> &[f32] {
        let start = y * self.stride;
        &self.data[start..start + self.w]
    }
    #[inline]
    fn as_slice(&self) -> Option<&[f32]> {
        (self.stride == self.w).then_some(&self.data[..self.w * self.h])
    }
}

impl crate::image::traits::ImageViewMut for ImageF32 {
    #[inline]
    fn row_mut(&mut self, y: usize) -> &mut [f32] {
        let start = y * self.stride;
        let end = start + self.w;
        &mut self.data[start..end]
    }
}
