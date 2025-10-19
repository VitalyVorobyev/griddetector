//! Owned single-channel f32 image in row-major layout (stride == width).
//!
//! Suited for numeric processing in the pipeline. Provides row access and a
//! contiguous slice when `stride == width`.
#[derive(Clone, Debug)]
pub struct ImageF32 {
    /// Image width in pixels
    pub w: usize,
    /// Image height in pixels
    pub h: usize,
    /// Number of f32 elements between consecutive rows (equals `w`)
    pub stride: usize,
    /// Backing storage in row-major order
    pub data: Vec<f32>,
}

impl ImageF32 {
    /// Construct a zero-initialized buffer of size `w × h`.
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            stride: w,
            data: vec![0.0; w * h],
        }
    }
    #[inline]
    /// Convert (x, y) to a linear index into `data`.
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.stride + x
    }
    #[inline]
    /// Get the pixel value at (x, y).
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[self.idx(x, y)]
    }
    #[inline]
    /// Set the pixel value at (x, y).
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

    #[inline]
    fn as_mut_slice(&mut self) -> Option<&mut [f32]> {
        if self.stride == self.w {
            Some(&mut self.data[..self.w * self.h])
        } else {
            None
        }
    }
}
