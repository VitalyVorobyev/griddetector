//! Read-only single-channel u8 view over externally owned grayscale data.
//!
//! Provides fast row access and an optional contiguous slice when
//! `stride == width`. Used as the input type for building the image pyramid.
#[derive(Clone, Debug)]
pub struct ImageU8<'a> {
    /// Image width in pixels
    pub w: usize,
    /// Image height in pixels
    pub h: usize,
    /// Bytes between rows (equals `w` for tightly packed buffers)
    pub stride: usize,
    /// Borrowed backing storage in row-major order
    pub data: &'a [u8],
}

impl<'a> ImageU8<'a> {
    #[inline]
    /// Get the pixel value at (x, y).
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.stride + x]
    }
}

impl<'a> crate::image::traits::ImageView for ImageU8<'a> {
    type Pixel = u8;

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
    fn row(&self, y: usize) -> &[u8] {
        let start = y * self.stride;
        &self.data[start..start + self.w]
    }
    #[inline]
    fn as_slice(&self) -> Option<&[u8]> {
        (self.stride == self.w).then_some(&self.data[..self.w * self.h])
    }
}
