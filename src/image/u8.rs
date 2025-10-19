#[derive(Clone, Debug)]
pub struct ImageU8<'a> {
    pub w: usize,
    pub h: usize,
    pub stride: usize, // bytes between rows
    pub data: &'a [u8],
}

impl<'a> ImageU8<'a> {
    #[inline]
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
