pub trait ImageView {
    type Pixel: Copy;

    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn stride(&self) -> usize;

    fn row(&self, y: usize) -> &[Self::Pixel];

    fn rows(&self) -> Rows<'_, Self>
    where
        Self: Sized,
    {
        Rows { image: self, y: 0 }
    }

    fn is_contiguous(&self) -> bool {
        self.stride() == self.width()
    }

    fn as_slice(&self) -> Option<&[Self::Pixel]> {
        None
    }
}

pub trait ImageViewMut: ImageView {
    fn row_mut(&mut self, y: usize) -> &mut [Self::Pixel];

    fn rows_mut(&mut self) -> RowsMut<'_, Self>
    where
        Self: Sized,
    {
        RowsMut { image: self, y: 0 }
    }
}

pub struct Rows<'a, I: ?Sized + ImageView> {
    image: &'a I,
    y: usize,
}

impl<'a, I: ImageView> Iterator for Rows<'a, I> {
    type Item = &'a [I::Pixel];

    fn next(&mut self) -> Option<Self::Item> {
        if self.y >= self.image.height() {
            return None;
        }
        let y = self.y;
        self.y += 1;
        Some(self.image.row(y))
    }
}

pub struct RowsMut<'a, I: ?Sized + ImageViewMut> {
    image: &'a mut I,
    y: usize,
}

impl<'a, I: ImageViewMut> Iterator for RowsMut<'a, I> {
    type Item = &'a mut [I::Pixel];

    fn next(&mut self) -> Option<Self::Item> {
        if self.y >= self.image.height() {
            return None;
        }
        // Reborrow trick to obtain a new &mut for each row
        let y = self.y;
        self.y += 1;
        let ptr = self.image as *mut I;
        // SAFETY: Each row y is returned at most once and rows do not alias.
        Some(unsafe { (&mut *ptr).row_mut(y) })
    }
}
