//! Traits and iterators for ergonomic, efficient image access.
//!
//! - `ImageView`: read-only, with `row()` and optional flat `pixels()` when
//!   contiguous.
//! - `ImageViewMut`: adds `row_mut()` and optional `pixels_mut()` when
//!   contiguous.
//! - `Rows`/`RowsMut`: iterators over scanlines.
//!
//! Design: favor row-based iteration for kernels; expose flat iteration as a
//! fast path for whole-image operations. Avoid making images themselves
//! implement `Iterator` to preserve flexible ownership semantics.
pub trait ImageView {
    type Pixel: Copy;

    /// Image width in pixels
    fn width(&self) -> usize;
    /// Image height in pixels
    fn height(&self) -> usize;
    /// Elements between rows (equals `width` when contiguous)
    fn stride(&self) -> usize;

    /// Borrow the `y`-th scanline as a slice
    fn row(&self, y: usize) -> &[Self::Pixel];

    /// Iterate over scanlines
    fn rows(&self) -> Rows<'_, Self>
    where
        Self: Sized,
    {
        Rows { image: self, y: 0 }
    }

    /// True when `stride == width`, enabling flat access
    fn is_contiguous(&self) -> bool {
        self.stride() == self.width()
    }

    /// Borrow the whole buffer as a flat slice if contiguous
    fn as_slice(&self) -> Option<&[Self::Pixel]> {
        None
    }

    /// Iterate over flat pixels when contiguous (None otherwise)
    fn pixels(&self) -> Option<std::slice::Iter<'_, Self::Pixel>> {
        self.as_slice().map(|s| s.iter())
    }
}

pub trait ImageViewMut: ImageView {
    /// Borrow the `y`-th scanline as a mutable slice
    fn row_mut(&mut self, y: usize) -> &mut [Self::Pixel];

    /// Iterate over mutable scanlines
    fn rows_mut(&mut self) -> RowsMut<'_, Self>
    where
        Self: Sized,
    {
        RowsMut { image: self, y: 0 }
    }

    /// Borrow the whole buffer as a flat mutable slice if contiguous
    fn as_mut_slice(&mut self) -> Option<&mut [Self::Pixel]> {
        None
    }

    /// Iterate over flat mutable pixels when contiguous (None otherwise)
    fn pixels_mut(&mut self) -> Option<std::slice::IterMut<'_, Self::Pixel>> {
        self.as_mut_slice().map(|s| s.iter_mut())
    }
}

/// Iterator over immutable scanlines returned by `ImageView::rows()`
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

/// Iterator over mutable scanlines returned by `ImageViewMut::rows_mut()`
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
