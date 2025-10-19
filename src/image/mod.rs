//! Image module: lightweight owned buffers, read-only views, and utilities.
//!
//! Components
//! - `u8`: read-only `ImageU8<'a>` view over 8-bit grayscale buffers.
//! - `f32`: owned `ImageF32` buffer for numeric processing (row-major, stride==w).
//! - `traits`: `ImageView`/`ImageViewMut` abstractions with row and flat iterators.
//! - `io`: helpers for loading/saving grayscale images and writing JSON.
//!
//! Design goals
//! - Keep hot loops simple and cache-friendly via row access.
//! - Expose a fast contiguous path (`as_slice`/`pixels`) when `stride == width`.
//! - Make ownership explicit: views borrow external data; `ImageF32` owns and mutates.
//!
//! See `doc/image.md` for a deeper overview and best practices.
pub mod f32;
pub mod io;
pub mod traits;
pub mod u8;

pub use self::f32::ImageF32;
pub use self::traits::{ImageView, ImageViewMut, Rows, RowsMut};
pub use self::u8::ImageU8;
