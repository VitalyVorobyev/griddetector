//! I/O helpers for grayscale images and JSON.
//!
//! - `load_grayscale_image`: read a PNG/JPEG/etc. into an owned 8-bit gray buffer.
//! - `save_grayscale_f32`: write an `ImageF32` to a grayscale PNG.
//! - `save_grayscale_u8`: write an owned 8-bit gray buffer to a PNG.
//! - `write_json_file`: pretty-print a serializable value to disk.
use super::{ImageF32, ImageU8, ImageView};
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use serde::Serialize;
use std::fs;
use std::path::Path;

/// Owned 8-bit grayscale buffer with stride and borrowed view conversion.
#[derive(Clone, Debug)]
pub struct GrayImageU8 {
    width: usize,
    height: usize,
    stride: usize,
    data: Vec<u8>,
}

impl GrayImageU8 {
    /// Construct an owned grayscale buffer given raw bytes.
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Self {
        let stride = width;
        Self {
            width,
            height,
            stride,
            data,
        }
    }

    /// Image width in pixels
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels
    pub fn height(&self) -> usize {
        self.height
    }

    /// Borrow as a read-only `ImageU8` view
    pub fn as_view(&self) -> ImageU8<'_> {
        ImageU8 {
            w: self.width,
            h: self.height,
            stride: self.stride,
            data: &self.data,
        }
    }
}

/// Load an image from disk and convert to 8-bit grayscale.
pub fn load_grayscale_image(path: &Path) -> Result<GrayImageU8, String> {
    let img = image::open(path)
        .map_err(|e| format!("Failed to open {}: {e}", path.display()))?
        .into_luma8();
    let width = img.width() as usize;
    let height = img.height() as usize;
    let data = img.into_raw();
    Ok(GrayImageU8::new(width, height, data))
}

/// Save a float image to a grayscale PNG, clamping values in [0, 255].
pub fn save_grayscale_f32(image: &ImageF32, path: &Path) -> Result<(), String> {
    ensure_parent_dir(path)?;
    let mut out = GrayImage::new(image.w as u32, image.h as u32);
    for y in 0..image.h {
        let row = image.row(y);
        for (x, &px) in row.iter().enumerate() {
            let v = (px * 255.0).clamp(0.0, 255.0);
            out.put_pixel(x as u32, y as u32, Luma([v as u8]));
        }
    }
    out.save(path)
        .map_err(|e| format!("Failed to save {}: {e}", path.display()))
}

/// Save an 8-bit grayscale buffer to a PNG.
pub fn save_grayscale_u8(buffer: &GrayImageU8, path: &Path) -> Result<(), String> {
    ensure_parent_dir(path)?;
    let data = buffer.data.clone();
    let image: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_raw(buffer.width as u32, buffer.height as u32, data)
            .ok_or_else(|| "Failed to create image buffer".to_string())?;
    DynamicImage::ImageLuma8(image)
        .save(path)
        .map_err(|e| format!("Failed to save {}: {e}", path.display()))
}

/// Serialize a value as pretty JSON to `path`, creating parent directories.
pub fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    ensure_parent_dir(path)?;
    let json = serde_json::to_string_pretty(value)
        .map_err(|e| format!("Failed to serialize JSON for {}: {e}", path.display()))?;
    fs::write(path, json).map_err(|e| format!("Failed to write JSON {}: {e}", path.display()))
}

fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create {}: {e}", parent.display()))?;
        }
    }
    Ok(())
}
