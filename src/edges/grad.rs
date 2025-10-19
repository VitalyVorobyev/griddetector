//! Image gradients (Sobel/Scharr) with magnitude and quantized orientation.
//!
//! - Convolves a 3×3 kernel pair (`X` and `Y`) with border clamping.
//! - Outputs per‑pixel `gx`, `gy`, `mag = sqrt(gx^2+gy^2)`.
//! - Caches a compact 8‑bin orientation quantization (π‑periodic) per pixel
//!   for downstream use (e.g., region growing, orientation histograms).
//!
//! Orientation quantization maps the continuous angle `atan2(gy, gx)` in
//! (−π, π] to 8 uniform bins over [0, 2π), then folds modulo π.
//!
//! Complexity: O(W·H) per pass; memory: three float buffers + 1 byte/pixel.
use crate::image::{ImageF32, ImageView, ImageViewMut};

type Kernel3 = [[f32; 3]; 3];

const SOBEL_KERNEL_X: Kernel3 = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
const SOBEL_KERNEL_Y: Kernel3 = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

const SCHARR_KERNEL_X: Kernel3 = [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]];
const SCHARR_KERNEL_Y: Kernel3 = [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]];

/// Per‑pixel gradient buffers and orientation quantization.
#[derive(Clone, Debug)]
pub struct Grad {
    /// Horizontal derivative (convolution with kernel X)
    pub gx: ImageF32,
    /// Vertical derivative (convolution with kernel Y)
    pub gy: ImageF32,
    /// Euclidean magnitude per pixel: `sqrt(gx^2 + gy^2)`
    pub mag: ImageF32,
    /// Per‑pixel quantized orientation in 8 bins (π‑periodic)
    pub ori_q8: Vec<u8>,
}

#[inline]
fn quantize_orientation(angle: f32) -> u8 {
    let wrapped = (angle + std::f32::consts::PI).rem_euclid(2.0 * std::f32::consts::PI);
    ((wrapped * (4.0 / std::f32::consts::PI)).floor() as i32 & 7) as u8
}

fn gradients_with_kernels(l: &ImageF32, kernel_x: &Kernel3, kernel_y: &Kernel3) -> Grad {
    let w = l.w;
    let h = l.h;
    let mut gx = ImageF32::new(w, h);
    let mut gy = ImageF32::new(w, h);
    let mut mag = ImageF32::new(w, h);
    let mut ori_q8 = vec![0u8; w * h];

    if w == 0 || h == 0 {
        return Grad {
            gx,
            gy,
            mag,
            ori_q8,
        };
    }

    for y in 0..h {
        let y_idx = [y.saturating_sub(1), y, (y + 1).min(h - 1)];
        let rows = [l.row(y_idx[0]), l.row(y_idx[1]), l.row(y_idx[2])];
        let out_gx = gx.row_mut(y);
        let out_gy = gy.row_mut(y);
        let out_mag = mag.row_mut(y);
        for x in 0..w {
            let x_idx = [x.saturating_sub(1), x, (x + 1).min(w - 1)];

            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            for (ky, yy_row) in rows.iter().enumerate() {
                let kx_row = &kernel_x[ky];
                let ky_row = &kernel_y[ky];
                sum_x += yy_row[x_idx[0]] * kx_row[0]
                    + yy_row[x_idx[1]] * kx_row[1]
                    + yy_row[x_idx[2]] * kx_row[2];
                sum_y += yy_row[x_idx[0]] * ky_row[0]
                    + yy_row[x_idx[1]] * ky_row[1]
                    + yy_row[x_idx[2]] * ky_row[2];
            }

            out_gx[x] = sum_x;
            out_gy[x] = sum_y;
            let magnitude = (sum_x * sum_x + sum_y * sum_y).sqrt();
            out_mag[x] = magnitude;
            let idx = y * w + x;
            ori_q8[idx] = quantize_orientation(sum_y.atan2(sum_x));
        }
    }

    Grad {
        gx,
        gy,
        mag,
        ori_q8,
    }
}

/// Compute Sobel gradients on a single‑channel float image.
pub fn sobel_gradients(l: &ImageF32) -> Grad {
    gradients_with_kernels(l, &SOBEL_KERNEL_X, &SOBEL_KERNEL_Y)
}

/// Compute Scharr gradients (better rotational symmetry than Sobel).
pub fn scharr_gradients(l: &ImageF32) -> Grad {
    gradients_with_kernels(l, &SCHARR_KERNEL_X, &SCHARR_KERNEL_Y)
}
