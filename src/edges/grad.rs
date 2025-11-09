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

/// Lightweight view over per-level gradient buffers reused by refinement.
#[derive(Clone, Copy, Debug)]
pub struct GradientLevel<'a> {
    pub width: usize,
    pub height: usize,
    pub gx: &'a [f32],
    pub gy: &'a [f32],
    pub mag: &'a [f32],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GradientSample {
    pub gx: f32,
    pub gy: f32,
    pub mag: f32,
}

impl<'a> GradientLevel<'a> {
    pub fn from_grad(grad: &'a Grad) -> Self {
        debug_assert_eq!(grad.gx.stride, grad.gx.w);
        debug_assert_eq!(grad.gy.stride, grad.gy.w);
        debug_assert_eq!(grad.mag.stride, grad.mag.w);
        Self {
            width: grad.gx.w,
            height: grad.gx.h,
            gx: &grad.gx.data[..],
            gy: &grad.gy.data[..],
            mag: &grad.mag.data[..],
        }
    }

    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> Option<GradientSample> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        if self.width == 0 || self.height == 0 {
            return None;
        }

        let max_x = (self.width - 1) as isize;
        let max_y = (self.height - 1) as isize;

        if x < 0.0 || y < 0.0 || x > max_x as f32 || y > max_y as f32 {
            return None;
        }

        let xf = x.floor() as isize;
        let yf = y.floor() as isize;
        let x0 = xf.clamp(0, max_x);
        let y0 = yf.clamp(0, max_y);
        let x1 = (x0 + 1).min(max_x);
        let y1 = (y0 + 1).min(max_y);

        let tx = if x1 == x0 {
            0.0
        } else {
            (x - x0 as f32).clamp(0.0, 1.0)
        };
        let ty = if y1 == y0 {
            0.0
        } else {
            (y - y0 as f32).clamp(0.0, 1.0)
        };

        let idx = |xx: isize, yy: isize| -> usize { yy as usize * self.width + xx as usize };

        let lerp = |a: f32, b: f32, t: f32| a * (1.0 - t) + b * t;

        let gx0 = lerp(self.gx[idx(x0, y0)], self.gx[idx(x1, y0)], tx);
        let gx1 = lerp(self.gx[idx(x0, y1)], self.gx[idx(x1, y1)], tx);
        let gx = lerp(gx0, gx1, ty);

        let gy0 = lerp(self.gy[idx(x0, y0)], self.gy[idx(x1, y0)], tx);
        let gy1 = lerp(self.gy[idx(x0, y1)], self.gy[idx(x1, y1)], tx);
        let gy = lerp(gy0, gy1, ty);

        let mag0 = lerp(self.mag[idx(x0, y0)], self.mag[idx(x1, y0)], tx);
        let mag1 = lerp(self.mag[idx(x0, y1)], self.mag[idx(x1, y1)], tx);
        let mag = lerp(mag0, mag1, ty);

        Some(GradientSample { gx, gy, mag })
    }
}

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
