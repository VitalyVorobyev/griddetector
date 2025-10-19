use crate::types::ImageF32;
use serde::Serialize;

type Kernel3 = [[f32; 3]; 3];

const SOBEL_KERNEL_X: Kernel3 = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
const SOBEL_KERNEL_Y: Kernel3 = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

const SCHARR_KERNEL_X: Kernel3 = [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]];
const SCHARR_KERNEL_Y: Kernel3 = [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]];

#[derive(Clone, Debug)]
pub struct Grad {
    pub gx: ImageF32,
    pub gy: ImageF32,
    pub mag: ImageF32,
    pub ori_q8: Vec<u8>, // per-pixel quantized orientation in 8 bins
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EdgeElement {
    pub x: u32,
    pub y: u32,
    pub magnitude: f32,
    /// Gradient direction in radians, range (-π, π]
    pub direction: f32,
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
        for x in 0..w {
            let x_idx = [x.saturating_sub(1), x, (x + 1).min(w - 1)];

            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            for (ky, &yy) in y_idx.iter().enumerate() {
                let kernel_row_x = &kernel_x[ky];
                let kernel_row_y = &kernel_y[ky];
                for (xx, (&kx_weight, &ky_weight)) in x_idx
                    .iter()
                    .zip(kernel_row_x.iter().zip(kernel_row_y.iter()))
                {
                    let sample = l.get(*xx, yy);
                    sum_x += sample * kx_weight;
                    sum_y += sample * ky_weight;
                }
            }

            gx.set(x, y, sum_x);
            gy.set(x, y, sum_y);

            let magnitude = (sum_x * sum_x + sum_y * sum_y).sqrt();
            mag.set(x, y, magnitude);
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

pub fn sobel_gradients(l: &ImageF32) -> Grad {
    gradients_with_kernels(l, &SOBEL_KERNEL_X, &SOBEL_KERNEL_Y)
}

pub fn scharr_gradients(l: &ImageF32) -> Grad {
    gradients_with_kernels(l, &SCHARR_KERNEL_X, &SCHARR_KERNEL_Y)
}

/// Simple Sobel-based edge detector with 4-neighborhood non-maximum suppression.
pub fn detect_edges_sobel_nms(l: &ImageF32, mag_thresh: f32) -> Vec<EdgeElement> {
    let grad = sobel_gradients(l);
    let w = l.w;
    let h = l.h;
    if w < 3 || h < 3 {
        return Vec::new();
    }

    let mut edges = Vec::new();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mag = grad.mag.get(x, y);
            if mag < mag_thresh {
                continue;
            }

            let gx = grad.gx.get(x, y);
            let gy = grad.gy.get(x, y);
            let angle = gy.atan2(gx);
            let mut angle_deg = angle.to_degrees();
            if angle_deg < 0.0 {
                angle_deg += 180.0;
            }

            let (n1x, n1y, n2x, n2y) = if angle_deg < 22.5 || angle_deg >= 157.5 {
                (x - 1, y, x + 1, y)
            } else if angle_deg < 67.5 {
                (x + 1, y - 1, x - 1, y + 1)
            } else if angle_deg < 112.5 {
                (x, y - 1, x, y + 1)
            } else {
                (x - 1, y - 1, x + 1, y + 1)
            };

            let neighbor1 = grad.mag.get(n1x, n1y);
            let neighbor2 = grad.mag.get(n2x, n2y);
            if mag <= neighbor1 || mag <= neighbor2 {
                continue;
            }

            edges.push(EdgeElement {
                x: x as u32,
                y: y as u32,
                magnitude: mag,
                direction: angle,
            });
        }
    }

    edges
}
