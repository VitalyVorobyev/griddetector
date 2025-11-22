//! Non‑maximum suppression on gradient magnitude with direction alignment.
//!
//! This module performs a Canny‑style, simplified non‑maximum suppression (NMS)
//! using Sobel gradients to estimate local edge direction. For each pixel, it
//! suppresses responses that are not strictly greater than their two neighbors
//! along the quantized gradient direction and emits a sparse list of edge
//! elements.
//!
//! Border handling uses clamping in gradient computation and ignores the outer
//!most 1‑pixel frame in NMS to avoid out‑of‑bounds checks in neighbor lookup.
use crate::edges::grad::{image_gradients, Grad, GradientKernel};
use crate::image::{ImageF32, ImageView};
use serde::Serialize;
use std::time::Instant;

/// A sparse edge sample after NMS suitable for visualization or simple post‑processing.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EdgeElement {
    /// X coordinate in pixels
    pub x: u32,
    /// Y coordinate in pixels
    pub y: u32,
    /// Gradient magnitude at (x, y)
    pub magnitude: f32,
    /// Gradient direction in radians, range (-π, π]
    pub direction: f32,
}

const TAN_22_5_DEG: f32 = 0.41421356237;

pub fn run_nms(grad: &Grad, mag_thresh: f32) -> Vec<EdgeElement> {
    let w = grad.gx.w;
    let h = grad.gx.h;
    if w < 3 || h < 3 {
        return Vec::new();
    }

    let inner_pixels = (w - 2) * (h - 2);
    let mut edges = Vec::with_capacity(inner_pixels / 8 + 1);
    for y in 1..h - 1 {
        let mag_prev = grad.mag.row(y - 1);
        let mag_row = grad.mag.row(y);
        let mag_next = grad.mag.row(y + 1);
        let gx_row = grad.gx.row(y);
        let gy_row = grad.gy.row(y);

        for x in 1..w - 1 {
            let mag = mag_row[x];
            if mag < mag_thresh {
                continue;
            }

            let gx = gx_row[x];
            let gy = gy_row[x];
            let abs_gx = gx.abs();
            let abs_gy = gy.abs();
            let same_sign = (gx >= 0.0 && gy >= 0.0) || (gx <= 0.0 && gy <= 0.0);

            let (neighbor1, neighbor2) = if abs_gx >= abs_gy {
                if abs_gy <= abs_gx * TAN_22_5_DEG {
                    (mag_row[x - 1], mag_row[x + 1])
                } else if same_sign {
                    (mag_prev[x + 1], mag_next[x - 1])
                } else {
                    (mag_prev[x - 1], mag_next[x + 1])
                }
            } else if abs_gx <= abs_gy * TAN_22_5_DEG {
                (mag_prev[x], mag_next[x])
            } else if same_sign {
                (mag_prev[x + 1], mag_next[x - 1])
            } else {
                (mag_prev[x - 1], mag_next[x + 1])
            };

            if mag <= neighbor1 || mag <= neighbor2 {
                continue;
            }

            let angle = gy.atan2(gx);
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

pub struct NmsEdgesResult {
    pub edges: Vec<EdgeElement>,
    pub grad: Grad,
    pub gradient_ms: f64,
    pub nms_ms: f64,
}

/// Simple Sobel-based edge detector with 4-neighborhood non-maximum suppression.
/// Detect edges by applying Sobel gradients followed by 4‑direction NMS.
///
/// - Direction quantization uses 4 bins (0°, 45°, 90°, 135°) to select the
///   two comparison neighbors.
/// - A pixel is kept if its magnitude is strictly greater than both neighbors
///   along that direction and above `mag_thresh`.
///
/// Returns a vector of `EdgeElement` containing position, magnitude, and
/// continuous gradient direction for visualization.
pub fn detect_edges_nms(l: &ImageF32, mag_thresh: f32) -> NmsEdgesResult {
    let gradient_start = Instant::now();
    let grad = image_gradients(l, GradientKernel::Scharr);
    let gradient_ms = gradient_start.elapsed().as_secs_f64() * 1000.0;

    let nms_start = Instant::now();
    let edges = run_nms(&grad, mag_thresh);
    let nms_ms = nms_start.elapsed().as_secs_f64() * 1000.0;

    NmsEdgesResult {
        edges,
        grad,
        gradient_ms,
        nms_ms,
    }
}
