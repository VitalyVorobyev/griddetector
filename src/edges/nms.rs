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
use crate::edges::grad::{image_gradients, GradientKernel, Grad};
use crate::image::ImageF32;
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

pub fn run_nms(grad: &Grad, mag_thresh: f32) -> Vec<EdgeElement> {
    let w = grad.gx.w;
    let h = grad.gx.h;
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

            let (n1x, n1y, n2x, n2y) = if !(22.5..157.5).contains(&angle_deg) {
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

pub struct NmsEdgesResult {
    pub edges: Vec<EdgeElement>,
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
    
    NmsEdgesResult { edges, gradient_ms, nms_ms }
}
