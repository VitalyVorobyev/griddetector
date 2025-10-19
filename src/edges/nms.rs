use crate::edges::grad::sobel_gradients;
use crate::types::ImageF32;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EdgeElement {
    pub x: u32,
    pub y: u32,
    pub magnitude: f32,
    /// Gradient direction in radians, range (-π, π]
    pub direction: f32,
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
