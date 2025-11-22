//! Edge processing utilities: image gradients and simple non‑maximum suppression.
//!
//! This module provides fast, minimal building blocks for edge‑based
//! processing used elsewhere in the detector:
//!
//! - Gradient computation (Sobel/Scharr) returning `gx`, `gy`, magnitude, and
//!   an 8‑bin quantized orientation.
//! - Lightweight non‑maximum suppression on the gradient magnitude with a
//!   direction‑aligned 4‑neighborhood, producing discrete edge elements.
//!
//! Design goals
//! - Favor clarity and cache‑friendly row access over micro‑optimizations.
//! - Handle borders by clamping indices (replicate).
//! - Keep outputs simple and serializable for tooling.
//!
//! See also: `doc/edges.md` for a deeper dive into algorithms and trade‑offs.

pub mod grad;
pub mod nms;

/// Per‑pixel gradients with magnitude and quantized orientation.
pub use grad::{scharr_gradients, sobel_gradients, Grad};
/// Simple Sobel NMS returning sparse edge elements suitable for visualization.
pub use nms::{detect_edges_nms, EdgeElement, NmsEdgesResult};
