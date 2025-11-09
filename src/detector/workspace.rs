//! Per-level detector workspace used to cache gradients.
//!
//! The detector reuses gradient buffers across frames to avoid repeated
//! allocations in hot paths. Cache entries are computed on demand.
use crate::edges::grad::{scharr_gradients, Grad, GradientLevel};
use crate::image::ImageF32;

/// Workspace storing per-level gradient buffers to avoid repeated allocations.
#[derive(Default)]
pub struct DetectorWorkspace {
    gradients: Vec<Option<Grad>>,
}

impl DetectorWorkspace {
    pub fn new() -> Self {
        Self {
            gradients: Vec::new(),
        }
    }

    /// Clears cached data and prepares space for `levels` entries.
    pub fn reset(&mut self, levels: usize) {
        if self.gradients.len() < levels {
            self.gradients.resize_with(levels, || None);
        } else {
            for entry in &mut self.gradients {
                *entry = None;
            }
        }
    }

    /// Returns Scharr gradients for the requested level, computing them on demand.
    pub fn scharr_gradients(&mut self, level_idx: usize, level: &ImageF32) -> &Grad {
        if level_idx >= self.gradients.len() {
            self.gradients.resize_with(level_idx + 1, || None);
        }
        if self.gradients[level_idx].is_none() {
            self.gradients[level_idx] = Some(scharr_gradients(level));
        }
        self.gradients[level_idx]
            .as_ref()
            .expect("gradient cache populated")
    }

    /// Provides a lightweight gradient view for refinement without cloning buffers.
    pub fn gradient_level(&mut self, level_idx: usize, level: &ImageF32) -> GradientLevel<'_> {
        let grad = self.scharr_gradients(level_idx, level);
        GradientLevel::from_grad(grad)
    }
}
