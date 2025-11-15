//! Per-level detector workspace used to cache gradients.
//!
//! The detector reuses gradient buffers across frames to avoid repeated
//! allocations in hot paths. Cache entries are computed on demand.
use crate::edges::grad::{scharr_gradients, scharr_gradients_window_into, Grad};
use crate::image::{traits::ImageView, ImageF32};
use crate::refine::segment::roi::IntBounds;
use std::time::Instant;

/// Workspace storing per-level gradient buffers to avoid repeated allocations.
pub struct DetectorWorkspace {
    gradients: Vec<Option<Grad>>,
    roi_gradients: Vec<Option<GradientPatch>>,
    #[cfg(feature = "profile_refine")]
    grad_timings_ms: Vec<f64>,
}

pub struct GradientTileView<'a> {
    pub origin_x: usize,
    pub origin_y: usize,
    pub tile_width: usize,
    pub tile_height: usize,
    pub gx: &'a [f32],
    pub gy: &'a [f32],
}

struct GradientPatch {
    origin_x: usize,
    origin_y: usize,
    tile_width: usize,
    tile_height: usize,
    gx: Vec<f32>,
    gy: Vec<f32>,
}

impl DetectorWorkspace {
    pub fn new() -> Self {
        Self::default()
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
        if self.roi_gradients.len() < levels {
            self.roi_gradients.resize_with(levels, || None);
        } else {
            for entry in &mut self.roi_gradients {
                *entry = None;
            }
        }
        #[cfg(feature = "profile_refine")]
        {
            if self.grad_timings_ms.len() < levels {
                self.grad_timings_ms.resize(levels, 0.0);
            }
            for t in &mut self.grad_timings_ms {
                *t = 0.0;
            }
        }
    }

    fn ensure_gradients(&mut self, level_idx: usize, level: &ImageF32) -> &Grad {
        if level_idx >= self.gradients.len() {
            self.gradients.resize_with(level_idx + 1, || None);
        }
        if self.gradients[level_idx].is_none() {
            let start = Instant::now();
            self.gradients[level_idx] = Some(scharr_gradients(level));
            let elapsed = start.elapsed();
            let elapsed_ms = elapsed.as_millis();
            #[cfg(feature = "profile_refine")]
            {
                if self.grad_timings_ms.len() <= level_idx {
                    self.grad_timings_ms.resize(level_idx + 1, 0.0);
                }
                self.grad_timings_ms[level_idx] = elapsed.as_secs_f64() * 1000.0;
            }
            println!(
                "Computed Scharr gradients for level {} in {} ms",
                level_idx, elapsed_ms
            );
        }
        self.gradients[level_idx]
            .as_ref()
            .expect("gradient cache populated")
    }

    /// Returns a full-frame gradient tile for the requested level.
    pub fn scharr_gradients_full(
        &mut self,
        level_idx: usize,
        level: &ImageF32,
    ) -> GradientTileView<'_> {
        let grad = self.ensure_gradients(level_idx, level);
        let gx = grad
            .gx
            .as_slice()
            .expect("workspace gradients must be contiguous");
        let gy = grad
            .gy
            .as_slice()
            .expect("workspace gradients must be contiguous");
        GradientTileView {
            origin_x: 0,
            origin_y: 0,
            tile_width: level.w,
            tile_height: level.h,
            gx,
            gy,
        }
    }

    /// Returns a cropped gradient tile computed on-demand for the requested window.
    pub fn scharr_gradients_window(
        &mut self,
        level_idx: usize,
        level: &ImageF32,
        window: &IntBounds,
    ) -> GradientTileView<'_> {
        if level_idx >= self.roi_gradients.len() {
            self.roi_gradients.resize_with(level_idx + 1, || None);
        }
        let patch = self.roi_gradients[level_idx].get_or_insert_with(|| GradientPatch {
            origin_x: 0,
            origin_y: 0,
            tile_width: 0,
            tile_height: 0,
            gx: Vec::new(),
            gy: Vec::new(),
        });
        let width = window.width();
        let height = window.height();
        if patch.tile_width != width || patch.tile_height != height {
            patch.gx.resize(width * height, 0.0);
            patch.gy.resize(width * height, 0.0);
            patch.tile_width = width;
            patch.tile_height = height;
        }
        patch.origin_x = window.x0;
        patch.origin_y = window.y0;
        #[cfg(feature = "profile_refine")]
        let start = Instant::now();
        scharr_gradients_window_into(
            level,
            window.x0,
            window.y0,
            width,
            height,
            &mut patch.gx[..],
            &mut patch.gy[..],
        );
        #[cfg(feature = "profile_refine")]
        {
            let elapsed = start.elapsed();
            if self.grad_timings_ms.len() <= level_idx {
                self.grad_timings_ms.resize(level_idx + 1, 0.0);
            }
            self.grad_timings_ms[level_idx] = elapsed.as_secs_f64() * 1000.0;
        }
        GradientTileView {
            origin_x: patch.origin_x,
            origin_y: patch.origin_y,
            tile_width: patch.tile_width,
            tile_height: patch.tile_height,
            gx: &patch.gx,
            gy: &patch.gy,
        }
    }

    #[cfg(feature = "profile_refine")]
    pub fn gradient_time_ms(&self, level_idx: usize) -> Option<f64> {
        self.grad_timings_ms.get(level_idx).copied()
    }
}

impl Default for DetectorWorkspace {
    fn default() -> Self {
        Self {
            gradients: Vec::new(),
            roi_gradients: Vec::new(),
            #[cfg(feature = "profile_refine")]
            grad_timings_ms: Vec::new(),
        }
    }
}
