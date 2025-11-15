//! Per-level detector workspace used to cache gradients.
//!
//! The detector reuses gradient buffers across frames to avoid repeated
//! allocations in hot paths. Cache entries are computed on demand.
//!
//! # Gradient tiles and refinement
//!
//! Gradient-driven refinement needs fast access to `gx, gy` at arbitrary
//! subpixel locations. To keep the core refinement loop simple and cache
//! friendly, this workspace exposes gradients as **tiles**:
//!
//! - [`scharr_gradients_full`] returns a single full-frame tile for a pyramid
//!   level. The tile covers the entire level and is reused across segments.
//! - [`scharr_gradients_window`] computes a cropped tile for an
//!   axis-aligned window in level coordinates, writing the result into a
//!   reusable buffer. This is used by the segment refiner to restrict work to
//!   the union of all per-segment ROIs when beneficial.
//!
//! Both functions return a lightweight [`GradientTileView`] that carries:
//! - The tile origin `(origin_x, origin_y)` in full-resolution coordinates.
//! - Tile dimensions `(tile_width, tile_height)`.
//! - Contiguous slices `gx`, `gy` storing horizontal/vertical derivatives
//!   in row-major order.
//!
//! The refinement module wraps this view into [`crate::refine::segment::types::PyramidLevel`]
//! and uses bilinear interpolation in [`crate::refine::segment::sampling::bilinear_grad`]
//! to sample gradients at subpixel positions. The global level width/height
//! are preserved in `PyramidLevel` so ROI logic can still work in full-image
//! coordinates while gradient lookups operate inside the cropped tile.
use crate::edges::grad::scharr_gradients_window_into;
use crate::image::ImageF32;
use crate::refine::segment::roi::IntBounds;
#[cfg(feature = "profile_refine")]
use std::time::Instant;

/// Workspace storing per-level gradient buffers to avoid repeated allocations.
#[derive(Default)]
pub struct DetectorWorkspace {
    full_tiles: Vec<Option<GradientPatch>>,
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

impl GradientPatch {
    fn new() -> Self {
        Self {
            origin_x: 0,
            origin_y: 0,
            tile_width: 0,
            tile_height: 0,
            gx: Vec::new(),
            gy: Vec::new(),
        }
    }

    fn ensure_size(&mut self, width: usize, height: usize) {
        if self.tile_width == width && self.tile_height == height {
            return;
        }
        self.tile_width = width;
        self.tile_height = height;
        let len = width * height;
        self.gx.resize(len, 0.0);
        self.gy.resize(len, 0.0);
    }
}

impl DetectorWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears cached data and prepares space for `levels` entries.
    pub fn reset(&mut self, levels: usize) {
        if self.full_tiles.len() < levels {
            self.full_tiles.resize_with(levels, || None);
        } else {
            for entry in &mut self.full_tiles {
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

    /// Returns a full-frame gradient tile for the requested level.
    pub fn scharr_gradients_full(
        &mut self,
        level_idx: usize,
        level: &ImageF32,
    ) -> GradientTileView<'_> {
        if level_idx >= self.full_tiles.len() {
            self.full_tiles.resize_with(level_idx + 1, || None);
        }
        let tile = self.full_tiles[level_idx].get_or_insert_with(GradientPatch::new);
        tile.ensure_size(level.w, level.h);
        tile.origin_x = 0;
        tile.origin_y = 0;
        #[cfg(feature = "profile_refine")]
        let start = Instant::now();
        scharr_gradients_window_into(level, 0, 0, level.w, level.h, &mut tile.gx, &mut tile.gy);
        #[cfg(feature = "profile_refine")]
        {
            let elapsed = start.elapsed();
            if self.grad_timings_ms.len() <= level_idx {
                self.grad_timings_ms.resize(level_idx + 1, 0.0);
            }
            self.grad_timings_ms[level_idx] = elapsed.as_secs_f64() * 1000.0;
        }
        GradientTileView {
            origin_x: 0,
            origin_y: 0,
            tile_width: level.w,
            tile_height: level.h,
            gx: &tile.gx,
            gy: &tile.gy,
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
        let patch = self.roi_gradients[level_idx].get_or_insert_with(GradientPatch::new);
        let width = window.width();
        let height = window.height();
        patch.ensure_size(width, height);
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
