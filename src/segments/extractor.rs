use super::options::LsdOptions;
use super::region_accumulator::RegionAccumulator;
use super::segment::{Segment, SegmentId};
use crate::edges::nms::EdgeElement;
use crate::edges::{scharr_gradients, Grad};
use crate::image::ImageF32;
use nalgebra::{Matrix2, SymmetricEigen};
use serde::Serialize;
use std::time::Instant;

/// Pixel visitation states:
/// - UNSEEN: never processed
/// - CLAIMED: currently part of an accepted region
/// - RETRY1/RETRY2: rejected once/twice, allow up to two re-seeds
/// - DISCARDED: rejected thrice (or deemed noise), never revisit
const STATE_UNSEEN: u8 = 0;
const STATE_CLAIMED: u8 = 1;
const STATE_RETRY1: u8 = 2;
const STATE_RETRY2: u8 = 3;
const STATE_DISCARDED: u8 = 4;

const NEIGH_OFFSETS: [(isize, isize); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

#[derive(Default, Serialize, Debug, Clone)]
pub struct LsdResult {
    pub segments: Vec<Segment>,
    #[serde(skip)]
    pub grad: Grad,
    pub elapsed_ms: f64,
}

pub(super) struct LsdExtractor {
    grad: Grad,
    width: usize,
    height: usize,
    options: LsdOptions,
    used: Vec<u8>,
    bins: Vec<u8>,
    stack: Vec<usize>,
    region: RegionAccumulator,
    segments: Vec<Segment>,
    next_id: u32,
    cos_tol: f32,
}

impl LsdExtractor {
    /// Build a new extractor for a single image. This keeps allocations local to the
    /// extractor instance; reuse an extractor if you need zero-allocation across frames.
    pub(super) fn from_image(l: &ImageF32, options: LsdOptions) -> Self {
        let grad = scharr_gradients(l);
        Self::from_grad(grad, options)
    }

    /// Build an extractor reusing precomputed gradients (e.g. from NMS).
    pub(super) fn from_grad(grad: Grad, options: LsdOptions) -> Self {
        let width = grad.gx.w;
        let height = grad.gx.h;
        let n = width * height;
        let cos_tol = options.angle_tolerance_deg.to_radians().cos();
        Self {
            grad,
            width,
            height,
            options,
            used: vec![STATE_UNSEEN; n],
            bins: vec![0u8; n],
            stack: Vec::with_capacity(64),
            region: RegionAccumulator::with_capacity(128),
            segments: Vec::new(),
            next_id: 0,
            cos_tol,
        }
    }

    pub(super) fn extract_full_scan(mut self) -> LsdResult {
        let start = Instant::now();
        for idx in 0..(self.width * self.height) {
            self.process_seed_full(idx);
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        LsdResult {
            segments: self.segments,
            grad: self.grad,
            elapsed_ms,
        }
    }

    pub(super) fn extract_from_seeds(mut self, seeds: &[Seed]) -> LsdResult {
        let start = Instant::now();
        for seed in seeds {
            if seed.idx >= self.used.len() {
                continue;
            }
            self.process_seed_with_dir(seed.idx, seed.dir, seed.bin_bit);
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        LsdResult {
            segments: self.segments,
            grad: self.grad,
            elapsed_ms,
        }
    }

    fn process_seed_full(&mut self, idx: usize) {
        let (dir, bin_bit) = match self.seed_direction_and_bin(idx) {
            Some(v) => v,
            None => return,
        };
        self.process_seed_with_dir(idx, dir, bin_bit);
    }

    fn process_seed_with_dir(&mut self, idx: usize, seed_dir: [f32; 2], bin_bit: u8) {
        let state = self.used[idx];
        if state == STATE_CLAIMED || state == STATE_DISCARDED {
            return;
        }
        if self.bins[idx] & bin_bit != 0 {
            return;
        }
        let x = idx % self.width;
        let y = idx / self.width;
        if self.grad.mag.get(x, y) < self.options.magnitude_threshold {
            return;
        }

        self.region.reset();
        self.stack.clear();

        self.used[idx] = STATE_CLAIMED;
        self.stack.push(idx);

        self.grow_region(seed_dir);

        if let Some(segment) = self.build_segment() {
            self.segments.push(segment);
        } else {
            // Record this bin attempt for all pixels in the region.
            self.region.mark_mask(&mut self.bins, bin_bit);
            let new_state = match state {
                STATE_RETRY2 => STATE_DISCARDED,
                STATE_RETRY1 => STATE_RETRY2,
                _ => STATE_RETRY1,
            };
            self.region.mark_as(&mut self.used, new_state);
        }
        self.region.reset();
    }

    fn grow_region(&mut self, seed_dir: [f32; 2]) {
        let cos_tol = self.cos_tol;
        let mag_thresh = self.options.magnitude_threshold;
        while let Some(idx) = self.stack.pop() {
            let x = idx % self.width;
            let y = idx / self.width;
            let mag = self.grad.mag.get(x, y);
            let aligned = self.is_aligned(seed_dir, x, y, mag, cos_tol);
            self.region.push(idx, x, y, mag, aligned);

            for (dx, dy) in NEIGH_OFFSETS {
                let xn = x as isize + dx;
                let yn = y as isize + dy;
                if xn < 0 || yn < 0 || xn >= self.width as isize || yn >= self.height as isize {
                    continue;
                }
                let neighbor_idx = yn as usize * self.width + xn as usize;
                if self.used[neighbor_idx] == STATE_CLAIMED
                    || self.used[neighbor_idx] == STATE_DISCARDED
                {
                    continue;
                }
                let nx = xn as usize;
                let ny = yn as usize;
                let nmag = self.grad.mag.get(nx, ny);
                if nmag < mag_thresh {
                    continue;
                }
                if self.is_aligned(seed_dir, nx, ny, nmag, cos_tol) {
                    self.used[neighbor_idx] = STATE_CLAIMED;
                    self.stack.push(neighbor_idx);
                }
            }
        }
    }

    fn build_segment(&mut self) -> Option<Segment> {
        if self.region.len() < 12 {
            return None;
        }

        let count = self.region.len() as f32;
        let cx = self.region.sum_x / count;
        let cy = self.region.sum_y / count;
        if !cx.is_finite() || !cy.is_finite() {
            return None;
        }

        let cxx = self.region.sum_xx / count - cx * cx;
        let cyy = self.region.sum_yy / count - cy * cy;
        let cxy = self.region.sum_xy / count - cx * cy;
        let cov = Matrix2::new(cxx, cxy, cxy, cyy);
        let eig = SymmetricEigen::new(cov);
        let (vmax, lambda_max) = if eig.eigenvalues[0] >= eig.eigenvalues[1] {
            (eig.eigenvectors.column(0), eig.eigenvalues[0])
        } else {
            (eig.eigenvectors.column(1), eig.eigenvalues[1])
        };
        if !lambda_max.is_finite() || lambda_max <= 0.0 {
            return None;
        }

        let mut tx = vmax[0];
        let mut ty = vmax[1];
        let norm = (tx * tx + ty * ty).sqrt();
        if !norm.is_finite() || norm < 1e-6 {
            return None;
        }
        tx /= norm;
        ty /= norm;

        let mut smin = f32::INFINITY;
        let mut smax = f32::NEG_INFINITY;
        let nx = -ty;
        let ny = tx;
        let mut nmin = f32::INFINITY;
        let mut nmax = f32::NEG_INFINITY;
        for &idx in &self.region.indices {
            let x = (idx % self.width) as f32;
            let y = (idx / self.width) as f32;
            let dx = x - cx;
            let dy = y - cy;
            let s = dx * tx + dy * ty;
            if s < smin {
                smin = s;
            }
            if s > smax {
                smax = s;
            }
            if self.options.normal_span_limit_px.is_some() {
                let n = dx * nx + dy * ny;
                if n < nmin {
                    nmin = n;
                }
                if n > nmax {
                    nmax = n;
                }
            }
        }

        if !smin.is_finite() || !smax.is_finite() {
            return None;
        }

        let len = smax - smin;
        if !len.is_finite() || len <= 0.0 || len < self.options.min_length_px {
            return None;
        }

        if self.region.aligned_fraction() < self.options.min_aligned_fraction {
            return None;
        }

        if let Some(limit) = self.options.normal_span_limit_px {
            if nmin.is_finite() && nmax.is_finite() {
                let normal_span = nmax - nmin;
                if !normal_span.is_finite() || normal_span > limit {
                    return None;
                }
            }
        }

        let p0 = [cx + smin * tx, cy + smin * ty];
        let p1 = [cx + smax * tx, cy + smax * ty];
        let avg_mag = self.region.avg_mag();
        let strength = len * avg_mag.max(1e-3);

        let id = SegmentId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        Some(Segment::new(id, p0, p1, avg_mag, strength))
    }

    #[inline]
    fn seed_direction_and_bin(&self, idx: usize) -> Option<([f32; 2], u8)> {
        let x = idx % self.width;
        let y = idx / self.width;
        let gx = self.grad.gx.get(x, y);
        let gy = self.grad.gy.get(x, y);
        let norm = (gx * gx + gy * gy).sqrt();
        if norm <= 0.0 || !norm.is_finite() {
            None
        } else {
            let dir = [gx / norm, gy / norm];
            let bin = direction_bin(dir[0], dir[1]);
            Some((dir, 1u8 << bin))
        }
    }

    #[inline]
    fn is_aligned(
        &self,
        seed_dir: [f32; 2],
        x: usize,
        y: usize,
        mag: f32,
        cos_tol: f32,
    ) -> bool {
        let gx = self.grad.gx.get(x, y);
        let gy = self.grad.gy.get(x, y);
        // dot == |g| * |seed| * cos(theta); |seed| is 1.
        let dot = gx * seed_dir[0] + gy * seed_dir[1];
        if self.options.enforce_polarity {
            dot >= mag * cos_tol
        } else {
            dot.abs() >= mag * cos_tol
        }
    }
}

/// Quantize a unit direction vector into 8 bins over [0, π), modulo polarity.
#[inline]
fn direction_bin(dx: f32, dy: f32) -> u8 {
    // Map angle to [0, π) by folding polarity.
    let angle = dy.atan2(dx).rem_euclid(std::f32::consts::PI);
    let scaled = angle * (8.0 / std::f32::consts::PI);
    let bin = scaled.floor() as i32;
    bin.clamp(0, 7) as u8
}

#[derive(Clone)]
pub(super) struct Seed {
    pub idx: usize,
    pub dir: [f32; 2],
    pub bin_bit: u8,
}

/// Build seeds from NMS edges using their direction for binning.
pub(super) fn seeds_from_edges(edges: &[EdgeElement], width: usize) -> Vec<Seed> {
    let mut seeds = Vec::with_capacity(edges.len());
    for e in edges {
        let x = e.x as usize;
        let y = e.y as usize;
        let dx = e.direction.cos();
        let dy = e.direction.sin();
        let bin_bit = 1u8 << direction_bin(dx, dy);
        seeds.push(Seed {
            idx: y * width + x,
            dir: [dx, dy],
            bin_bit,
        });
    }
    seeds
}
