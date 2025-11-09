use super::region_accumulator::RegionAccumulator;
use super::types::{LsdOptions, Segment, SegmentId};
use crate::angle::{angular_difference, normalize_half_pi};
use crate::edges::{sobel_gradients, Grad};
use crate::image::ImageF32;
use nalgebra::{Matrix2, SymmetricEigen};

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

pub(super) struct LsdExtractor<'a> {
    grad: Grad,
    width: usize,
    height: usize,
    options: LsdOptions,
    used: Vec<u8>,
    angle_cache: Vec<f32>,
    stack: Vec<usize>,
    region: RegionAccumulator,
    segments: Vec<Segment>,
    mask: Option<&'a [u8]>,
    enforce_polarity: bool,
    normal_span_limit: Option<f32>,
    next_id: u32,
}

impl<'a> LsdExtractor<'a> {
    pub(super) fn new(l: &ImageF32, options: LsdOptions, mask: Option<&'a [u8]>) -> Self {
        let grad = sobel_gradients(l);
        let width = l.w;
        let height = l.h;
        let n = width * height;
        if cfg!(debug_assertions) {
            if let Some(m) = mask {
                debug_assert!(
                    m.len() >= n,
                    "mask length {} must be at least width*height ({})",
                    m.len(),
                    n
                );
            }
        }
        Self {
            grad,
            width,
            height,
            options,
            used: vec![0u8; n],
            angle_cache: vec![f32::NAN; n],
            stack: Vec::with_capacity(64),
            region: RegionAccumulator::with_capacity(128),
            segments: Vec::new(),
            mask,
            enforce_polarity: options.enforce_polarity,
            normal_span_limit: options
                .normal_span_limit_px
                .filter(|v| v.is_finite() && *v > 0.0),
            next_id: 0,
        }
    }

    pub(super) fn extract(mut self) -> Vec<Segment> {
        for idx in 0..(self.width * self.height) {
            self.process_seed(idx);
        }
        self.segments
    }

    fn process_seed(&mut self, idx: usize) {
        if self.used[idx] != 0 {
            return;
        }
        if let Some(mask) = self.mask {
            if mask[idx] == 0 {
                return;
            }
        }
        let x = idx % self.width;
        let y = idx / self.width;
        if self.grad.mag.get(x, y) < self.options.magnitude_threshold {
            return;
        }

        self.region.reset();
        self.stack.clear();

        let seed_angle = self.angle_at(idx);
        self.used[idx] = 1;
        self.stack.push(idx);

        self.grow_region(seed_angle);

        if let Some(segment) = self.build_segment() {
            self.segments.push(segment);
        } else {
            self.region.release(&mut self.used);
        }
        self.region.reset();
    }

    fn grow_region(&mut self, seed_angle: f32) {
        let angle_tol_rad = self.options.angle_tolerance_deg.to_radians();
        while let Some(idx) = self.stack.pop() {
            let x = idx % self.width;
            let y = idx / self.width;
            let angle = self.angle_at(idx);
            let aligned = self.angle_difference(angle, seed_angle) <= angle_tol_rad;
            let mag = self.grad.mag.get(x, y);
            self.region.push(idx, x, y, mag, aligned);

            for (dx, dy) in NEIGH_OFFSETS {
                let xn = x as isize + dx;
                let yn = y as isize + dy;
                if xn < 0 || yn < 0 || xn >= self.width as isize || yn >= self.height as isize {
                    continue;
                }
                let neighbor_idx = yn as usize * self.width + xn as usize;
                if let Some(mask) = self.mask {
                    if mask[neighbor_idx] == 0 {
                        continue;
                    }
                }
                if self.used[neighbor_idx] != 0 {
                    continue;
                }
                let nx = xn as usize;
                let ny = yn as usize;
                if self.grad.mag.get(nx, ny) < self.options.magnitude_threshold {
                    continue;
                }
                let neighbor_angle = self.angle_at(neighbor_idx);
                if self.angle_difference(neighbor_angle, seed_angle) <= angle_tol_rad {
                    self.used[neighbor_idx] = 1;
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
            if self.normal_span_limit.is_some() {
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

        if self.region.aligned_fraction() < 0.6 {
            return None;
        }

        if let Some(limit) = self.normal_span_limit {
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

    fn angle_at(&mut self, idx: usize) -> f32 {
        let cached = self.angle_cache[idx];
        if cached.is_nan() {
            let x = idx % self.width;
            let y = idx / self.width;
            let raw_angle = self.grad.gy.get(x, y).atan2(self.grad.gx.get(x, y));
            let angle = if self.enforce_polarity {
                normalize_signed_pi(raw_angle)
            } else {
                normalize_half_pi(raw_angle)
            };
            self.angle_cache[idx] = angle;
            angle
        } else {
            cached
        }
    }

    fn angle_difference(&self, a: f32, b: f32) -> f32 {
        if self.enforce_polarity {
            let mut diff = (a - b).abs();
            if diff > std::f32::consts::PI {
                diff = 2.0 * std::f32::consts::PI - diff;
            }
            diff
        } else {
            angular_difference(a, b)
        }
    }
}

#[inline]
fn normalize_signed_pi(angle: f32) -> f32 {
    let mut norm = angle.rem_euclid(2.0 * std::f32::consts::PI);
    if norm > std::f32::consts::PI {
        norm -= 2.0 * std::f32::consts::PI;
    }
    norm
}
