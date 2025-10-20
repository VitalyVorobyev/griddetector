use super::types::{LsdOptions, Segment};
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

struct RegionAccumulator {
    indices: Vec<usize>,
    sum_x: f32,
    sum_y: f32,
    sum_xx: f32,
    sum_yy: f32,
    sum_xy: f32,
    aligned: usize,
    sum_mag: f32,
}

impl RegionAccumulator {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            indices: Vec::with_capacity(capacity),
            sum_x: 0.0,
            sum_y: 0.0,
            sum_xx: 0.0,
            sum_yy: 0.0,
            sum_xy: 0.0,
            aligned: 0,
            sum_mag: 0.0,
        }
    }

    fn reset(&mut self) {
        self.indices.clear();
        self.sum_x = 0.0;
        self.sum_y = 0.0;
        self.sum_xx = 0.0;
        self.sum_yy = 0.0;
        self.sum_xy = 0.0;
        self.aligned = 0;
        self.sum_mag = 0.0;
    }

    fn push(&mut self, idx: usize, x: usize, y: usize, mag: f32, aligned: bool) {
        self.indices.push(idx);
        let xf = x as f32;
        let yf = y as f32;
        self.sum_x += xf;
        self.sum_y += yf;
        self.sum_xx += xf * xf;
        self.sum_yy += yf * yf;
        self.sum_xy += xf * yf;
        if aligned {
            self.aligned += 1;
        }
        self.sum_mag += mag;
    }

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn aligned_fraction(&self) -> f32 {
        if self.indices.is_empty() {
            0.0
        } else {
            self.aligned as f32 / self.indices.len() as f32
        }
    }

    fn avg_mag(&self) -> f32 {
        if self.indices.is_empty() {
            0.0
        } else {
            self.sum_mag / self.indices.len() as f32
        }
    }

    fn release(&self, used: &mut [u8]) {
        for &idx in &self.indices {
            used[idx] = 0;
        }
    }
}

pub(super) struct LsdExtractor<'a> {
    grad: Grad,
    width: usize,
    height: usize,
    mag_thresh: f32,
    angle_tol: f32,
    half_angle_tol: f32,
    min_len: f32,
    used: Vec<u8>,
    angle_cache: Vec<f32>,
    stack: Vec<usize>,
    region: RegionAccumulator,
    segments: Vec<Segment>,
    mask: Option<&'a [u8]>,
    enforce_polarity: bool,
    normal_span_limit: Option<f32>,
}

impl<'a> LsdExtractor<'a> {
    pub(super) fn new(
        l: &ImageF32,
        mag_thresh: f32,
        angle_tol: f32,
        min_len: f32,
        mask: Option<&'a [u8]>,
        options: LsdOptions,
    ) -> Self {
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
            mag_thresh,
            angle_tol,
            half_angle_tol: angle_tol * 0.5,
            min_len,
            used: vec![0u8; n],
            angle_cache: vec![f32::NAN; n],
            stack: Vec::with_capacity(64),
            region: RegionAccumulator::with_capacity(128),
            segments: Vec::new(),
            mask,
            enforce_polarity: options.enforce_polarity,
            normal_span_limit: options
                .normal_span_limit
                .filter(|v| v.is_finite() && *v > 0.0),
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
        if self.grad.mag.get(x, y) < self.mag_thresh {
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
        while let Some(idx) = self.stack.pop() {
            let x = idx % self.width;
            let y = idx / self.width;
            let angle = self.angle_at(idx);
            let aligned = self.angle_difference(angle, seed_angle) <= self.half_angle_tol;
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
                if self.grad.mag.get(nx, ny) < self.mag_thresh {
                    continue;
                }
                let neighbor_angle = self.angle_at(neighbor_idx);
                if self.angle_difference(neighbor_angle, seed_angle) <= self.angle_tol {
                    self.used[neighbor_idx] = 1;
                    self.stack.push(neighbor_idx);
                }
            }
        }
    }

    fn build_segment(&self) -> Option<Segment> {
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
        if !len.is_finite() || len <= 0.0 || len < self.min_len {
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
        let c = -(nx * cx + ny * cy);
        let avg_mag = self.region.avg_mag();
        let strength = len * avg_mag.max(1e-3);

        Some(Segment {
            p0,
            p1,
            dir: [tx, ty],
            len,
            line: [nx, ny, c],
            avg_mag,
            strength,
        })
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
