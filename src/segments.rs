use crate::edges::{sobel_gradients, Grad};
use crate::types::ImageF32;
use nalgebra::{Matrix2, SymmetricEigen};

#[derive(Clone, Debug)]
pub struct Segment {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub dir: [f32; 2],
    pub len: f32,
    pub line: [f32; 3], // ax + by + c = 0, with sqrt(a^2+b^2)=1
}

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

#[inline]
fn normalize_half_pi(angle: f32) -> f32 {
    let norm = angle.rem_euclid(std::f32::consts::PI);
    if norm >= std::f32::consts::PI {
        norm - std::f32::consts::PI
    } else {
        norm
    }
}

#[inline]
fn angular_difference(a: f32, b: f32) -> f32 {
    let mut diff = (a - b).abs();
    if diff > std::f32::consts::PI {
        diff = diff.rem_euclid(std::f32::consts::PI);
    }
    if diff > std::f32::consts::PI * 0.5 {
        std::f32::consts::PI - diff
    } else {
        diff
    }
}

struct RegionAccumulator {
    indices: Vec<usize>,
    sum_x: f32,
    sum_y: f32,
    sum_xx: f32,
    sum_yy: f32,
    sum_xy: f32,
    aligned: usize,
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
    }

    fn push(&mut self, idx: usize, x: usize, y: usize, aligned: bool) {
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

    fn release(&self, used: &mut [u8]) {
        for &idx in &self.indices {
            used[idx] = 0;
        }
    }
}

/// Lightweight LSD-like extractor (region growing on gradient orientation, PCA fit, simple significance test)
pub fn lsd_extract_segments(
    l: &ImageF32,
    mag_thresh: f32, // min gradient magnitude (0..1 scale at this pyramid level)
    angle_tol: f32,  // radians, tolerance around seed normal angle
    min_len: f32,    // min length in pixels at this level
) -> Vec<Segment> {
    LsdExtractor::new(l, mag_thresh, angle_tol, min_len).extract()
}

struct LsdExtractor {
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
}

impl LsdExtractor {
    fn new(l: &ImageF32, mag_thresh: f32, angle_tol: f32, min_len: f32) -> Self {
        let grad = sobel_gradients(l);
        let width = l.w;
        let height = l.h;
        let n = width * height;
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
        }
    }

    fn extract(mut self) -> Vec<Segment> {
        for idx in 0..(self.width * self.height) {
            self.process_seed(idx);
        }
        self.segments
    }

    fn process_seed(&mut self, idx: usize) {
        if self.used[idx] != 0 {
            return;
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
            let aligned = angular_difference(angle, seed_angle) <= self.half_angle_tol;
            self.region.push(idx, x, y, aligned);

            for (dx, dy) in NEIGH_OFFSETS {
                let xn = x as isize + dx;
                let yn = y as isize + dy;
                if xn < 0 || yn < 0 || xn >= self.width as isize || yn >= self.height as isize {
                    continue;
                }
                let neighbor_idx = yn as usize * self.width + xn as usize;
                if self.used[neighbor_idx] != 0 {
                    continue;
                }
                let nx = xn as usize;
                let ny = yn as usize;
                if self.grad.mag.get(nx, ny) < self.mag_thresh {
                    continue;
                }
                let neighbor_angle = self.angle_at(neighbor_idx);
                if angular_difference(neighbor_angle, seed_angle) <= self.angle_tol {
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

        let p0 = [cx + smin * tx, cy + smin * ty];
        let p1 = [cx + smax * tx, cy + smax * ty];
        let nx = -ty;
        let ny = tx;
        let c = -(nx * cx + ny * cy);

        Some(Segment {
            p0,
            p1,
            dir: [tx, ty],
            len,
            line: [nx, ny, c],
        })
    }

    fn angle_at(&mut self, idx: usize) -> f32 {
        let cached = self.angle_cache[idx];
        if cached.is_nan() {
            let x = idx % self.width;
            let y = idx / self.width;
            let angle = normalize_half_pi(self.grad.gy.get(x, y).atan2(self.grad.gx.get(x, y)));
            self.angle_cache[idx] = angle;
            angle
        } else {
            cached
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step_image(width: usize, height: usize, split_x: usize) -> ImageF32 {
        let mut img = ImageF32::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let v = if x < split_x { 0.0 } else { 1.0 };
                img.set(x, y, v);
            }
        }
        img
    }

    #[test]
    fn lsd_extractor_finds_vertical_segment() {
        let img = step_image(32, 32, 16);
        let segs = LsdExtractor::new(&img, 0.1, std::f32::consts::FRAC_PI_6, 4.0).extract();
        assert!(
            !segs.is_empty(),
            "expected at least one segment on a vertical edge"
        );
        let longest = segs
            .iter()
            .max_by(|a, b| a.len.partial_cmp(&b.len).unwrap())
            .unwrap();
        assert!(
            longest.dir[1].abs() > longest.dir[0].abs(),
            "expected vertical-oriented segment, got dir={:?}",
            longest.dir
        );
        assert!(
            longest.len >= 8.0,
            "expected a reasonably long segment, got len={}",
            longest.len
        );
    }

    #[test]
    fn lsd_extractor_rejects_flat_image() {
        let img = ImageF32::new(16, 16);
        let segs = LsdExtractor::new(&img, 0.05, std::f32::consts::FRAC_PI_4, 2.0).extract();
        assert!(
            segs.is_empty(),
            "no segments should be detected in a flat image, got {:?}",
            segs
        );
    }
}
