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
    let grad = sobel_gradients(l);
    let (w, h) = (l.w, l.h);
    let n = w * h;
    let mut used = vec![0u8; n];
    let mut angle_cache = vec![f32::NAN; n];
    let mut segs: Vec<Segment> = Vec::new();
    let mut stack: Vec<usize> = Vec::with_capacity(64);
    let mut region = RegionAccumulator::with_capacity(128);
    let half_angle_tol = angle_tol * 0.5;

    let mut angle_at = |idx: usize| -> f32 {
        let cached = angle_cache[idx];
        if cached.is_nan() {
            let x = idx % w;
            let y = idx / w;
            let angle = normalize_half_pi(grad.gy.get(x, y).atan2(grad.gx.get(x, y)));
            angle_cache[idx] = angle;
            angle
        } else {
            cached
        }
    };

    for y0 in 0..h {
        for x0 in 0..w {
            let idx0 = y0 * w + x0;
            if used[idx0] != 0 {
                continue;
            }
            if grad.mag.get(x0, y0) < mag_thresh {
                continue;
            }

            region.reset();
            stack.clear();

            let seed_angle = angle_at(idx0);
            used[idx0] = 1;
            stack.push(idx0);

            while let Some(idx) = stack.pop() {
                let x = idx % w;
                let y = idx / w;
                let angle = angle_at(idx);
                let aligned = angular_difference(angle, seed_angle) <= half_angle_tol;
                region.push(idx, x, y, aligned);

                for (dx, dy) in NEIGH_OFFSETS {
                    let xn = x as isize + dx;
                    let yn = y as isize + dy;
                    if xn < 0 || yn < 0 || xn >= w as isize || yn >= h as isize {
                        continue;
                    }
                    let xu = xn as usize;
                    let yu = yn as usize;
                    let neighbor_idx = yu * w + xu;
                    if used[neighbor_idx] != 0 {
                        continue;
                    }
                    if grad.mag.get(xu, yu) < mag_thresh {
                        continue;
                    }
                    let neighbor_angle = angle_at(neighbor_idx);
                    if angular_difference(neighbor_angle, seed_angle) <= angle_tol {
                        used[neighbor_idx] = 1;
                        stack.push(neighbor_idx);
                    }
                }
            }

            let maybe_segment = (|| {
                if region.len() < 12 {
                    return None;
                }

                let count = region.len() as f32;
                let cx = region.sum_x / count;
                let cy = region.sum_y / count;
                if !cx.is_finite() || !cy.is_finite() {
                    return None;
                }

                let cxx = region.sum_xx / count - cx * cx;
                let cyy = region.sum_yy / count - cy * cy;
                let cxy = region.sum_xy / count - cx * cy;
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
                for &idx in &region.indices {
                    let x = (idx % w) as f32;
                    let y = (idx / w) as f32;
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
                if !len.is_finite() || len <= 0.0 || len < min_len {
                    return None;
                }

                if region.aligned_fraction() < 0.6 {
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
            })();

            match maybe_segment {
                Some(segment) => segs.push(segment),
                None => region.release(&mut used),
            }
            region.reset();
        }
    }

    segs
}

/// (Optional) Older tensor-tiling extractor kept as a fallback
pub fn extract_segments_tensor(
    l: &ImageF32,
    grad: &Grad,
    tile: usize,
    mag_thresh: f32,
    aniso_thresh: f32,
    min_len: f32,
) -> Vec<Segment> {
    let mut segs: Vec<Segment> = Vec::new();
    let w = l.w;
    let h = l.h;

    let tile = tile.max(4);
    let mut y0 = 0usize;
    while y0 < h {
        let mut x0 = 0usize;
        while x0 < w {
            let x1 = (x0 + tile).min(w);
            let y1 = (y0 + tile).min(h);
            // Accumulate tensor and centroid
            let mut sxx = 0.0f32;
            let mut syy = 0.0f32;
            let mut sxy = 0.0f32;
            let mut sw = 0.0f32;
            let mut sx = 0.0f32;
            let mut sy = 0.0f32;
            for y in y0..y1 {
                for x in x0..x1 {
                    let gx = grad.gx.get(x, y);
                    let gy = grad.gy.get(x, y);
                    let m = grad.mag.get(x, y);
                    sxx += gx * gx;
                    syy += gy * gy;
                    sxy += gx * gy;
                    let w = m.max(0.0);
                    sw += w;
                    sx += w * x as f32;
                    sy += w * y as f32;
                }
            }
            if sw > 0.0 {
                // Structure tensor eigen
                let J = Matrix2::new(sxx, sxy, sxy, syy);
                let eig = SymmetricEigen::new(J);
                let l0 = eig.eigenvalues[0];
                let l1 = eig.eigenvalues[1];
                let (lmax, lmin, vmax, vmin) = if l0 >= l1 {
                    (
                        l0,
                        l1,
                        eig.eigenvectors.column(0),
                        eig.eigenvectors.column(1),
                    )
                } else {
                    (
                        l1,
                        l0,
                        eig.eigenvectors.column(1),
                        eig.eigenvectors.column(0),
                    )
                };
                let ratio = (lmax + 1e-6) / (lmin + 1e-6);
                if lmax > 1e-4 && ratio >= aniso_thresh {
                    let cx = sx / sw;
                    let cy = sy / sw;
                    // Tangent direction ~ eigenvector of min eigen (along-edge)
                    let tx = vmin[0] as f32;
                    let ty = vmin[1] as f32;
                    let norm = (tx * tx + ty * ty).sqrt().max(1e-6);
                    let tx = tx / norm;
                    let ty = ty / norm;
                    // Collect supporting pixels and compute extent along t
                    let mut smin = f32::INFINITY;
                    let mut smax = f32::NEG_INFINITY;
                    let mut align_sum = 0.0f32;
                    let mut count = 0u32;
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let m = grad.mag.get(x, y);
                            if m >= mag_thresh {
                                let gx = grad.gx.get(x, y);
                                let gy = grad.gy.get(x, y);
                                let gn = (gx * gx + gy * gy).sqrt().max(1e-6);
                                // normal direction (dominant gradient) is vmax
                                let nx = vmax[0] as f32;
                                let ny = vmax[1] as f32;
                                let nn = (nx * nx + ny * ny).sqrt().max(1e-6);
                                let nx = nx / nn;
                                let ny = ny / nn;
                                let gnx = gx / gn;
                                let gny = gy / gn;
                                align_sum += (gnx * nx + gny * ny).abs();
                                let dx = x as f32 - cx;
                                let dy = y as f32 - cy;
                                let s = dx * tx + dy * ty; // projection onto tangent
                                if s < smin {
                                    smin = s;
                                }
                                if s > smax {
                                    smax = s;
                                }
                                count += 1;
                            }
                        }
                    }
                    let len = smax - smin;
                    if count >= 10 && len >= min_len {
                        let mean_align = align_sum / (count as f32);
                        if mean_align >= 0.7 {
                            let p0 = [cx + smin * tx, cy + smin * ty];
                            let p1 = [cx + smax * tx, cy + smax * ty];
                            // line normal from vmax; normalize
                            let mut nx = vmax[0] as f32;
                            let mut ny = vmax[1] as f32;
                            let nrm = (nx * nx + ny * ny).sqrt().max(1e-6);
                            nx /= nrm;
                            ny /= nrm;
                            let c = -(nx * cx + ny * cy);
                            let seg = Segment {
                                p0,
                                p1,
                                dir: [tx, ty],
                                len,
                                line: [nx, ny, c],
                            };
                            segs.push(seg);
                        }
                    }
                }
            }
            x0 += tile;
        }
        y0 += tile;
    }

    segs
}
