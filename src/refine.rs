//! Coarse-to-fine homography refinement using bundled edges and Huber loss.
//!
//! This module refines a coarse projective basis `H0` (from the LSD→VP engine)
//! into a more stable homography by:
//!
//! 1) Extracting LSD-like segments per pyramid level and bundling collinear,
//!    nearby segments into weighted line constraints. Each bundle stores a
//!    normalized line `ax+by+c=0`, a representative center, and a weight
//!    proportional to segment strength (length×avg gradient).
//! 2) Splitting bundles into two families (u/v) using directions implied by
//!    the current homography’s vanishing points relative to the translation
//!    anchor. The assignment is soft but biased by angular proximity.
//! 3) Re-estimating vanishing points with IRLS under a Huber loss: solve a
//!    weighted normal equation on line constraints (ax+by+c≈0), iteratively
//!    down-weighting outliers by `delta/|residual|`.
//! 4) Updating the anchor from the highest-weight pair of orthogonal bundles
//!    (their line intersection) as a stabilizer for translation.
//! 5) Proceeding from coarse → fine pyramid levels, carrying the homography
//!    forward and stopping early when the relative update is sufficiently
//!    small. Confidence accumulates across levels and a failure falls back to
//!    the last good homography.
//!
//! The refiner returns:
//! - `h_refined`: refined homography in image coordinates.
//! - `confidence`: aggregated confidence from IRLS inlier weights.
//! - `inlier_ratio`: per-level inlier ratio at the last refinement.
//! - `levels_used`: number of pyramid levels that contributed.
//!
//! Tuning
//! - `RefineParams` controls thresholds for bundling, Huber delta, maximum
//!   iterations, and minimal family support. For high-res images (e.g.,
//!   1280×1024), consider increasing `base_min_len` and relaxing
//!   `merge_dist_px` slightly if edges are aliased.
//!
//! Limitations
//! - Assumes rectified input (no lens distortion). For distorted inputs,
//!   either pre-rectify or integrate distortion into the projection model.
//! - The current energy uses line constraints; grid-spacing metric refinement
//!   (equal spacing) is intentionally deferred to a later milestone.
use crate::pyramid::Pyramid;
use crate::segments::{lsd_extract_segments, Segment};
use log::debug;
use nalgebra::{Matrix3, Vector3};

const EPS: f32 = 1e-6;

/// Parameters controlling coarse-to-fine homography refinement.
///
/// These parameters are level-aware where noted; the refiner adapts some
/// thresholds as it traverses from coarse → fine levels.
#[derive(Clone, Debug)]
pub struct RefineParams {
    /// Base gradient magnitude threshold at the coarsest level (0..1).
    /// This threshold is reduced slightly for coarser levels.
    pub base_mag_thresh: f32,
    /// Angular tolerance (degrees) used by the LSD-like extractor while
    /// growing regions at each level.
    pub angle_tol_deg: f32,
    /// Minimum accepted segment length (pixels) at the coarsest level.
    /// The refiner increases this threshold on finer levels.
    pub base_min_len: f32,
    /// Orientation tolerance (degrees) for bundling/assignment into the two
    /// line families (u/v) relative to the current VP directions.
    pub orientation_tol_deg: f32,
    /// Merge distance (pixels) between normalized lines when forming bundles;
    /// applied to the absolute difference of the line offsets (c terms).
    pub merge_dist_px: f32,
    /// Huber delta (in pixels) controlling the inlier region for the IRLS
    /// re-estimation of vanishing points.
    pub huber_delta: f32,
    /// Maximum IRLS iterations per pyramid level.
    pub max_iterations: usize,
    /// Minimum strength (length × average gradient magnitude) required for a
    /// bundle to participate in refinement.
    pub min_bundle_weight: f32,
    /// Minimum number of bundles per family (u and v) to attempt an update.
    pub min_bundles_per_family: usize,
}

impl Default for RefineParams {
    fn default() -> Self {
        Self {
            base_mag_thresh: 0.03,
            angle_tol_deg: 20.0,
            base_min_len: 6.0,
            orientation_tol_deg: 22.5,
            merge_dist_px: 1.5,
            huber_delta: 1.0,
            max_iterations: 6,
            min_bundle_weight: 3.0,
            min_bundles_per_family: 4,
        }
    }
}

/// Result of the coarse-to-fine refinement.
#[derive(Clone, Debug)]
pub struct RefinementResult {
    /// Refined homography in input image pixel coordinates.
    pub h_refined: Matrix3<f32>,
    /// Aggregated (0..1) confidence across levels based on inlier support.
    pub confidence: f32,
    /// Weighted inlier ratio from the last successful level (0..1).
    pub inlier_ratio: f32,
    /// Number of pyramid levels that contributed to refinement.
    pub levels_used: usize,
}

#[derive(Clone, Debug)]
struct Bundle {
    line: [f32; 3],
    center: [f32; 2],
    weight: f32,
}

/// Coarse-to-fine refiner that bundles collinear segments and re-estimates
/// vanishing points with a Huber-weighted IRLS at each pyramid level.
///
/// The refiner assumes rectified input (no lens distortion) and refines a
/// projective basis (not enforcing metric grid spacing yet).
pub struct Refiner {
    params: RefineParams,
}

impl Refiner {
    /// Creates a refiner with the provided parameters.
    pub fn new(params: RefineParams) -> Self {
        Self { params }
    }

    /// Refines an initial homography `initial_h` across the given pyramid.
    ///
    /// Expects `initial_h` to be expressed in the input image coordinates
    /// (i.e., already rescaled from the pyramid level it was estimated on).
    /// Returns `None` if there are insufficient constraints on all levels.
    pub fn refine(
        &self,
        pyr: &Pyramid,
        initial_h: Matrix3<f32>,
    ) -> Option<RefinementResult> {
        if pyr.levels.is_empty() {
            debug!("Refiner: empty pyramid");
            return None;
        }
        let mut current_h = initial_h;
        let mut last_good_h = initial_h;
        let mut last_inlier_ratio = 0.0f32;
        let mut total_levels = 0usize;
        let mut accumulated_conf = 0.0f32;
        let mut accumulated_weight = 0.0f32;

        let levels = pyr.levels.len();
        for (lvl_idx, img) in pyr.levels.iter().enumerate().rev() {
            total_levels += 1;
            let scale = 2.0f32.powi((levels - 1 - lvl_idx) as i32);
            let mag_thresh = (self.params.base_mag_thresh / scale.sqrt()).max(0.005);
            let min_len = (self.params.base_min_len * scale).max(4.0);
            let angle_tol = self.params.angle_tol_deg.to_radians();

            let segs = lsd_extract_segments(img, mag_thresh, angle_tol, min_len);
            debug!(
                "Refiner: level {} size {}x{} segs={}",
                lvl_idx,
                img.w,
                img.h,
                segs.len()
            );
            if segs.len() < 12 {
                continue;
            }
            let bundles = bundle_segments(
                &segs,
                self.params.orientation_tol_deg.to_radians(),
                self.params.merge_dist_px,
                self.params.min_bundle_weight,
            );
            debug!(
                "Refiner: level {} bundles={}",
                lvl_idx,
                bundles.len()
            );
            if bundles.len() < self.params.min_bundles_per_family * 2 {
                continue;
            }

            if let Some(level_res) = self.refine_level(&bundles, current_h) {
                let improvement = (frobenius_norm(&(level_res.h_new - current_h))
                    / (frobenius_norm(&current_h) + EPS))
                    .abs();
                debug!(
                    "Refiner: level {} improvement={:.4} confidence={:.3} inliers={:.3}",
                    lvl_idx, improvement, level_res.confidence, level_res.inlier_ratio
                );
                current_h = level_res.h_new;
                let weight = (lvl_idx + 1) as f32;
                accumulated_conf += level_res.confidence * weight;
                accumulated_weight += weight;
                last_inlier_ratio = level_res.inlier_ratio;
                last_good_h = current_h;
                if improvement < 1e-3 {
                    debug!("Refiner: early stop due to small improvement");
                    break;
                }
            } else {
                debug!("Refiner: level {} refinement failed, continuing", lvl_idx);
            }
        }

        if accumulated_weight <= EPS {
            None
        } else {
            Some(RefinementResult {
                h_refined: last_good_h,
                confidence: (accumulated_conf / accumulated_weight).clamp(0.0, 1.0),
                inlier_ratio: last_inlier_ratio,
                levels_used: total_levels,
            })
        }
    }

    fn refine_level(
        &self,
        bundles: &[Bundle],
        h_current: Matrix3<f32>,
    ) -> Option<LevelRefinement> {
        let vpu = h_current.column(0).into_owned();
        let vpv = h_current.column(1).into_owned();
        let anchor = h_current.column(2).into_owned();
        let dir_u = vp_direction(&vpu, &anchor)?;
        let dir_v = vp_direction(&vpv, &anchor)?;

        let orientation_tol = self.params.orientation_tol_deg.to_radians();
        let mut fam_u: Vec<&Bundle> = Vec::new();
        let mut fam_v: Vec<&Bundle> = Vec::new();
        for b in bundles {
            let tangent = bundle_tangent(b);
            let du = angle_between(&tangent, &dir_u);
            let dv = angle_between(&tangent, &dir_v);
            if du <= orientation_tol && du < dv {
                fam_u.push(b);
            } else if dv <= orientation_tol && dv < du {
                fam_v.push(b);
            } else if du < dv {
                fam_u.push(b);
            } else {
                fam_v.push(b);
            }
        }

        if fam_u.len() < self.params.min_bundles_per_family
            || fam_v.len() < self.params.min_bundles_per_family
        {
            debug!(
                "Refiner: not enough bundles per family (u={}, v={})",
                fam_u.len(),
                fam_v.len()
            );
            return None;
        }

        let delta = self.params.huber_delta;
        let (vpu_new, stats_u) = estimate_vp_huber(&fam_u, &vpu, delta, self.params.max_iterations)?;
        let (vpv_new, stats_v) = estimate_vp_huber(&fam_v, &vpv, delta, self.params.max_iterations)?;
        let anchor_new = estimate_anchor(&fam_u, &fam_v).unwrap_or(anchor);

        let h_new = Matrix3::from_columns(&[vpu_new, vpv_new, anchor_new]);
        let combined_inlier =
            (stats_u.inlier_weight + stats_v.inlier_weight) / (stats_u.total_weight + stats_v.total_weight + EPS);
        let confidence = ((stats_u.confidence + stats_v.confidence) * 0.5).clamp(0.0, 1.0);
        Some(LevelRefinement {
            h_new,
            confidence,
            inlier_ratio: combined_inlier,
        })
    }
}

struct LevelRefinement {
    h_new: Matrix3<f32>,
    confidence: f32,
    inlier_ratio: f32,
}

struct VpStats {
    confidence: f32,
    total_weight: f32,
    inlier_weight: f32,
}

fn bundle_segments(
    segs: &[Segment],
    orientation_tol: f32,
    dist_tol: f32,
    min_weight: f32,
) -> Vec<Bundle> {
    let mut bundles: Vec<Bundle> = Vec::new();
    for seg in segs {
        let weight = seg.strength;
        if weight < min_weight {
            continue;
        }
        let line = seg.line;
        let mut placed = false;
        for existing in bundles.iter_mut() {
            let dot = existing.line[0] * line[0] + existing.line[1] * line[1];
            let mut adj_line = line;
            if dot < 0.0 {
                adj_line[0] = -adj_line[0];
                adj_line[1] = -adj_line[1];
                adj_line[2] = -adj_line[2];
            }
            let dot_norm = (existing.line[0] * adj_line[0] + existing.line[1] * adj_line[1]).clamp(-1.0, 1.0);
            let angle = dot_norm.acos();
            let dist = (existing.line[2] - adj_line[2]).abs();
            if angle <= orientation_tol && dist <= dist_tol {
                merge_bundle(existing, &adj_line, seg, weight);
                placed = true;
                break;
            }
        }
        if !placed {
            bundles.push(Bundle {
                line,
                center: segment_center(seg),
                weight,
            });
        }
    }
    bundles
}

fn merge_bundle(target: &mut Bundle, line: &[f32; 3], seg: &Segment, weight: f32) {
    let total = target.weight + weight;
    if total <= EPS {
        return;
    }
    target.line[0] = (target.line[0] * target.weight + line[0] * weight) / total;
    target.line[1] = (target.line[1] * target.weight + line[1] * weight) / total;
    target.line[2] = (target.line[2] * target.weight + line[2] * weight) / total;
    let norm = (target.line[0] * target.line[0] + target.line[1] * target.line[1]).sqrt().max(EPS);
    target.line[0] /= norm;
    target.line[1] /= norm;
    target.line[2] /= norm;

    target.center[0] = (target.center[0] * target.weight + segment_center(seg)[0] * weight) / total;
    target.center[1] = (target.center[1] * target.weight + segment_center(seg)[1] * weight) / total;
    target.weight = total;
}

fn segment_center(seg: &Segment) -> [f32; 2] {
    [
        0.5 * (seg.p0[0] + seg.p1[0]),
        0.5 * (seg.p0[1] + seg.p1[1]),
    ]
}

fn bundle_tangent(bundle: &Bundle) -> [f32; 2] {
    [-bundle.line[1], bundle.line[0]]
}

fn angle_between(a: &[f32; 2], b: &[f32; 2]) -> f32 {
    let dot = a[0] * b[0] + a[1] * b[1];
    let na = (a[0] * a[0] + a[1] * a[1]).sqrt().max(EPS);
    let nb = (b[0] * b[0] + b[1] * b[1]).sqrt().max(EPS);
    (dot / (na * nb)).clamp(-1.0, 1.0).acos()
}

fn vp_direction(vp: &Vector3<f32>, anchor: &Vector3<f32>) -> Option<[f32; 2]> {
    if vp[2].abs() <= 1e-3 {
        let norm = (vp[0] * vp[0] + vp[1] * vp[1]).sqrt();
        if norm <= EPS {
            return None;
        }
        Some([vp[0] / norm, vp[1] / norm])
    } else {
        let vx = vp[0] / vp[2];
        let vy = vp[1] / vp[2];
        let ax = anchor[0] / anchor[2];
        let ay = anchor[1] / anchor[2];
        let dx = vx - ax;
        let dy = vy - ay;
        let norm = (dx * dx + dy * dy).sqrt();
        if norm <= EPS {
            None
        } else {
            Some([dx / norm, dy / norm])
        }
    }
}

fn huber_weight(residual: f32, delta: f32) -> (f32, bool) {
    let abs = residual.abs();
    if abs <= delta {
        (1.0, true)
    } else {
        (delta / abs, false)
    }
}

fn estimate_vp_huber(
    bundles: &[&Bundle],
    vp_init: &Vector3<f32>,
    delta: f32,
    max_iters: usize,
) -> Option<(Vector3<f32>, VpStats)> {
    let mut vp = *vp_init;
    if vp[2].abs() <= 1e-3 {
        vp[2] = 1.0;
    }
    let mut stats = VpStats {
        confidence: 0.0,
        total_weight: 0.0,
        inlier_weight: 0.0,
    };
    for _ in 0..max_iters {
        let mut a11 = 0.0f32;
        let mut a12 = 0.0f32;
        let mut a22 = 0.0f32;
        let mut bx = 0.0f32;
        let mut by = 0.0f32;
        let mut total_w = 0.0f32;
        let mut inlier_w = 0.0f32;
        for bundle in bundles {
            let line = bundle.line;
            let residual = line[0] * vp[0] + line[1] * vp[1] + line[2] * vp[2];
            let (h_weight, inlier) = huber_weight(residual, delta);
            let w = bundle.weight * h_weight;
            if w <= EPS {
                continue;
            }
            a11 += w * line[0] * line[0];
            a12 += w * line[0] * line[1];
            a22 += w * line[1] * line[1];
            bx += -w * line[2] * line[0];
            by += -w * line[2] * line[1];
            total_w += w;
            if inlier {
                inlier_w += bundle.weight;
            }
        }
        if total_w <= EPS {
            break;
        }
        let det = a11 * a22 - a12 * a12;
        if det.abs() <= EPS {
            return fallback_vp_from_bundles(bundles);
        }
        let inv11 = a22 / det;
        let inv12 = -a12 / det;
        let inv22 = a11 / det;
        let x = inv11 * bx + inv12 * by;
        let y = inv12 * bx + inv22 * by;
        let new_vp = Vector3::new(x, y, 1.0);
        if (new_vp - vp).norm() < 1e-3 {
            vp = new_vp;
            stats.total_weight = total_w;
            stats.inlier_weight = inlier_w;
            stats.confidence = (inlier_w / (bundles.len() as f32 + EPS)).clamp(0.0, 1.0);
            return Some((vp, stats));
        }
        vp = new_vp;
        stats.total_weight = total_w;
        stats.inlier_weight = inlier_w;
        stats.confidence = (inlier_w / (bundles.len() as f32 + EPS)).clamp(0.0, 1.0);
    }
    Some((vp, stats))
}

fn fallback_vp_from_bundles(bundles: &[&Bundle]) -> Option<(Vector3<f32>, VpStats)> {
    let mut sum_tx = 0.0f32;
    let mut sum_ty = 0.0f32;
    let mut total_w = 0.0f32;
    for bundle in bundles {
        let tangent = bundle_tangent(bundle);
        sum_tx += tangent[0] * bundle.weight;
        sum_ty += tangent[1] * bundle.weight;
        total_w += bundle.weight;
    }
    let norm = (sum_tx * sum_tx + sum_ty * sum_ty).sqrt();
    if norm <= EPS {
        return None;
    }
    let vp = Vector3::new(sum_tx / norm, sum_ty / norm, 0.0);
    Some((
        vp,
        VpStats {
            confidence: 0.0,
            total_weight: total_w,
            inlier_weight: total_w,
        },
    ))
}

fn estimate_anchor(fam_u: &[&Bundle], fam_v: &[&Bundle]) -> Option<Vector3<f32>> {
    let best_u = fam_u.iter().max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())?;
    let best_v = fam_v.iter().max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())?;
    let line_u = Vector3::new(best_u.line[0], best_u.line[1], best_u.line[2]);
    let line_v = Vector3::new(best_v.line[0], best_v.line[1], best_v.line[2]);
    let cross = line_u.cross(&line_v);
    if cross[2].abs() <= EPS {
        None
    } else {
        Some(cross / cross[2])
    }
}

fn frobenius_norm(m: &Matrix3<f32>) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            sum += m[(i, j)] * m[(i, j)];
        }
    }
    sum.sqrt()
}
