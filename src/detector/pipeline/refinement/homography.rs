use crate::detector::params::RefinementSchedule;
use crate::refine::Refiner;
use nalgebra::Matrix3;

use super::prepare::PreparedLevels;

const EPS: f32 = 1e-6;

/// Result of running the homography refinement step.
#[derive(Debug)]
pub struct RefinementComputation {
    /// Refined homography (or the input if refinement did not run).
    pub hmtx: Matrix3<f32>,
    /// Updated confidence after combining with refinement statistics.
    pub confidence: f32,
    /// Optional low-level diagnostic describing the last refinement outcome.
    pub stage: Option<crate::diagnostics::RefinementStage>,
    /// Total time spent in refinement (ms).
    pub elapsed_ms: f64,
}

/// Run IRLS refinement across prepared levels and merge confidence.
///
/// - `initial_h` is the starting homography (full-res coordinates).
/// - `base_confidence` is the coarse hypothesis confidence before refinement.
/// - `schedule` controls number of passes and minimum improvement to continue.
/// - If `enable_refine` is false or inputs are degenerate, returns early.
pub fn refine_homography(
    refiner: &mut Refiner,
    prepared_levels: &PreparedLevels,
    initial_h: Option<Matrix3<f32>>,
    base_confidence: f32,
    schedule: &RefinementSchedule,
    enable_refine: bool,
    last_hypothesis: Option<Matrix3<f32>>,
) -> RefinementComputation {
    use std::time::Instant;

    let mut confidence = base_confidence;
    let mut hmtx = initial_h.unwrap_or_else(Matrix3::identity);

    if !enable_refine
        || initial_h.is_none()
        || hmtx == Matrix3::identity()
        || prepared_levels.is_empty()
    {
        return RefinementComputation {
            hmtx,
            confidence,
            stage: None,
            elapsed_ms: 0.0,
        };
    }

    let refine_levels = prepared_levels.as_refine_levels();
    let mut current_h = hmtx;
    let mut passes = 0usize;
    let mut refine_ms = 0.0f64;
    let mut last_outcome: Option<crate::diagnostics::RefinementOutcome> = None;
    let mut attempted = false;

    while passes < schedule.passes {
        attempted = true;
        let refine_start = Instant::now();
        match refiner.refine(current_h, &refine_levels) {
            Some(refine_res) => {
                refine_ms += refine_start.elapsed().as_secs_f64() * 1000.0;
                passes += 1;
                let improvement = frobenius_improvement(&current_h, &refine_res.h_refined);
                let outcome = crate::diagnostics::RefinementOutcome {
                    levels_used: refine_res.levels_used,
                    confidence: refine_res.confidence,
                    inlier_ratio: refine_res.inlier_ratio,
                    iterations: refine_res.level_reports.clone(),
                };
                last_outcome = Some(outcome);

                let base_conf = confidence;
                let combined =
                    combine_confidence(base_conf, refine_res.confidence, refine_res.inlier_ratio);
                confidence = combined.max(base_conf);
                current_h = refine_res.h_refined;
                hmtx = current_h;

                if passes >= schedule.passes || improvement < schedule.improvement_thresh {
                    break;
                }
            }
            None => {
                refine_ms += refine_start.elapsed().as_secs_f64() * 1000.0;
                if let Some(prev) = last_hypothesis {
                    log::debug!("GridDetector::process refine failed -> fallback to last_hmtx");
                    hmtx = prev;
                    confidence *= 0.5;
                } else {
                    log::debug!("GridDetector::process refine failed -> keeping coarse hypothesis");
                }
                break;
            }
        }
    }

    let stage = if attempted {
        Some(crate::diagnostics::RefinementStage {
            elapsed_ms: refine_ms,
            passes,
            outcome: last_outcome,
        })
    } else {
        None
    };

    RefinementComputation {
        hmtx,
        confidence,
        stage,
        elapsed_ms: refine_ms,
    }
}

fn combine_confidence(base: f32, refine_conf: f32, inlier_ratio: f32) -> f32 {
    if inlier_ratio <= 1e-6 {
        return base.clamp(0.0, 1.0);
    }
    let blended = 0.5 * base + 0.5 * refine_conf;
    (blended * inlier_ratio.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

fn frobenius_improvement(a: &Matrix3<f32>, b: &Matrix3<f32>) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            let diff = b[(i, j)] - a[(i, j)];
            sum += diff * diff;
        }
    }
    sum.sqrt() / (frobenius_norm(a) + EPS)
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
