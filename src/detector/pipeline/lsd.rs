use crate::diagnostics::builders::compute_family_counts;
use crate::diagnostics::LsdStage;
use crate::lsd_vp::{DetailedInference, Engine as LsdVpEngine};
use crate::pyramid::Pyramid;
use crate::segments::Segment;
use log::debug;
use nalgebra::Matrix3;
use std::time::Instant;

#[derive(Debug)]
pub struct LsdComputation {
    pub stage: Option<LsdStage>,
    pub segments: Vec<Segment>,
    pub coarse_h: Option<Matrix3<f32>>,
    pub full_h: Option<Matrix3<f32>>,
    pub confidence: f32,
    pub elapsed_ms: f64,
}

pub fn run_on_coarsest(
    engine: &LsdVpEngine,
    pyramid: &Pyramid,
    width: usize,
    height: usize,
) -> LsdComputation {
    let Some(coarse_level) = pyramid.levels.last() else {
        debug!("GridDetector::process pyramid has no levels");
        return LsdComputation {
            stage: None,
            segments: Vec::new(),
            coarse_h: None,
            full_h: None,
            confidence: 0.0,
            elapsed_ms: 0.0,
        };
    };

    let lsd_start = Instant::now();
    match engine.infer(coarse_level) {
        Some(DetailedInference {
            hypothesis,
            dominant_angles_rad,
            families,
            segments,
        }) => {
            let scale_x = if coarse_level.w > 0 {
                width as f32 / coarse_level.w as f32
            } else {
                1.0
            };
            let scale_y = if coarse_level.h > 0 {
                height as f32 / coarse_level.h as f32
            } else {
                1.0
            };
            let h_full = if coarse_level.w > 0 && coarse_level.h > 0 {
                hypothesis.scaled(scale_x, scale_y)
            } else {
                hypothesis.hmtx0
            };

            let dominant_angles_deg = [
                dominant_angles_rad[0].to_degrees(),
                dominant_angles_rad[1].to_degrees(),
            ];
            let family_counts = compute_family_counts(&families);
            let stage = LsdStage {
                elapsed_ms: 0.0,
                confidence: hypothesis.confidence,
                dominant_angles_deg,
                family_counts,
                segment_families: families,
                sample_ids: Vec::new(),
                used_gradient_refinement: false,
            };

            LsdComputation {
                stage: Some(stage),
                segments,
                coarse_h: Some(hypothesis.hmtx0),
                full_h: Some(h_full),
                confidence: hypothesis.confidence,
                elapsed_ms: lsd_start.elapsed().as_secs_f64() * 1000.0,
            }
        }
        None => {
            debug!("GridDetector::process LSD→VP engine returned no hypothesis");
            LsdComputation {
                stage: None,
                segments: Vec::new(),
                coarse_h: None,
                full_h: None,
                confidence: 0.0,
                elapsed_ms: lsd_start.elapsed().as_secs_f64() * 1000.0,
            }
        }
    }
}

pub fn refit_with_segments(
    engine: &LsdVpEngine,
    pyramid: &Pyramid,
    width: usize,
    height: usize,
    segments: Vec<Segment>,
) -> LsdComputation {
    let Some(coarse_level) = pyramid.levels.last() else {
        debug!("GridDetector::process pyramid has no levels");
        return LsdComputation {
            stage: None,
            segments: Vec::new(),
            coarse_h: None,
            full_h: None,
            confidence: 0.0,
            elapsed_ms: 0.0,
        };
    };

    let start = Instant::now();
    match engine.infer_with_segments(coarse_level, segments) {
        Some(DetailedInference {
            hypothesis,
            dominant_angles_rad,
            families,
            segments,
        }) => {
            let scale_x = if coarse_level.w > 0 {
                width as f32 / coarse_level.w as f32
            } else {
                1.0
            };
            let scale_y = if coarse_level.h > 0 {
                height as f32 / coarse_level.h as f32
            } else {
                1.0
            };
            let h_full = if coarse_level.w > 0 && coarse_level.h > 0 {
                hypothesis.scaled(scale_x, scale_y)
            } else {
                hypothesis.hmtx0
            };

            let dominant_angles_deg = [
                dominant_angles_rad[0].to_degrees(),
                dominant_angles_rad[1].to_degrees(),
            ];
            let family_counts = compute_family_counts(&families);
            let mut stage = LsdStage {
                elapsed_ms: 0.0,
                confidence: hypothesis.confidence,
                dominant_angles_deg,
                family_counts,
                segment_families: families,
                sample_ids: Vec::new(),
                used_gradient_refinement: true,
            };
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            stage.elapsed_ms = elapsed_ms;

            LsdComputation {
                stage: Some(stage),
                segments,
                coarse_h: Some(hypothesis.hmtx0),
                full_h: Some(h_full),
                confidence: hypothesis.confidence,
                elapsed_ms,
            }
        }
        None => {
            debug!("GridDetector::process VP refit returned no hypothesis");
            LsdComputation {
                stage: None,
                segments: Vec::new(),
                coarse_h: None,
                full_h: None,
                confidence: 0.0,
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            }
        }
    }
}
