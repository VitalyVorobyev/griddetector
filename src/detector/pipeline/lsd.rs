use crate::detector::params::LsdVpParams;
use crate::diagnostics::{FamilyCounts, LsdStage};
use crate::lsd_vp::{DetailedInference, Engine as LsdVpEngine, FamilyLabel};
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

pub fn make_engine(params: &LsdVpParams) -> LsdVpEngine {
    LsdVpEngine {
        mag_thresh: params.mag_thresh,
        angle_tol_deg: params.angle_tol_deg,
        min_len: params.min_len,
        options: params.to_lsd_options(),
    }
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
    match engine.infer_detailed(coarse_level) {
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
            let scale = Matrix3::new(scale_x, 0.0, 0.0, 0.0, scale_y, 0.0, 0.0, 0.0, 1.0);
            let h_full = if coarse_level.w > 0 && coarse_level.h > 0 {
                scale * hypothesis.hmtx0
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

fn compute_family_counts(families: &[Option<FamilyLabel>]) -> FamilyCounts {
    let mut counts = FamilyCounts {
        family_u: 0,
        family_v: 0,
        unassigned: 0,
    };
    for fam in families {
        match fam {
            Some(FamilyLabel::U) => counts.family_u += 1,
            Some(FamilyLabel::V) => counts.family_v += 1,
            None => counts.unassigned += 1,
        }
    }
    counts
}
