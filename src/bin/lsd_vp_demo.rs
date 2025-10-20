use grid_detector::config::lsd_vp::{self, EngineParameters};
use grid_detector::config::segments::LsdConfig;
use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::lsd_vp::{Engine, FamilyLabel};
use grid_detector::pyramid::Pyramid;
use grid_detector::segments::{lsd_extract_segments_with_options, LsdOptions, Segment};
use serde::Serialize;
use std::env;
use std::path::Path;

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config_path = env::args().nth(1).ok_or_else(usage)?;
    let config = lsd_vp::load_config(Path::new(&config_path))?;

    let gray = load_grayscale_image(&config.input)?;
    let levels = config.pyramid.levels.max(1);
    let pyramid = if let Some(blur_levels_cfg) = config.pyramid.blur_levels {
        let blur_levels = blur_levels_cfg.min(levels.saturating_sub(1));
        Pyramid::build_u8_with_blur_levels(gray.as_view(), levels, blur_levels)
    } else {
        Pyramid::build_u8(gray.as_view(), levels)
    };
    let coarsest_index = pyramid
        .levels
        .len()
        .checked_sub(1)
        .ok_or("Pyramid has no levels")?;
    let coarsest = &pyramid.levels[coarsest_index];

    save_grayscale_f32(coarsest, &config.output.coarsest_image)?;

    let lsd_options = LsdOptions {
        enforce_polarity: config.lsd.enforce_polarity,
        normal_span_limit: if config.lsd.limit_normal_span {
            Some(config.lsd.normal_span_limit_px)
        } else {
            None
        },
    };
    let angle_tol_rad = config.lsd.angle_tolerance_deg.to_radians();
    let segments = lsd_extract_segments_with_options(
        coarsest,
        config.lsd.magnitude_threshold,
        angle_tol_rad,
        config.lsd.min_length,
        lsd_options,
    );

    let engine_params = config.engine.resolve(&config.lsd);
    let engine = Engine {
        mag_thresh: engine_params.magnitude_threshold,
        angle_tol_deg: engine_params.angle_tolerance_deg,
        min_len: engine_params.min_length,
    };
    let inference = engine.infer_with_segments(coarsest, segments.clone());

    let (hypothesis, dominant_angles_deg, family_counts, segment_outputs) =
        if let Some(detail) = inference {
            let mut fam_u = 0usize;
            let mut fam_v = 0usize;
            let DetailedInferenceFields {
                hypothesis,
                dominant_angles_deg,
                segments,
                families,
            } = DetailedInferenceFields::from(detail);
            let segment_outputs = segments
                .into_iter()
                .zip(families.into_iter())
                .map(|(seg, fam)| {
                    if let Some(FamilyLabel::U) = fam {
                        fam_u += 1;
                    } else if let Some(FamilyLabel::V) = fam {
                        fam_v += 1;
                    }
                    SegmentClusterOutput::from_segment(seg, fam)
                })
                .collect::<Vec<_>>();
            let unassigned = segment_outputs.len().saturating_sub(fam_u + fam_v);
            (
                Some(hypothesis),
                Some(dominant_angles_deg),
                Some(FamilyCounts {
                    family_u: fam_u,
                    family_v: fam_v,
                    unassigned,
                }),
                segment_outputs,
            )
        } else {
            let segment_outputs = segments
                .into_iter()
                .map(|seg| SegmentClusterOutput::from_segment(seg, None))
                .collect::<Vec<_>>();
            (None, None, None, segment_outputs)
        };

    let result = LsdVpDemoOutput {
        width: coarsest.w,
        height: coarsest.h,
        pyramid_level_index: coarsest_index,
        pyramid_levels: pyramid.levels.len(),
        lsd_config: LsdParamsOut::from(&config.lsd),
        engine_config: EngineParamsOut::from(engine_params),
        segment_count: segment_outputs.len(),
        family_counts,
        dominant_angles_deg,
        hypothesis,
        segments: segment_outputs,
    };

    write_json_file(&config.output.result_json, &result)?;

    println!(
        "Saved coarsest level image to {} (level {})",
        config.output.coarsest_image.display(),
        coarsest_index
    );
    println!(
        "Saved LSD-VP result with {} segments to {}",
        result.segment_count,
        config.output.result_json.display()
    );

    Ok(())
}

fn usage() -> String {
    "Usage: lsd_vp_demo <config.json>".to_string()
}

#[derive(Debug)]
struct DetailedInferenceFields {
    hypothesis: grid_detector::lsd_vp::Hypothesis,
    dominant_angles_deg: [f32; 2],
    segments: Vec<Segment>,
    families: Vec<Option<FamilyLabel>>,
}

impl From<grid_detector::lsd_vp::DetailedInference> for DetailedInferenceFields {
    fn from(detail: grid_detector::lsd_vp::DetailedInference) -> Self {
        let grid_detector::lsd_vp::DetailedInference {
            hypothesis,
            dominant_angles_rad,
            families,
            segments,
        } = detail;
        DetailedInferenceFields {
            hypothesis,
            dominant_angles_deg: dominant_angles_rad.map(|a| a.to_degrees()),
            families,
            segments,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LsdVpDemoOutput {
    width: usize,
    height: usize,
    pyramid_level_index: usize,
    pyramid_levels: usize,
    lsd_config: LsdParamsOut,
    engine_config: EngineParamsOut,
    segment_count: usize,
    family_counts: Option<FamilyCounts>,
    dominant_angles_deg: Option<[f32; 2]>,
    hypothesis: Option<grid_detector::lsd_vp::Hypothesis>,
    segments: Vec<SegmentClusterOutput>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LsdParamsOut {
    magnitude_threshold: f32,
    angle_tolerance_deg: f32,
    min_length: f32,
    enforce_polarity: bool,
    limit_normal_span: bool,
    normal_span_limit_px: Option<f32>,
}

impl From<&LsdConfig> for LsdParamsOut {
    fn from(cfg: &LsdConfig) -> Self {
        Self {
            magnitude_threshold: cfg.magnitude_threshold,
            angle_tolerance_deg: cfg.angle_tolerance_deg,
            min_length: cfg.min_length,
            enforce_polarity: cfg.enforce_polarity,
            limit_normal_span: cfg.limit_normal_span,
            normal_span_limit_px: if cfg.limit_normal_span {
                Some(cfg.normal_span_limit_px)
            } else {
                None
            },
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EngineParamsOut {
    magnitude_threshold: f32,
    angle_tolerance_deg: f32,
    min_length: f32,
}

impl From<EngineParameters> for EngineParamsOut {
    fn from(params: EngineParameters) -> Self {
        Self {
            magnitude_threshold: params.magnitude_threshold,
            angle_tolerance_deg: params.angle_tolerance_deg,
            min_length: params.min_length,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct FamilyCounts {
    family_u: usize,
    family_v: usize,
    unassigned: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SegmentClusterOutput {
    p0: [f32; 2],
    p1: [f32; 2],
    direction: [f32; 2],
    length: f32,
    line: [f32; 3],
    average_magnitude: f32,
    strength: f32,
    family: Option<FamilyLabel>,
}

impl SegmentClusterOutput {
    fn from_segment(seg: Segment, family: Option<FamilyLabel>) -> Self {
        let Segment {
            p0,
            p1,
            dir,
            len,
            line,
            avg_mag,
            strength,
        } = seg;
        Self {
            p0,
            p1,
            direction: dir,
            length: len,
            line,
            average_magnitude: avg_mag,
            strength,
            family,
        }
    }
}
