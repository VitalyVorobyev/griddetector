use grid_detector::config::lsd_vp::{self, EngineParameters};
use grid_detector::config::segments::LsdConfig;
use grid_detector::diagnostics::builders::run_lsd_stage;
use grid_detector::diagnostics::{LsdStage, PyramidStage, SegmentDescriptor, SegmentId};
use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::lsd_vp::Engine;
use grid_detector::pyramid::{Pyramid, PyramidOptions};
use grid_detector::segments::{lsd_extract_segments_with_options, LsdOptions};
use nalgebra::Matrix3;
use serde::Serialize;
use std::env;
use std::path::Path;
use std::time::Instant;

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
    let pyramid_opts = if let Some(blur_levels_cfg) = config.pyramid.blur_levels {
        let blur_levels = blur_levels_cfg.min(levels.saturating_sub(1));
        PyramidOptions::new(levels).with_blur_levels(Some(blur_levels))
    } else {
        PyramidOptions::new(levels)
    };
    let pyramid = Pyramid::build_u8(gray.as_view(), pyramid_opts);
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
    let pyramid_stage = PyramidStage::from_pyramid(&pyramid, 0.0);

    let lsd_start = Instant::now();
    let lsd_output = run_lsd_stage(
        &engine,
        coarsest,
        Some(segments.clone()),
        gray.width(),
        gray.height(),
    );
    let lsd_ms = lsd_start.elapsed().as_secs_f64() * 1000.0;

    let (lsd_stage, descriptors, coarse_h, full_h) = match lsd_output {
        Some(mut output) => {
            output.stage.elapsed_ms = lsd_ms;
            (
                Some(output.stage),
                output.descriptors,
                Some(output.coarse_h),
                Some(output.full_h),
            )
        }
        None => (
            None,
            segments
                .iter()
                .enumerate()
                .map(|(idx, seg)| SegmentDescriptor::from_segment(SegmentId(idx as u32), seg))
                .collect(),
            None,
            None,
        ),
    };

    let result = LsdVpDemoOutput {
        image_width: gray.width(),
        image_height: gray.height(),
        pyramid: pyramid_stage,
        lsd: lsd_stage,
        lsd_config: LsdParamsOut::from(&config.lsd),
        engine_config: EngineParamsOut::from(engine_params),
        coarse_h: coarse_h.map(matrix_to_array),
        full_h: full_h.map(matrix_to_array),
        segments: descriptors,
    };

    write_json_file(&config.output.result_json, &result)?;

    println!(
        "Saved coarsest level image to {} (level {})",
        config.output.coarsest_image.display(),
        coarsest_index
    );
    println!(
        "Saved LSD-VP diagnostics to {} (segments={})",
        config.output.result_json.display(),
        result.segments.len()
    );

    Ok(())
}

fn usage() -> String {
    "Usage: lsd_vp_demo <config.json>".to_string()
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LsdVpDemoOutput {
    image_width: usize,
    image_height: usize,
    pyramid: PyramidStage,
    #[serde(skip_serializing_if = "Option::is_none")]
    lsd: Option<LsdStage>,
    lsd_config: LsdParamsOut,
    engine_config: EngineParamsOut,
    #[serde(skip_serializing_if = "Option::is_none")]
    coarse_h: Option<[[f32; 3]; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    full_h: Option<[[f32; 3]; 3]>,
    segments: Vec<SegmentDescriptor>,
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

fn matrix_to_array(m: Matrix3<f32>) -> [[f32; 3]; 3] {
    [
        [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
        [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
        [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
    ]
}
