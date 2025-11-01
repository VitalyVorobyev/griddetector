use grid_detector::config::lsd_vp;
use grid_detector::config::segments::LsdConfig;
use grid_detector::detector::params::LsdVpParams;
use grid_detector::diagnostics::builders::run_lsd_stage;
use grid_detector::diagnostics::{LsdStage, PyramidStage};
use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::lsd_vp::Engine;
use grid_detector::pyramid::{Pyramid, PyramidOptions};
use grid_detector::segments::{lsd_extract_segments_with_options, LsdOptions, Segment};
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

    let lsd_options: LsdOptions = config.lsd.to_lsd_options();
    let angle_tol_rad = config.lsd.angle_tolerance_deg.to_radians();
    let initial_segments = lsd_extract_segments_with_options(
        coarsest,
        config.lsd.magnitude_threshold,
        angle_tol_rad,
        config.lsd.min_length,
        lsd_options,
    );

    let engine_params = config.lsd.to_lsd_vp_params();
    let engine = Engine {
        mag_thresh: engine_params.mag_thresh,
        angle_tol_deg: engine_params.angle_tol_deg,
        min_len: engine_params.min_len,
        options: lsd_options,
    };
    let pyramid_stage = PyramidStage::from_pyramid(&pyramid, 0.0);

    let lsd_start = Instant::now();
    let lsd_output = run_lsd_stage(
        &engine,
        coarsest,
        Some(initial_segments.clone()),
        gray.width(),
        gray.height(),
    );
    let lsd_ms = lsd_start.elapsed().as_secs_f64() * 1000.0;

    let (lsd_stage, segments, coarse_h, full_h) = match lsd_output {
        Some(mut output) => {
            output.stage.elapsed_ms = lsd_ms;
            (
                Some(output.stage),
                output.segments,
                Some(output.coarse_h),
                Some(output.full_h),
            )
        }
        None => (None, initial_segments, None, None),
    };

    let result = LsdVpDemoOutput {
        image_width: gray.width(),
        image_height: gray.height(),
        pyramid: pyramid_stage,
        lsd: lsd_stage,
        lsd_config: LsdParamsOut::from(&config.lsd),
        engine_config: EngineParamsOut::from(&engine_params),
        coarse_h,
        full_h,
        segments,
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
    coarse_h: Option<Matrix3<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    full_h: Option<Matrix3<f32>>,
    segments: Vec<Segment>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LsdParamsOut {
    magnitude_threshold: f32,
    angle_tolerance_deg: f32,
    min_length: f32,
    enforce_polarity: bool,
    normal_span_limit_px: Option<f32>,
}

impl From<&LsdConfig> for LsdParamsOut {
    fn from(cfg: &LsdConfig) -> Self {
        Self {
            magnitude_threshold: cfg.magnitude_threshold,
            angle_tolerance_deg: cfg.angle_tolerance_deg,
            min_length: cfg.min_length,
            enforce_polarity: cfg.enforce_polarity,
            normal_span_limit_px: cfg.normal_span_limit_px,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EngineParamsOut {
    magnitude_threshold: f32,
    angle_tolerance_deg: f32,
    min_length: f32,
    enforce_polarity: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    normal_span_limit_px: Option<f32>,
}

impl From<&LsdVpParams> for EngineParamsOut {
    fn from(params: &LsdVpParams) -> Self {
        Self {
            magnitude_threshold: params.mag_thresh,
            angle_tolerance_deg: params.angle_tol_deg,
            min_length: params.min_len,
            enforce_polarity: params.enforce_polarity,
            normal_span_limit_px: params.normal_span_limit,
        }
    }
}

// matrix_to_array removed: use nalgebra serde directly for Matrix3.
