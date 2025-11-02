use grid_detector::config::lsd_vp;
use grid_detector::diagnostics::builders::run_lsd_stage;
use grid_detector::diagnostics::{LsdStage, PyramidStage};
use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::lsd_vp::Engine;
use grid_detector::pyramid::{Pyramid, PyramidOptions};
use grid_detector::segments::{lsd_extract_segments, LsdOptions, Segment};
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
    let pyramid_opts = PyramidOptions::new(levels).with_blur_levels(config.pyramid.blur_levels);
    let pyramid = Pyramid::build_u8(gray.as_view(), pyramid_opts);
    let coarsest_index = pyramid
        .levels
        .len()
        .checked_sub(1)
        .ok_or("Pyramid has no levels")?;
    let coarsest = &pyramid.levels[coarsest_index];

    save_grayscale_f32(coarsest, &config.output.coarsest_image)?;

    let initial_segments = lsd_extract_segments(coarsest, config.lsd);

    let engine = Engine {
        options: config.lsd,
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
        lsd_config: config.lsd,
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
    lsd_config: LsdOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    coarse_h: Option<Matrix3<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    full_h: Option<Matrix3<f32>>,
    segments: Vec<Segment>,
}
