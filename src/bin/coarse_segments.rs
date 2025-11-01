use grid_detector::config::segments;
use grid_detector::diagnostics::pyramid::PyramidStage;
use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::pyramid::{Pyramid, PyramidOptions};
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
    let config = segments::load_config(Path::new(&config_path))?;

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
    let pyramid_stage = PyramidStage::from_pyramid(&pyramid, 0.0);

    let angle_tol_rad = config.lsd.angle_tolerance_deg.to_radians();
    let options = LsdOptions {
        enforce_polarity: config.lsd.enforce_polarity,
        normal_span_limit: if config.lsd.limit_normal_span {
            Some(config.lsd.normal_span_limit_px)
        } else {
            None
        },
    };
    let segments = lsd_extract_segments_with_options(
        coarsest,
        config.lsd.magnitude_threshold,
        angle_tol_rad,
        config.lsd.min_length,
        options,
    );

    let summary = CoarseSegmentsReport {
        pyramid: pyramid_stage,
        segments,
    };

    save_grayscale_f32(coarsest, &config.output.coarsest_image)?;
    write_json_file(&config.output.segments_json, &summary)?;

    println!(
        "Saved coarsest level image to {} (level {})",
        config.output.coarsest_image.display(),
        coarsest_index
    );
    println!(
        "Saved {} line segments to {}",
        summary.segments.len(),
        config.output.segments_json.display()
    );

    Ok(())
}

fn usage() -> String {
    "Usage: coarse_segments <config.json>".to_string()
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CoarseSegmentsReport {
    pyramid: PyramidStage,
    segments: Vec<Segment>,
}
