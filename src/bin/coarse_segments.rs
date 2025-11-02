use grid_detector::config::segments;
use grid_detector::diagnostics::pyramid::PyramidStage;
use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::pyramid::{Pyramid, PyramidOptions};
use grid_detector::segments::{lsd_extract_segments, Segment};
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
    let pyramid_opts = PyramidOptions::new(levels).with_blur_levels(config.pyramid.blur_levels);
    let pyramid = Pyramid::build_u8(gray.as_view(), pyramid_opts);
    let coarsest_index = pyramid
        .levels
        .len()
        .checked_sub(1)
        .ok_or("Pyramid has no levels")?;
    let coarsest = &pyramid.levels[coarsest_index];
    let pyramid_stage = PyramidStage::from_pyramid(&pyramid, 0.0);

    let scale = 1 as f32 / (1 << pyramid.levels.len()) as f32;
    let segments = lsd_extract_segments(coarsest, config.lsd.with_scale(scale));

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
