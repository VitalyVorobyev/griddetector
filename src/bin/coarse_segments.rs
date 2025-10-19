use grid_detector::config::segments;
use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
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
    let config = segments::load_config(Path::new(&config_path))?;

    let gray = load_grayscale_image(&config.input)?;
    let levels = config.pyramid.levels.max(1);
    let pyramid = Pyramid::build_u8(gray.as_view(), levels);
    let coarsest_index = pyramid
        .levels
        .len()
        .checked_sub(1)
        .ok_or("Pyramid has no levels")?;
    let coarsest = &pyramid.levels[coarsest_index];

    let angle_tol_rad = config.lsd.angle_tolerance_deg.to_radians();
    let options = LsdOptions {
        enforce_polarity: config.lsd.enforce_polarity,
        normal_span_limit: if config.lsd.limit_normal_span {
            Some(config.lsd.normal_span_limit_px)
        } else {
            None
        },
    };
    let raw_segments = lsd_extract_segments_with_options(
        coarsest,
        config.lsd.magnitude_threshold,
        angle_tol_rad,
        config.lsd.min_length,
        options,
    );
    let segment_count = raw_segments.len();
    let segments: Vec<SegmentOutput> = raw_segments.into_iter().map(Into::into).collect();

    let summary = SegmentDetectionSummary {
        width: coarsest.w,
        height: coarsest.h,
        pyramid_level_index: coarsest_index,
        magnitude_threshold: config.lsd.magnitude_threshold,
        angle_tolerance_deg: config.lsd.angle_tolerance_deg,
        min_length: config.lsd.min_length,
        enforce_polarity: config.lsd.enforce_polarity,
        limit_normal_span: config.lsd.limit_normal_span,
        normal_span_limit_px: if config.lsd.limit_normal_span {
            Some(config.lsd.normal_span_limit_px)
        } else {
            None
        },
        segment_count,
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
        summary.segment_count,
        config.output.segments_json.display()
    );

    Ok(())
}

fn usage() -> String {
    "Usage: coarse_segments <config.json>".to_string()
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SegmentDetectionSummary {
    width: usize,
    height: usize,
    pyramid_level_index: usize,
    magnitude_threshold: f32,
    angle_tolerance_deg: f32,
    min_length: f32,
    enforce_polarity: bool,
    limit_normal_span: bool,
    normal_span_limit_px: Option<f32>,
    segment_count: usize,
    segments: Vec<SegmentOutput>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SegmentOutput {
    p0: [f32; 2],
    p1: [f32; 2],
    direction: [f32; 2],
    length: f32,
    line: [f32; 3],
    average_magnitude: f32,
    strength: f32,
}

impl From<Segment> for SegmentOutput {
    fn from(seg: Segment) -> Self {
        Self {
            p0: seg.p0,
            p1: seg.p1,
            direction: seg.dir,
            length: seg.len,
            line: seg.line,
            average_magnitude: seg.avg_mag,
            strength: seg.strength,
        }
    }
}
