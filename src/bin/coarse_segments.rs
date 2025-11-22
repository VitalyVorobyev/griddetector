use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::pyramid::{build_pyramid, PyramidOptions, PyramidResult};
use grid_detector::segments::{
    lsd_extract_segments_coarse, lsd_extract_segments_nms, LsdOptions, LsdResult, Segment,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::{env, fs};

#[derive(Debug, Deserialize)]
pub struct SegmentToolConfig {
    #[serde(rename = "input")]
    pub input: PathBuf,
    pub pyramid: PyramidOptions,
    #[serde(default)]
    pub lsd: LsdOptions,
    pub output: SegmentOutputConfig,
}

#[derive(Debug, Deserialize)]
pub struct SegmentOutputConfig {
    #[serde(rename = "coarsest_image")]
    pub coarsest_image: PathBuf,
    #[serde(rename = "segments_json")]
    pub segments_json: PathBuf,
}

pub fn load_config(path: &Path) -> Result<SegmentToolConfig, String> {
    let data = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read config {}: {e}", path.display()))?;
    serde_json::from_str(&data)
        .map_err(|e| format!("Failed to parse config {}: {e}", path.display()))
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

#[derive(Serialize)]
struct Report {
    segments: Vec<Segment>,
    pyr_ms: f64,
    pyr_lo_ms: f64,
    lsd_full_ms: f64,
    lsd_nms_ms: f64,
    nms_ms: f64,
    nms_gradient_ms: f64,
    nms_edges: usize,
}

fn run() -> Result<(), String> {
    let config_path = env::args().nth(1).ok_or_else(usage)?;
    let config = load_config(Path::new(&config_path))?;

    let gray = load_grayscale_image(&config.input)?;

    let PyramidResult {
        pyramid,
        elapsed_ms: pyr_ms,
        elapsed_convert_l0_ms: pyr_lo_ms,
    } = build_pyramid(gray.as_view(), config.pyramid);
    let coarsest = pyramid
        .levels
        .last()
        .ok_or_else(|| "Pyramid must have at least one level".to_string())?;
    let scale = pyramid.scale_for_level(coarsest);
    let scaled_opts = config.lsd.with_scale(scale);

    let LsdResult {
        segments: segments_full,
        grad: _,
        elapsed_ms: lsd_full_ms,
    } = lsd_extract_segments_coarse(&pyramid, config.lsd);

    let (lsd_nms, nms) = lsd_extract_segments_nms(
        coarsest,
        scaled_opts,
        scaled_opts.magnitude_threshold,
    );

    println!("        l0: {:5.2} ms", pyr_lo_ms);
    println!("       pyr: {:5.2} ms", pyr_ms);
    println!("  lsd full: {:5.2} ms ({} segments)", lsd_full_ms, segments_full.len());
    println!("       nms: {:5.2} ms (edges {})", nms.nms_ms, nms.edges.len());
    println!("lsd seeded: {:5.2} ms ({} segments)", lsd_nms.elapsed_ms, lsd_nms.segments.len());

    save_grayscale_f32(coarsest, &config.output.coarsest_image)?;

    let summary = Report {
        segments: lsd_nms.segments,
        pyr_ms,
        pyr_lo_ms,
        lsd_full_ms,
        lsd_nms_ms: lsd_nms.elapsed_ms,
        nms_ms: nms.nms_ms,
        nms_gradient_ms: nms.gradient_ms,
        nms_edges: nms.edges.len(),
    };
    write_json_file(&config.output.segments_json, &summary)?;

    println!(
        "Saved coarsest level image to {} (level {})",
        config.output.coarsest_image.display(),
        pyramid.levels.len()
    );

    Ok(())
}

fn usage() -> String {
    "Usage: coarse_segments <config.json>".to_string()
}
