use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::pyramid::{build_pyramid, PyramidOptions, PyramidResult};
use grid_detector::segments::{lsd_extract_segments_coarse, LsdOptions, LsdResult, Segment};
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
    lsd_ms: f64,
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

    let LsdResult {
        segments,
        grad: _,
        elapsed_ms: lsd_ms,
    } = lsd_extract_segments_coarse(&pyramid, config.lsd);

    println!(" l0: {:5.2} ms", pyr_lo_ms);
    println!("pyr: {:5.2} ms", pyr_ms);
    println!("lsd: {:5.2} ms", lsd_ms);

    let coarsest = pyramid
        .levels
        .last()
        .ok_or_else(|| "Pyramid must have at least one level".to_string())?;
    save_grayscale_f32(coarsest, &config.output.coarsest_image)?;

    let summary = Report {
        segments,
        pyr_ms,
        pyr_lo_ms,
        lsd_ms,
    };
    write_json_file(&config.output.segments_json, &summary)?;

    println!(
        "Saved coarsest level image to {} (level {})",
        config.output.coarsest_image.display(),
        pyramid.levels.len()
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
