//! Demonstration binary for the gradient-based segment refiner.
//!
//! The demo mirrors the early detector pipeline stages:
//! 1. Build an image pyramid.
//! 2. Run LSD on the coarsest level.
//! 3. Build an orientation histogram and assign segment families.
//! 4. Refine the segments down the pyramid using only local gradients.
//! 5. Emit diagnostics identical to the pipeline (pyramid, LSD stage, refinement levels).

use grid_detector::image::io::{
    load_grayscale_image, save_grayscale_f32
};
use grid_detector::pyramid::{PyramidOptions, PyramidResult, build_pyramid};
use grid_detector::refine::{RefineOptions, refine_coarse_segments, SegmentsRefinementResult};
use grid_detector::segments::{LsdOptions, LsdResult, lsd_extract_segments_coarse};
use grid_detector::diagnostics::{run_with_timer, ResultWithTime};

#[cfg(feature = "profile_refine")]
use grid_detector::refine::segment::take_profile;

use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct SegmentRefineDemoConfig {
    #[serde(rename = "input")]
    pub input: PathBuf,
    pub pyramid: PyramidOptions,
    #[serde(default)]
    pub lsd: LsdOptions,
    #[serde(default)]
    pub refine: RefineOptions,
    #[serde(default)]
    pub performance_mode: bool,
    pub output: SegmentRefineDemoOutputConfig,
}

#[derive(Debug, Deserialize)]
struct SegmentRefineDemoOutputConfig {
    #[serde(rename = "dir")]
    pub dir: PathBuf,
}

#[derive(Serialize, Debug, Default)]
struct DemoReport {
    pub pyramid: PyramidResult,
    pub lsd: LsdResult,
    pub refine: SegmentsRefinementResult
}

fn load_config(path: &Path) -> Result<SegmentRefineDemoConfig, String> {
    let data = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read config {}: {e}", path.display()))?;
    serde_json::from_str(&data)
        .map_err(|e| format!("Failed to parse config {}: {e}", path.display()))
}


fn main() {
    let config = load_config_from_args().map_err(

    );
    ensure_output_dir(&config).map_err();

    let ResultWithTime {
        result: gray,
        elapsed_ms: load_ms,
    } = run_with_timer(|| load_grayscale_image(&config.input))?;
    println!("Image loaded in {} ms", load_ms);

    let total_start = Instant::now();
    run(config, gray.as_slice()).unwrap_or_else(|err| {
        eprintln!("Error: {err}");
        std::process::exit(1);
    })
    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    println!("Total execution time: {:.2} ms", total_ms);

    let report_path = config.output.dir.join("report.json");
    // write_json_file(&report_path, &stage)?;
    println!(
        "Segment refinement report written to {}",
        report_path.display()
    );

    save_pyramid_images(gray.as_slice(), out_dir);
}

fn run(config: SegmentRefineDemoConfig, gray:ImageU8) -> Result<DemoReport, String> {
    let diagnostics_enabled = !config.performance_mode;

    let pyramid = build_pyramid(gray);
    let lsd  = lsd_extract_segments_coarse(&pyramid, self.params.lsd);
    let refine = refine_coarse_segments(&pyramid, &segments, self.params.refine, grad);

    Ok(DemoReport {pyramid, lsd, refine})
}

fn usage() -> String {
    "Usage: segment_refine_demo <config.json>".to_string()
}

fn load_config_from_args() -> Result<SegmentRefineDemoConfig, String> {
    let config_path = env::args().nth(1).ok_or_else(usage)?;
    load_config(Path::new(&config_path))
}

fn ensure_output_dir(config: &SegmentRefineDemoConfig) -> Result<(), String> {
    fs::create_dir_all(&config.output.dir)
        .map_err(|e| format!("Failed to create {}: {e}", config.output.dir.display()))
}

fn save_pyramid_images(gray:ImageU8, out_dir: &Path) -> Result<(), String> {
    let pyramid = build_pyramid(gray);
    for (idx, level) in pyramid.levels.iter().enumerate() {
        let path = out_dir.join(format!("pyramid_L{idx}.png"));
        save_grayscale_f32(level, &path)?;
    }
    Ok(())
}
