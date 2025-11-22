//! Demonstration binary for the gradient-based segment refiner.
//!
//! Flow:
//! 1. Build an image pyramid.
//! 2. Run LSD on the coarsest level.
//! 3. Refine coarse segments down the pyramid using only local gradients.
//! 4. Emit a small JSON report and optional images.

use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::pyramid::{build_pyramid, Pyramid, PyramidOptions, PyramidResult};
use grid_detector::refine::{refine_coarse_segments, RefineOptions, SegmentsRefinementResult};
use grid_detector::segments::{lsd_extract_segments_coarse, LsdOptions, LsdResult};

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

#[derive(Debug, Serialize, Default)]
struct DemoReport {
    pub pyramid: PyramidResult,
    pub refine: SegmentsRefinementResult,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config = load_config_from_args()?;
    ensure_output_dir(&config)?;

    let gray = load_grayscale_image(&config.input)?;
    println!(
        "Loaded image {} ({}×{})",
        config.input.display(),
        gray.width(),
        gray.height()
    );

    let total_start = Instant::now();

    // Pyramid, LSD, and refinement share the same pyramid instance.
    let pyramid_result = build_pyramid(gray.as_view(), config.pyramid);
    let pyramid = &pyramid_result.pyramid;

    // Coarse LSD
    let LsdResult {
        segments,
        grad,
        elapsed_ms: lsd_ms
    } = lsd_extract_segments_coarse(pyramid, config.lsd);

    // Gradient-driven refinement
    let refine_result = refine_coarse_segments(
        pyramid,
        &segments,
        &config.refine,
        Some(grad),
    );

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    println!("L0 convert: {:5.2} ms", pyramid_result.elapsed_convert_l0_ms);
    println!("   pyramid: {:5.2} ms ({} levels)", pyramid_result.elapsed_ms, pyramid.levels.len());
    println!("       LSD: {:5.2} ms ({} segments)", lsd_ms, segments.len());
    println!("refinement: {:5.2} ms", refine_result.elapsed_ms);
    for (i, item) in refine_result.levels.iter().enumerate() {
        // Level indices descend from coarsest->finest; map back to the finer level index.
        let level_idx = pyramid.levels.len().saturating_sub(i + 2);
        println!(
            "  - level {}: {:5.2} ms, {:4} / {:4} accepted",
            level_idx,
            item.elapsed_ms,
            item.accepted,
            item.attempted
        );
    }
    println!(
        "accepted segments: {} (from {} seeds)",
        refine_result.segments.len(),
        segments.len()
    );
    println!("total: {:5.2} ms", total_ms);
    
    if !config.performance_mode {
        save_pyramid_images(pyramid, &config.output.dir)?;
    }

    // Save report and optional images.
    let report = DemoReport {
        pyramid: pyramid_result,
        refine: refine_result,
    };

    let report_path = config.output.dir.join("report.json");
    write_json_file(&report_path, &report)?;
    println!(
        "Segment refinement report written to {}",
        report_path.display()
    );

    Ok(())
}

fn load_config(path: &Path) -> Result<SegmentRefineDemoConfig, String> {
    let data = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read config {}: {e}", path.display()))?;
    serde_json::from_str(&data)
        .map_err(|e| format!("Failed to parse config {}: {e}", path.display()))
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

fn save_pyramid_images(pyramid: &Pyramid, out_dir: &Path) -> Result<(), String> {
    for (idx, level) in pyramid.levels.iter().enumerate() {
        let path = out_dir.join(format!("pyramid_L{idx}.png"));
        save_grayscale_f32(level, &path)?;
    }
    Ok(())
}
