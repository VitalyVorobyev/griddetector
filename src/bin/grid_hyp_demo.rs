use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::pyramid::{build_pyramid, PyramidOptions};
use grid_detector::refine::{refine_coarse_segments, RefineOptions};
use grid_detector::segments::{lsd_extract_segments_coarse, LsdOptions};
use grid_detector::grid::bundling::BundlingParams;
use grid_detector::grid::hypothesis::{build_grid_hypothesis, GridHypothesis};
use grid_detector::grid::vp::VpEstimationOptions;

use serde::Deserialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Deserialize)]
struct Config {
    pub input: PathBuf,
    #[serde(default = "default_pyramid")]
    pub pyramid: PyramidOptions,
    #[serde(default)]
    pub lsd: LsdOptions,
    #[serde(default)]
    pub refine: RefineOptions,
    #[serde(default)]
    pub bundling: BundlingParams,
    #[serde(default)]
    pub vp: VpEstimationOptions,
    pub output_json: PathBuf,
    pub output_image: Option<PathBuf>,
}

#[derive(Clone, Deserialize)]
struct RuntimeConfig {
    pub config: Config,
}

#[derive(serde::Serialize)]
struct DemoReport {
    pub refined_segments: Vec<grid_detector::segments::Segment>,
    pub hypothesis: GridHypothesis,
    pub lsd_ms: f64,
    pub refine_ms: f64,
    pub pyramid_ms: f64,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config_path = env::args().nth(1).ok_or_else(usage)?;
    let cfg = load_config(Path::new(&config_path))?.config;

    let gray = load_grayscale_image(&cfg.input)?;
    let pyr = build_pyramid(gray.as_view(), cfg.pyramid);

    let coarsest = &pyr.pyramid;
    let lsd = lsd_extract_segments_coarse(coarsest, cfg.lsd);
    let lsd_ms = lsd.elapsed_ms;

    let refine_start = std::time::Instant::now();
    let refined = refine_coarse_segments(&pyr.pyramid, &lsd.segments, &cfg.refine, Some(lsd.grad));
    let refine_ms = refine_start.elapsed().as_secs_f64() * 1000.0;

    let hyp = build_grid_hypothesis(&refined.segments, &cfg.bundling, &cfg.vp);

    let report = DemoReport {
        refined_segments: refined.segments,
        hypothesis: hyp,
        lsd_ms,
        refine_ms,
        pyramid_ms: pyr.elapsed_ms,
    };
    write_json_file(&cfg.output_json, &report)?;
    if let Some(path) = cfg.output_image.as_ref() {
        if let Some(level0) = pyr.pyramid.levels.first() {
            save_grayscale_f32(level0, path)?;
        }
    }
    println!(
        "Saved grid hypothesis report to {} (bundles: {}, vp: {})",
        cfg.output_json.display(),
        report.hypothesis.bundles.len(),
        report.hypothesis.vp.is_some()
    );
    Ok(())
}

fn load_config(path: &Path) -> Result<RuntimeConfig, String> {
    let contents = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read config {}: {e}", path.display()))?;
    serde_json::from_str(&contents)
        .map_err(|e| format!("Failed to parse config {}: {e}", path.display()))
}

fn usage() -> String {
    "Usage: grid_hyp_demo <config.json>".to_string()
}

fn default_pyramid() -> PyramidOptions {
    PyramidOptions::new(4)
}
