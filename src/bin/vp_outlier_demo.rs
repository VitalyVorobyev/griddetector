use grid_detector::detector::{BundlingParams, OutlierFilterParams};
use grid_detector::image::io::{
    load_grayscale_image, save_grayscale_f32, write_json_file, GrayImageU8,
};
use grid_detector::pyramid::{Pyramid, PyramidOptions};
use grid_detector::refine::RefineOptions;
use grid_detector::segments::LsdOptions;
use grid_detector::{GridDetector, GridParams};
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct VpOutlierDemoConfig {
    #[serde(rename = "input")]
    pub input: PathBuf,
    pub pyramid: PyramidOptions,
    #[serde(default)]
    pub lsd: LsdOptions,
    #[serde(default)]
    pub outlier: OutlierFilterParams,
    #[serde(default)]
    pub bundling: BundlingParams,
    #[serde(default)]
    pub refine: RefineOptions,
    pub output: DemoOutputConfig,
}

#[derive(Debug, Deserialize)]
pub struct DemoOutputConfig {
    #[serde(rename = "dir")]
    pub dir: PathBuf,
    #[serde(rename = "coarsest_image")]
    pub coarsest_image: PathBuf,
    #[serde(rename = "result_json")]
    pub result_json: PathBuf,
}

impl DemoOutputConfig {
    pub fn coarsest_path(&self) -> PathBuf {
        resolve_path(&self.dir, &self.coarsest_image)
    }

    pub fn result_path(&self) -> PathBuf {
        resolve_path(&self.dir, &self.result_json)
    }
}

pub fn load_config(path: &Path) -> Result<VpOutlierDemoConfig, String> {
    let data = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read config {}: {e}", path.display()))?;
    serde_json::from_str(&data)
        .map_err(|e| format!("Failed to parse config {}: {e}", path.display()))
}

fn resolve_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn build_pyramid_save_coarsest_image(
    img: &GrayImageU8,
    config: &VpOutlierDemoConfig,
    levels: usize,
) -> Result<(), String> {
    let pyramid_opts = build_pyramid_options(config, levels);
    let pyramid = Pyramid::build_u8(img.as_view(), pyramid_opts);
    save_coarsest_level(config, &pyramid)
}

fn run() -> Result<(), String> {
    let config_path = env::args().nth(1).ok_or_else(usage)?;
    let config = load_config(Path::new(&config_path))?;

    fs::create_dir_all(&config.output.dir)
        .map_err(|e| format!("Failed to create {}: {e}", config.output.dir.display()))?;

    let gray = load_grayscale_image(&config.input)?;
    let levels = config.pyramid.levels;
    build_pyramid_save_coarsest_image(&gray, &config, levels)?;

    let params = grid_params_from_config(&config, levels);
    let mut detector = GridDetector::new(params);

    let report = detector.process_coarsest(gray.as_view());
    report.print_text_summary();

    write_json_file(&config.output.result_path(), &report)?;

    println!(
        "Wrote diagnostics JSON to {}",
        config.output.result_path().display()
    );

    Ok(())
}

fn usage() -> String {
    "Usage: vp_outlier_demo <config.json>".to_string()
}

fn build_pyramid_options(config: &VpOutlierDemoConfig, levels: usize) -> PyramidOptions {
    PyramidOptions::new(levels).with_blur_levels(config.pyramid.blur_levels)
}

fn save_coarsest_level(config: &VpOutlierDemoConfig, pyramid: &Pyramid) -> Result<(), String> {
    let idx = pyramid
        .levels
        .len()
        .checked_sub(1)
        .ok_or_else(|| "Pyramid must contain at least one level".to_string())?;
    let coarsest = &pyramid.levels[idx];
    save_grayscale_f32(coarsest, &config.output.coarsest_path())
        .map_err(|e| format!("Failed to save coarsest level: {e}"))?;
    println!(
        "Saved coarsest level image to {} (level {})",
        config.output.coarsest_path().display(),
        idx
    );
    Ok(())
}

fn grid_params_from_config(config: &VpOutlierDemoConfig, levels: usize) -> GridParams {
    GridParams {
        pyramid_levels: levels,
        pyramid_blur_levels: config.pyramid.blur_levels,
        lsd_params: config.lsd,
        outlier_filter: config.outlier.clone(),
        bundling_params: config.bundling.clone(),
        refine_params: config.refine.clone(),
        ..GridParams::default()
    }
}
