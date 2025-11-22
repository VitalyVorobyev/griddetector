use grid_detector::edges::{detect_edges_nms, EdgeElement};
use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::pyramid::{Pyramid, PyramidOptions};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct EdgeToolConfig {
    #[serde(rename = "input")]
    pub input: PathBuf,
    pub pyramid: PyramidOptions,
    #[serde(default)]
    pub edge: EdgeDetectorConfig,
    pub output: EdgeOutputConfig,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct EdgeDetectorConfig {
    pub magnitude_threshold: f32,
}

impl Default for EdgeDetectorConfig {
    fn default() -> Self {
        Self {
            magnitude_threshold: 0.1,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EdgeOutputConfig {
    #[serde(rename = "coarsest_image")]
    pub coarsest_image: PathBuf,
    #[serde(rename = "edges_json")]
    pub edges_json: PathBuf,
}

pub fn load_config(path: &Path) -> Result<EdgeToolConfig, String> {
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

fn run() -> Result<(), String> {
    let config_path = env::args().nth(1).ok_or_else(usage)?;
    let config = load_config(Path::new(&config_path))?;

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

    let edges = detect_edges_nms(coarsest, config.edge.magnitude_threshold);
    let summary = EdgeDetectionSummary {
        width: coarsest.w,
        height: coarsest.h,
        pyramid_level_index: coarsest_index,
        magnitude_threshold: config.edge.magnitude_threshold,
        edge_count: edges.len(),
        edges,
    };

    save_grayscale_f32(coarsest, &config.output.coarsest_image)?;
    write_json_file(&config.output.edges_json, &summary)?;

    println!(
        "Saved coarsest level image to {} (level {})",
        config.output.coarsest_image.display(),
        coarsest_index
    );
    println!(
        "Saved {} edge elements to {}",
        summary.edge_count,
        config.output.edges_json.display()
    );

    Ok(())
}

fn usage() -> String {
    "Usage: coarse_edges <config.json>".to_string()
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EdgeDetectionSummary {
    width: usize,
    height: usize,
    pyramid_level_index: usize,
    magnitude_threshold: f32,
    edge_count: usize,
    edges: Vec<EdgeElement>,
}
