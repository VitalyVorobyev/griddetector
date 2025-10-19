use grid_detector::edges::{detect_edges_sobel_nms, EdgeElement};
use grid_detector::pyramid::Pyramid;
use grid_detector::types::ImageU8;
use image::{GrayImage, Luma};
use serde::Deserialize;
use serde::Serialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config_path = env::args().nth(1).ok_or_else(|| usage())?;
    let config = load_config(Path::new(&config_path))?;

    let img = image::open(&config.input_image)
        .map_err(|e| format!("Failed to open {}: {e}", config.input_image.display()))?
        .to_luma8();
    let width = img.width() as usize;
    let height = img.height() as usize;
    let stride = width;
    let gray_buf = img.into_raw();
    let image = ImageU8 {
        w: width,
        h: height,
        stride,
        data: &gray_buf,
    };

    let levels = config.pyramid.levels.max(1);
    let pyramid = Pyramid::build_u8(image, levels);
    let coarsest_index = pyramid
        .levels
        .len()
        .checked_sub(1)
        .ok_or("Pyramid has no levels")?;
    let coarsest = &pyramid.levels[coarsest_index];

    let edges = detect_edges_sobel_nms(coarsest, config.edge.magnitude_threshold);
    let summary = EdgeDetectionSummary {
        width: coarsest.w,
        height: coarsest.h,
        pyramid_level_index: coarsest_index,
        magnitude_threshold: config.edge.magnitude_threshold,
        edge_count: edges.len(),
        edges,
    };

    save_coarsest_image(coarsest, &config.output.coarsest_image)?;
    save_edges_json(&summary, &config.output.edges_json)?;

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

fn load_config(path: &Path) -> Result<PipelineConfig, String> {
    let data = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read config {}: {e}", path.display()))?;
    serde_json::from_str(&data)
        .map_err(|e| format!("Failed to parse config {}: {e}", path.display()))
}

fn save_coarsest_image(image: &grid_detector::types::ImageF32, path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create {}: {e}", parent.display()))?;
        }
    }

    let mut out = GrayImage::new(image.w as u32, image.h as u32);
    for y in 0..image.h {
        for x in 0..image.w {
            let v = (image.get(x, y) * 255.0).clamp(0.0, 255.0);
            out.put_pixel(x as u32, y as u32, Luma([v as u8]));
        }
    }

    out.save(path)
        .map_err(|e| format!("Failed to save {}: {e}", path.display()))
}

fn save_edges_json(summary: &EdgeDetectionSummary, path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create {}: {e}", parent.display()))?;
        }
    }

    let json = serde_json::to_string_pretty(summary)
        .map_err(|e| format!("Failed to serialize edges JSON: {e}"))?;
    fs::write(path, json).map_err(|e| format!("Failed to write {}: {e}", path.display()))
}

fn usage() -> String {
    "Usage: coarse_edges <config.json>".to_string()
}

#[derive(Debug, Deserialize)]
struct PipelineConfig {
    #[serde(rename = "input")]
    input_image: PathBuf,
    #[serde(default)]
    pyramid: PyramidConfig,
    #[serde(default)]
    edge: EdgeConfig,
    output: OutputConfig,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct PyramidConfig {
    levels: usize,
}

impl Default for PyramidConfig {
    fn default() -> Self {
        Self { levels: 4 }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct EdgeConfig {
    magnitude_threshold: f32,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            magnitude_threshold: 0.1,
        }
    }
}

#[derive(Debug, Deserialize)]
struct OutputConfig {
    coarsest_image: PathBuf,
    edges_json: PathBuf,
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
