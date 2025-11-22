use grid_detector::detector::GridParams;
use grid_detector::image::io::{load_grayscale_image, write_json_file};
use grid_detector::GridDetector;
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Default, Deserialize)]
pub struct OutputConfig {
    pub json_out: Option<PathBuf>,
    pub debug_dir: Option<PathBuf>,
}

#[derive(Clone, Deserialize)]
pub struct RuntimeConfig {
    pub input_path: PathBuf,
    pub output: OutputConfig,
    pub grid_params: GridParams,
}

pub fn load_config(path: &Path) -> Result<RuntimeConfig, String> {
    let contents = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read config {}: {e}", path.display()))?;
    let config: RuntimeConfig = serde_json::from_str(&contents)
        .map_err(|e| format!("Failed to parse config {}: {e}", path.display()))?;
    Ok(config)
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

    let gray = load_grayscale_image(&config.input_path)?;
    let image = gray.as_view();

    let mut detector = GridDetector::new(config.grid_params.clone());
    let detailed = detector.process(image);

    if let Some(path) = &config.output.json_out {
        write_json_file(path, &detailed)?;
        println!("JSON report written to {}", path.display());
    } else {
        eprintln!("No JSON output path specified, skipping JSON report.");
    }

    Ok(())
}

fn usage() -> String {
    "Usage: grid_demo <config.json>".to_string()
}
