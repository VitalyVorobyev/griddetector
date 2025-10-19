use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct EdgeToolConfig {
    #[serde(rename = "input")]
    pub input: PathBuf,
    #[serde(default)]
    pub pyramid: PyramidConfig,
    #[serde(default)]
    pub edge: EdgeDetectorConfig,
    pub output: EdgeOutputConfig,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct PyramidConfig {
    pub levels: usize,
    /// How many downscale steps apply Gaussian blur before decimation.
    ///
    /// - `None` (default): blur on all steps (legacy behavior).
    /// - `0`: never blur (fastest, may alias).
    /// - `1`: blur only before the first 2× downscale, etc.
    pub blur_levels: Option<usize>,
}

impl Default for PyramidConfig {
    fn default() -> Self {
        Self {
            levels: 4,
            blur_levels: None,
        }
    }
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
