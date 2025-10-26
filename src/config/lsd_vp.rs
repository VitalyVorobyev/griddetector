use super::edge::PyramidConfig;
use super::segments::LsdConfig;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct LsdVpDemoConfig {
    #[serde(rename = "input")]
    pub input: PathBuf,
    #[serde(default)]
    pub pyramid: PyramidConfig,
    #[serde(default)]
    pub lsd: LsdConfig,
    pub output: LsdVpOutputConfig,
}

#[derive(Debug, Deserialize)]
pub struct LsdVpOutputConfig {
    #[serde(rename = "coarsest_image")]
    pub coarsest_image: PathBuf,
    #[serde(rename = "result_json")]
    pub result_json: PathBuf,
}

pub fn load_config(path: &Path) -> Result<LsdVpDemoConfig, String> {
    let data = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read config {}: {e}", path.display()))?;
    serde_json::from_str(&data)
        .map_err(|e| format!("Failed to parse config {}: {e}", path.display()))
}
