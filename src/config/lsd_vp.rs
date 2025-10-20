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
    #[serde(default)]
    pub engine: VpEngineConfig,
    pub output: LsdVpOutputConfig,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
pub struct VpEngineConfig {
    pub magnitude_threshold: Option<f32>,
    pub angle_tolerance_deg: Option<f32>,
    pub min_length: Option<f32>,
}

impl VpEngineConfig {
    pub fn resolve(&self, lsd: &LsdConfig) -> EngineParameters {
        EngineParameters {
            magnitude_threshold: self.magnitude_threshold.unwrap_or(lsd.magnitude_threshold),
            angle_tolerance_deg: self.angle_tolerance_deg.unwrap_or(lsd.angle_tolerance_deg),
            min_length: self.min_length.unwrap_or(lsd.min_length),
        }
    }
}

#[derive(Debug)]
pub struct EngineParameters {
    pub magnitude_threshold: f32,
    pub angle_tolerance_deg: f32,
    pub min_length: f32,
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
