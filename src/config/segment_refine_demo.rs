use super::edge::PyramidConfig;
use crate::refine::segment::SegmentRefineParams;
use crate::segments::LsdOptions;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct SegmentRefineDemoConfig {
    #[serde(rename = "input")]
    pub input: PathBuf,
    #[serde(default)]
    pub pyramid: PyramidConfig,
    #[serde(default)]
    pub lsd: LsdOptions,
    #[serde(default)]
    pub refine: SegmentRefineParams,
    pub output: SegmentRefineDemoOutputConfig,
}

#[derive(Debug, Deserialize)]
pub struct SegmentRefineDemoOutputConfig {
    #[serde(rename = "dir")]
    pub dir: PathBuf,
}

pub fn load_config(path: &Path) -> Result<SegmentRefineDemoConfig, String> {
    let data = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read config {}: {e}", path.display()))?;
    serde_json::from_str(&data)
        .map_err(|e| format!("Failed to parse config {}: {e}", path.display()))
}
