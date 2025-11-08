use super::edge::PyramidConfig;
use crate::detector::params::{BundlingParams, OutlierFilterParams};
use crate::refine::RefineParams;
use crate::segments::LsdOptions;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct VpOutlierDemoConfig {
    #[serde(rename = "input")]
    pub input: PathBuf,
    #[serde(default)]
    pub pyramid: PyramidConfig,
    #[serde(default)]
    pub lsd: LsdOptions,
    #[serde(default)]
    pub outlier: OutlierFilterParams,
    #[serde(default)]
    pub bundling: BundlingParams,
    #[serde(default)]
    pub refine: RefineParams,
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
