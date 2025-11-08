use crate::GridParams;
use std::fs;
use std::path::{Path, PathBuf};
use serde::Deserialize;

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
