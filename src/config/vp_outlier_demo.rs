use super::edge::PyramidConfig;
use super::segments::LsdConfig;
use crate::detector::params::{
    BundlingParams, BundlingScaleMode, LsdVpParams, OutlierFilterParams,
};
use crate::refine::RefineParams as HomographyRefineParams;
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
    pub lsd: LsdConfig,
    #[serde(default)]
    pub outlier: OutlierConfig,
    #[serde(default)]
    pub bundling: BundlingConfig,
    #[serde(default)]
    pub refine: RefineConfig,
    pub output: DemoOutputConfig,
}

impl VpOutlierDemoConfig {
    /// Convert resolved engine parameters into the detector's `LsdVpParams`.
    pub fn resolve_lsd_vp_params(&self) -> LsdVpParams {
        self.lsd.to_lsd_vp_params()
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
pub struct OutlierConfig {
    pub angle_margin_deg: Option<f32>,
    pub residual_thresh_px: Option<f32>,
}

impl OutlierConfig {
    pub fn resolve(&self) -> OutlierFilterParams {
        let mut params = OutlierFilterParams::default();
        if let Some(v) = self.angle_margin_deg {
            params.angle_margin_deg = v;
        }
        if let Some(v) = self.residual_thresh_px {
            params.line_residual_thresh_px = v;
        }
        params
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
pub struct BundlingConfig {
    pub orientation_tol_deg: Option<f32>,
    pub merge_dist_px: Option<f32>,
    pub min_weight: Option<f32>,
    pub full_res_invariant: Option<bool>,
}

impl BundlingConfig {
    pub fn resolve(&self) -> BundlingParams {
        let mut params = BundlingParams::default();
        if let Some(v) = self.orientation_tol_deg {
            params.orientation_tol_deg = v;
        }
        if let Some(v) = self.merge_dist_px {
            params.merge_dist_px = v;
        }
        if let Some(v) = self.min_weight {
            params.min_weight = v;
        }
        params.scale_mode = if self.full_res_invariant.unwrap_or(false) {
            BundlingScaleMode::FullResInvariant
        } else {
            BundlingScaleMode::FixedPixel
        };
        params
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
pub struct RefineConfig {
    pub orientation_tol_deg: Option<f32>,
    pub huber_delta: Option<f32>,
    pub max_iterations: Option<usize>,
    pub min_bundles_per_family: Option<usize>,
}

impl RefineConfig {
    pub fn resolve(&self) -> HomographyRefineParams {
        let mut params = HomographyRefineParams::default();
        if let Some(v) = self.orientation_tol_deg {
            params.orientation_tol_deg = v;
        }
        if let Some(v) = self.huber_delta {
            params.huber_delta = v;
        }
        if let Some(v) = self.max_iterations {
            params.max_iterations = v;
        }
        if let Some(v) = self.min_bundles_per_family {
            params.min_bundles_per_family = v;
        }
        params
    }
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
