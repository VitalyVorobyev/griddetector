use super::edge::PyramidConfig;
use crate::detector::params::LsdVpParams;
use crate::segments::LsdOptions;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct SegmentToolConfig {
    #[serde(rename = "input")]
    pub input: PathBuf,
    #[serde(default)]
    pub pyramid: PyramidConfig,
    #[serde(default)]
    pub lsd: LsdConfig,
    pub output: SegmentOutputConfig,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct LsdConfig {
    /// Minimum gradient magnitude for seed pixels in the coarsest level (Sobel units).
    pub magnitude_threshold: f32,
    /// Orientation tolerance around the seed normal in degrees.
    pub angle_tolerance_deg: f32,
    /// Minimum accepted segment length in pixels at the current level.
    pub min_length: f32,
    /// If true, require gradient polarity consistency during region growth.
    #[serde(default)]
    pub enforce_polarity: bool,
    /// If true, reject regions whose normal span exceeds `normal_span_limit_px`.
    #[serde(default)]
    pub limit_normal_span: bool,
    /// Maximum allowed span (pixels) along the region normal when `limit_normal_span` is true.
    pub normal_span_limit_px: f32,
}

impl Default for LsdConfig {
    fn default() -> Self {
        Self {
            magnitude_threshold: 0.1,
            angle_tolerance_deg: 12.0,
            min_length: 8.0,
            enforce_polarity: false,
            limit_normal_span: false,
            normal_span_limit_px: 2.0,
        }
    }
}

impl LsdConfig {
    pub fn to_lsd_options(&self) -> LsdOptions {
        LsdOptions {
            enforce_polarity: self.enforce_polarity,
            normal_span_limit: if self.limit_normal_span {
                Some(self.normal_span_limit_px)
            } else {
                None
            },
        }
    }

    pub fn to_lsd_vp_params(&self) -> LsdVpParams {
        LsdVpParams {
            mag_thresh: self.magnitude_threshold,
            angle_tol_deg: self.angle_tolerance_deg,
            min_len: self.min_length,
            enforce_polarity: self.enforce_polarity,
            normal_span_limit: if self.limit_normal_span {
                Some(self.normal_span_limit_px)
            } else {
                None
            },
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SegmentOutputConfig {
    #[serde(rename = "coarsest_image")]
    pub coarsest_image: PathBuf,
    #[serde(rename = "segments_json")]
    pub segments_json: PathBuf,
}

pub fn load_config(path: &Path) -> Result<SegmentToolConfig, String> {
    let data = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read config {}: {e}", path.display()))?;
    serde_json::from_str(&data)
        .map_err(|e| format!("Failed to parse config {}: {e}", path.display()))
}
