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
    pub refine: SegmentRefineConfig,
    pub output: SegmentRefineDemoOutputConfig,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
pub struct SegmentRefineConfig {
    pub anchor_search_radius: Option<f32>,
    pub normal_step: Option<f32>,
    pub tangent_step: Option<f32>,
    pub max_grow_steps: Option<usize>,
    pub gradient_threshold: Option<f32>,
    pub max_normal_shift: Option<f32>,
    pub max_point_dist: Option<f32>,
    pub min_support_points: Option<usize>,
    pub max_seeds: Option<usize>,
}

impl SegmentRefineConfig {
    pub fn resolve(&self) -> SegmentRefineParams {
        let mut p = SegmentRefineParams::default();
        if let Some(v) = self.anchor_search_radius {
            p.anchor_search_radius = v;
        }
        if let Some(v) = self.normal_step {
            p.normal_step = v;
        }
        if let Some(v) = self.tangent_step {
            p.tangent_step = v;
        }
        if let Some(v) = self.max_grow_steps {
            p.max_grow_steps = v;
        }
        if let Some(v) = self.gradient_threshold {
            p.gradient_threshold = v;
        }
        if let Some(v) = self.max_normal_shift {
            p.max_normal_shift = v;
        }
        if let Some(v) = self.max_point_dist {
            p.max_point_dist = v;
        }
        if let Some(v) = self.min_support_points {
            p.min_support_points = v;
        }
        if let Some(v) = self.max_seeds {
            p.max_seeds = v;
        }
        p
    }
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
