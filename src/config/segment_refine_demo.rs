use super::edge::PyramidConfig;
use crate::refine::segment::driver::ParallelRefineOptions;
use crate::refine::segment::RefineParams as SegmentRefineParams;
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
    #[serde(default)]
    pub performance_mode: bool,
    #[serde(default)]
    pub timing: SegmentRefineTimingConfig,
    pub output: SegmentRefineDemoOutputConfig,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
pub struct SegmentRefineConfig {
    pub delta_s: Option<f32>,
    pub w_perp: Option<f32>,
    pub delta_t: Option<f32>,
    pub pad: Option<f32>,
    pub tau_mag: Option<f32>,
    pub tau_ori_deg: Option<f32>,
    pub huber_delta: Option<f32>,
    pub max_iters: Option<usize>,
    pub min_inlier_frac: Option<f32>,
}

impl SegmentRefineConfig {
    pub fn resolve(&self) -> SegmentRefineParams {
        let mut p = SegmentRefineParams::default();
        if let Some(v) = self.delta_s {
            p.delta_s = v;
        }
        if let Some(v) = self.w_perp {
            p.w_perp = v;
        }
        if let Some(v) = self.delta_t {
            p.delta_t = v;
        }
        if let Some(v) = self.pad {
            p.pad = v;
        }
        if let Some(v) = self.tau_mag {
            p.tau_mag = v;
        }
        if let Some(v) = self.tau_ori_deg {
            p.tau_ori_deg = v;
        }
        if let Some(v) = self.huber_delta {
            p.huber_delta = v;
        }
        if let Some(v) = self.max_iters {
            p.max_iters = v;
        }
        if let Some(v) = self.min_inlier_frac {
            p.min_inlier_frac = v;
        }
        p
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
pub struct SegmentRefineTimingConfig {
    /// Number of warm-up runs skipped from reporting.
    pub warmup_runs: usize,
    /// Number of measurement runs recorded (>=1).
    pub runs: usize,
    /// Optional Rayon worker limit.
    pub max_threads: Option<usize>,
    /// Optional threshold overriding the parallel fallback.
    pub min_segments_for_parallel: Option<usize>,
    /// Forces sequential refinement when true.
    pub force_sequential: bool,
}

impl SegmentRefineTimingConfig {
    pub fn measurement_runs(&self) -> usize {
        self.runs.max(1)
    }

    pub fn warmup_runs(&self) -> usize {
        self.warmup_runs
    }

    pub fn parallel_options(&self) -> ParallelRefineOptions {
        if self.force_sequential {
            return ParallelRefineOptions::disabled();
        }
        if let Some(min_segments) = self.min_segments_for_parallel {
            ParallelRefineOptions::default().with_min_segments(min_segments)
        } else {
            ParallelRefineOptions::default()
        }
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
