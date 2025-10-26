use crate::detector::{
    BundlingParams, BundlingScaleMode, LsdVpParams, OutlierFilterParams, RefinementSchedule,
};
use crate::refine::segment::RefineParams as SegmentRefineParams;
use crate::refine::RefineParams;
use crate::GridParams;
use nalgebra::Matrix3;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Default)]
pub struct OutputConfig {
    pub json_out: Option<PathBuf>,
    pub debug_dir: Option<PathBuf>,
}

#[derive(Clone)]
pub struct RuntimeConfig {
    pub input_path: PathBuf,
    pub output: OutputConfig,
    pub grid_params: GridParams,
}

pub fn load_config(path: &Path) -> Result<RuntimeConfig, String> {
    let contents = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read config {}: {e}", path.display()))?;
    let file_config: FileConfig = serde_json::from_str(&contents)
        .map_err(|e| format!("Failed to parse config {}: {e}", path.display()))?;

    let input_path = file_config
        .input
        .clone()
        .ok_or_else(|| format!("Config {} missing 'input' field", path.display()))?;

    let mut output = OutputConfig::default();
    if let Some(ref file_output) = file_config.output {
        if let Some(ref json) = file_output.json_out {
            output.json_out = Some(json.clone());
        }
        if let Some(ref dir) = file_output.debug_dir {
            output.debug_dir = Some(dir.clone());
        }
    }

    let mut grid_params = GridParams::default();
    if let Some(ref grid_cfg) = file_config.grid {
        apply_grid_file_config(&mut grid_params, grid_cfg);
    }

    Ok(RuntimeConfig {
        input_path,
        output,
        grid_params,
    })
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct FileConfig {
    input: Option<PathBuf>,
    output: Option<FileOutputConfig>,
    grid: Option<FileGridConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct FileOutputConfig {
    json_out: Option<PathBuf>,
    #[serde(rename = "debug_dir")]
    debug_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct FileGridConfig {
    pyramid_levels: Option<usize>,
    pyramid_blur_levels: Option<Option<usize>>,
    spacing_mm: Option<f32>,
    intrinsics: Option<IntrinsicsConfig>,
    min_cells: Option<i32>,
    confidence_thresh: Option<f32>,
    enable_refine: Option<bool>,
    refine: Option<FileRefineConfig>,
    refinement_schedule: Option<FileRefinementScheduleConfig>,
    segment_refine: Option<FileSegmentRefineConfig>,
    lsd_vp: Option<FileLsdVpConfig>,
    bundling: Option<FileBundlingConfig>,
    outlier_filter: Option<FileOutlierFilterConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct FileRefineConfig {
    orientation_tol_deg: Option<f32>,
    huber_delta: Option<f32>,
    max_iterations: Option<usize>,
    min_bundles_per_family: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct FileRefinementScheduleConfig {
    passes: Option<usize>,
    improvement_thresh: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct FileSegmentRefineConfig {
    delta_s: Option<f32>,
    w_perp: Option<f32>,
    delta_t: Option<f32>,
    pad: Option<f32>,
    tau_mag: Option<f32>,
    tau_ori_deg: Option<f32>,
    huber_delta: Option<f32>,
    max_iters: Option<usize>,
    min_inlier_frac: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct FileLsdVpConfig {
    mag_thresh: Option<f32>,
    angle_tol_deg: Option<f32>,
    min_len: Option<f32>,
    enforce_polarity: Option<bool>,
    limit_normal_span: Option<bool>,
    normal_span_limit_px: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct FileBundlingConfig {
    orientation_tol_deg: Option<f32>,
    merge_dist_px: Option<f32>,
    min_weight: Option<f32>,
    scale_mode: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct FileOutlierFilterConfig {
    angle_margin_deg: Option<f32>,
    line_residual_thresh_px: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum IntrinsicsConfig {
    Matrix([[f32; 3]; 3]),
    Parameters { fx: f32, fy: f32, cx: f32, cy: f32 },
}

impl IntrinsicsConfig {
    fn into_matrix(self) -> Matrix3<f32> {
        match self {
            IntrinsicsConfig::Matrix(m) => Matrix3::from_row_slice(&[
                m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
            ]),
            IntrinsicsConfig::Parameters { fx, fy, cx, cy } => {
                Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
            }
        }
    }
}

fn apply_grid_file_config(params: &mut GridParams, cfg: &FileGridConfig) {
    if let Some(levels) = cfg.pyramid_levels {
        params.pyramid_levels = levels;
    }
    if let Some(blur) = cfg.pyramid_blur_levels {
        params.pyramid_blur_levels = blur;
    }
    if let Some(spacing) = cfg.spacing_mm {
        params.spacing_mm = spacing;
    }
    if let Some(ref intr) = cfg.intrinsics {
        params.kmtx = intr.clone().into_matrix();
    }
    if let Some(min_cells) = cfg.min_cells {
        params.min_cells = min_cells;
    }
    if let Some(thresh) = cfg.confidence_thresh {
        params.confidence_thresh = thresh;
    }
    if let Some(enable_refine) = cfg.enable_refine {
        params.enable_refine = enable_refine;
    }
    if let Some(ref refine_cfg) = cfg.refine {
        apply_refine_file_config(&mut params.refine_params, refine_cfg);
    }
    if let Some(ref schedule_cfg) = cfg.refinement_schedule {
        apply_refinement_schedule_config(&mut params.refinement_schedule, schedule_cfg);
    }
    if let Some(ref seg_cfg) = cfg.segment_refine {
        apply_segment_refine_file_config(&mut params.segment_refine_params, seg_cfg);
    }
    if let Some(ref lsd_cfg) = cfg.lsd_vp {
        apply_lsd_vp_file_config(&mut params.lsd_vp_params, lsd_cfg);
    }
    if let Some(ref bundling_cfg) = cfg.bundling {
        apply_bundling_file_config(&mut params.bundling_params, bundling_cfg);
    }
    if let Some(ref outlier_cfg) = cfg.outlier_filter {
        apply_outlier_filter_config(&mut params.outlier_filter, outlier_cfg);
    }
}

fn apply_refine_file_config(params: &mut RefineParams, cfg: &FileRefineConfig) {
    if let Some(v) = cfg.orientation_tol_deg {
        params.orientation_tol_deg = v;
    }
    if let Some(v) = cfg.huber_delta {
        params.huber_delta = v;
    }
    if let Some(v) = cfg.max_iterations {
        params.max_iterations = v;
    }
    if let Some(v) = cfg.min_bundles_per_family {
        params.min_bundles_per_family = v;
    }
}

fn apply_refinement_schedule_config(
    params: &mut RefinementSchedule,
    cfg: &FileRefinementScheduleConfig,
) {
    if let Some(v) = cfg.passes {
        params.passes = v.max(1);
    }
    if let Some(v) = cfg.improvement_thresh {
        params.improvement_thresh = v.max(0.0);
    }
}

fn apply_segment_refine_file_config(
    params: &mut SegmentRefineParams,
    cfg: &FileSegmentRefineConfig,
) {
    if let Some(v) = cfg.delta_s {
        params.delta_s = v;
    }
    if let Some(v) = cfg.w_perp {
        params.w_perp = v;
    }
    if let Some(v) = cfg.delta_t {
        params.delta_t = v;
    }
    if let Some(v) = cfg.pad {
        params.pad = v;
    }
    if let Some(v) = cfg.tau_mag {
        params.tau_mag = v;
    }
    if let Some(v) = cfg.tau_ori_deg {
        params.tau_ori_deg = v;
    }
    if let Some(v) = cfg.huber_delta {
        params.huber_delta = v;
    }
    if let Some(v) = cfg.max_iters {
        params.max_iters = v;
    }
    if let Some(v) = cfg.min_inlier_frac {
        params.min_inlier_frac = v;
    }
}

fn apply_lsd_vp_file_config(params: &mut LsdVpParams, cfg: &FileLsdVpConfig) {
    if let Some(v) = cfg.mag_thresh {
        params.mag_thresh = v;
    }
    if let Some(v) = cfg.angle_tol_deg {
        params.angle_tol_deg = v;
    }
    if let Some(v) = cfg.min_len {
        params.min_len = v;
    }
    if let Some(flag) = cfg.enforce_polarity {
        params.enforce_polarity = flag;
    }

    let mut normal_span = params.normal_span_limit;
    if let Some(limit_px) = cfg.normal_span_limit_px {
        normal_span = Some(limit_px.max(0.0));
    }
    if let Some(limit_enabled) = cfg.limit_normal_span {
        if limit_enabled {
            if normal_span.is_none() {
                normal_span = Some(super::segments::LsdConfig::default().normal_span_limit_px);
            }
        } else {
            normal_span = None;
        }
    }
    params.normal_span_limit = normal_span;
}

fn apply_bundling_file_config(params: &mut BundlingParams, cfg: &FileBundlingConfig) {
    if let Some(v) = cfg.orientation_tol_deg {
        params.orientation_tol_deg = v;
    }
    if let Some(v) = cfg.merge_dist_px {
        params.merge_dist_px = v;
    }
    if let Some(v) = cfg.min_weight {
        params.min_weight = v;
    }
    if let Some(ref mode) = cfg.scale_mode {
        if let Some(parsed) = parse_bundling_scale_mode(mode) {
            params.scale_mode = parsed;
        }
    }
}

fn apply_outlier_filter_config(params: &mut OutlierFilterParams, cfg: &FileOutlierFilterConfig) {
    if let Some(v) = cfg.angle_margin_deg {
        params.angle_margin_deg = v;
    }
    if let Some(v) = cfg.line_residual_thresh_px {
        params.line_residual_thresh_px = v.max(0.0);
    }
}

fn parse_bundling_scale_mode(value: &str) -> Option<BundlingScaleMode> {
    match value.to_ascii_lowercase().as_str() {
        "fixed" | "fixed_pixel" | "fixed_px" => Some(BundlingScaleMode::FixedPixel),
        "full_res" | "full" | "fullres" | "full_resolution" => {
            Some(BundlingScaleMode::FullResInvariant)
        }
        _ => None,
    }
}
