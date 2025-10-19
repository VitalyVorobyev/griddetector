use grid_detector::diagnostics::DetailedResult;
use grid_detector::refine::RefineParams;
use grid_detector::types::ImageU8;
use grid_detector::{GridDetector, GridParams};
use nalgebra::Matrix3;
use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer};
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let program = env::args()
        .next()
        .unwrap_or_else(|| "grid_demo".to_string());
    let config = parse_args(&program)?;

    let img = image::open(&config.input_path)
        .map_err(|e| format!("Failed to open {}: {e}", config.input_path.display()))?
        .to_luma8();
    let width = img.width() as usize;
    let height = img.height() as usize;
    let stride = width;
    let gray_buf = img.into_raw();
    let image = ImageU8 {
        w: width,
        h: height,
        stride,
        data: &gray_buf,
    };

    let mut detector = GridDetector::new(config.grid_params.clone());
    let detailed = detector.process_with_diagnostics(image);

    if config.output.format.includes_text() {
        print_text_summary(&detailed);
    }

    if config.output.format.includes_json() {
        let json = serde_json::to_string_pretty(&detailed)
            .map_err(|e| format!("Failed to serialize JSON: {e}"))?;
        if let Some(path) = &config.output.json_out {
            fs::write(path, &json)
                .map_err(|e| format!("Failed to write JSON report to {}: {e}", path.display()))?;
            if !config.output.format.includes_text() {
                println!("JSON report written to {}", path.display());
            } else {
                println!("\nJSON report written to {}", path.display());
            }
        } else if config.output.format == OutputFormat::Both {
            println!("\nJSON report:\n{json}");
        } else {
            println!("{json}");
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputFormat {
    Text,
    Json,
    Both,
}

impl OutputFormat {
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            "both" => Ok(Self::Both),
            other => Err(format!("Unknown format '{other}'. Use text|json|both.")),
        }
    }

    fn includes_text(&self) -> bool {
        matches!(self, Self::Text | Self::Both)
    }

    fn includes_json(&self) -> bool {
        matches!(self, Self::Json | Self::Both)
    }
}

impl<'de> Deserialize<'de> for OutputFormat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        OutputFormat::from_str(&s).map_err(DeError::custom)
    }
}

#[derive(Clone)]
struct OutputConfig {
    format: OutputFormat,
    json_out: Option<PathBuf>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Json,
            json_out: None,
        }
    }
}

#[derive(Clone)]
struct RuntimeConfig {
    input_path: PathBuf,
    output: OutputConfig,
    grid_params: GridParams,
}

fn parse_args(program: &str) -> Result<RuntimeConfig, String> {
    let mut args = env::args().skip(1).peekable();
    let mut input_override: Option<PathBuf> = None;
    let mut format_override: Option<OutputFormat> = None;
    let mut json_out_override: Option<PathBuf> = None;
    let mut spacing_override: Option<f32> = None;
    let mut intrinsics_override: Option<Matrix3<f32>> = None;
    let mut config_path: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{}", usage(program));
                std::process::exit(0);
            }
            "--config" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("--config expects a path\n{}", usage(program)))?;
                config_path = Some(PathBuf::from(value));
            }
            "--format" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("--format expects a value\n{}", usage(program)))?;
                format_override = Some(OutputFormat::from_str(&value)?);
            }
            "--json-out" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("--json-out expects a path\n{}", usage(program)))?;
                json_out_override = Some(PathBuf::from(value));
            }
            "--spacing-mm" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("--spacing-mm expects a value\n{}", usage(program)))?;
                let parsed: f32 = value
                    .parse()
                    .map_err(|_| format!("Invalid spacing '{value}'"))?;
                spacing_override = Some(parsed);
            }
            "--intrinsics" => {
                let value = args.next().ok_or_else(|| {
                    format!("--intrinsics expects fx,fy,cx,cy\n{}", usage(program))
                })?;
                intrinsics_override = Some(parse_intrinsics(&value)?);
            }
            _ if arg.starts_with('-') => {
                return Err(format!("Unknown option '{arg}'\n{}", usage(program)));
            }
            _ => {
                if input_override.is_some() {
                    return Err(format!(
                        "Unexpected positional argument '{arg}'\n{}",
                        usage(program)
                    ));
                }
                input_override = Some(PathBuf::from(arg));
            }
        }
    }

    let file_config = if let Some(path) = config_path {
        let contents = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read config {}: {e}", path.display()))?;
        serde_json::from_str::<FileConfig>(&contents)
            .map_err(|e| format!("Failed to parse config {}: {e}", path.display()))?
    } else {
        FileConfig::default()
    };

    let input_path = input_override
        .or_else(|| file_config.input.clone())
        .ok_or_else(|| usage(program))?;

    let mut output = OutputConfig::default();
    if let Some(ref file_output) = file_config.output {
        if let Some(format) = file_output.format {
            output.format = format;
        }
        if let Some(ref path) = file_output.json_out {
            output.json_out = Some(path.clone());
        }
    }
    if let Some(format) = format_override {
        output.format = format;
    }
    if let Some(path) = json_out_override {
        output.json_out = Some(path);
    }

    let mut grid_params = GridParams::default();
    if let Some(ref grid_cfg) = file_config.grid {
        apply_grid_file_config(&mut grid_params, grid_cfg);
    }
    if let Some(spacing) = spacing_override {
        grid_params.spacing_mm = spacing;
    }
    if let Some(intrinsics) = intrinsics_override {
        grid_params.kmtx = intrinsics;
    }

    Ok(RuntimeConfig {
        input_path,
        output,
        grid_params,
    })
}

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} <image.png> [--config config.json] [--format text|json|both] [--json-out report.json] \\\n         [--spacing-mm mm] [--intrinsics fx,fy,cx,cy]\n\n\
Runs the grid detector on a grayscale PNG image and emits diagnostics.\n\
Examples:\n  {program} data/sample.png --format both --json-out sample_report.json\n  {program} board.png --config cfg.json --format text\n"
    )
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
    format: Option<OutputFormat>,
    json_out: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct FileGridConfig {
    pyramid_levels: Option<usize>,
    spacing_mm: Option<f32>,
    intrinsics: Option<IntrinsicsConfig>,
    min_cells: Option<i32>,
    confidence_thresh: Option<f32>,
    enable_refine: Option<bool>,
    refine: Option<FileRefineConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct FileRefineConfig {
    base_mag_thresh: Option<f32>,
    angle_tol_deg: Option<f32>,
    base_min_len: Option<f32>,
    orientation_tol_deg: Option<f32>,
    merge_dist_px: Option<f32>,
    roi_band_px: Option<f32>,
    roi_widen_factor: Option<f32>,
    huber_delta: Option<f32>,
    max_iterations: Option<usize>,
    min_bundle_weight: Option<f32>,
    min_bundles_per_family: Option<usize>,
    restrict_to_roi: Option<bool>,
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
}

fn apply_refine_file_config(params: &mut RefineParams, cfg: &FileRefineConfig) {
    if let Some(v) = cfg.base_mag_thresh {
        params.base_mag_thresh = v;
    }
    if let Some(v) = cfg.angle_tol_deg {
        params.angle_tol_deg = v;
    }
    if let Some(v) = cfg.base_min_len {
        params.base_min_len = v;
    }
    if let Some(v) = cfg.orientation_tol_deg {
        params.orientation_tol_deg = v;
    }
    if let Some(v) = cfg.merge_dist_px {
        params.merge_dist_px = v;
    }
    if let Some(v) = cfg.roi_band_px {
        params.roi_band_px = v;
    }
    if let Some(v) = cfg.roi_widen_factor {
        params.roi_widen_factor = v;
    }
    if let Some(v) = cfg.huber_delta {
        params.huber_delta = v;
    }
    if let Some(v) = cfg.max_iterations {
        params.max_iterations = v;
    }
    if let Some(v) = cfg.min_bundle_weight {
        params.min_bundle_weight = v;
    }
    if let Some(v) = cfg.min_bundles_per_family {
        params.min_bundles_per_family = v;
    }
    if let Some(v) = cfg.restrict_to_roi {
        params.restrict_to_roi = v;
    }
}

fn parse_intrinsics(value: &str) -> Result<Matrix3<f32>, String> {
    let parts: Vec<&str> = value.split(',').collect();
    if parts.len() != 4 {
        return Err(format!("Expected fx,fy,cx,cy but got '{value}'"));
    }
    let fx: f32 = parts[0]
        .trim()
        .parse()
        .map_err(|_| format!("Invalid fx '{}'", parts[0]))?;
    let fy: f32 = parts[1]
        .trim()
        .parse()
        .map_err(|_| format!("Invalid fy '{}'", parts[1]))?;
    let cx: f32 = parts[2]
        .trim()
        .parse()
        .map_err(|_| format!("Invalid cx '{}'", parts[2]))?;
    let cy: f32 = parts[3]
        .trim()
        .parse()
        .map_err(|_| format!("Invalid cy '{}'", parts[3]))?;
    Ok(Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0))
}

fn print_text_summary(detailed: &DetailedResult) {
    let res = &detailed.result;
    println!("Detection summary");
    println!("  found: {}", res.found);
    println!("  confidence: {:.3}", res.confidence);
    println!("  latency_ms: {:.3}", res.latency_ms);
    let h = &res.hmtx;
    println!(
        "  homography:\n    [{:.4} {:.4} {:.4}]\n    [{:.4} {:.4} {:.4}]\n    [{:.4} {:.4} {:.4}]",
        h[(0, 0)],
        h[(0, 1)],
        h[(0, 2)],
        h[(1, 0)],
        h[(1, 1)],
        h[(1, 2)],
        h[(2, 0)],
        h[(2, 1)],
        h[(2, 2)]
    );
    if let Some(pose) = &res.pose {
        println!(
            "  pose.t: [{:.3}, {:.3}, {:.3}]",
            pose.t[0], pose.t[1], pose.t[2]
        );
    } else {
        println!("  pose: unavailable (insufficient confidence)");
    }

    let diag = &detailed.diagnostics;
    println!("\nPyramid (built in {:.3} ms)", diag.pyramid_build_ms);
    for lvl in &diag.pyramid_levels {
        println!(
            "  L{}: {}x{} mean={:.4}",
            lvl.level, lvl.width, lvl.height, lvl.mean_intensity
        );
    }

    match &diag.lsd {
        Some(lsd) => {
            println!(
                "\nLSD stage: segments={} fam_u={} fam_v={} conf={:.3} elapsed_ms={:.3}",
                lsd.segments_total,
                lsd.family_u_count,
                lsd.family_v_count,
                lsd.confidence,
                lsd.elapsed_ms
            );
            println!(
                "  dominant_angles_deg=[{:.1}, {:.1}] sample_count={}",
                lsd.dominant_angles_deg[0],
                lsd.dominant_angles_deg[1],
                lsd.segments_sample.len()
            );
        }
        None => println!("\nLSD stage: no viable hypothesis"),
    }

    match &diag.refinement {
        Some(refine) => {
            println!(
                "\nRefinement: levels={} aggregated_confidence={:.3} inlier_ratio={:.3}",
                refine.levels_used, refine.aggregated_confidence, refine.final_inlier_ratio
            );
            for lvl in &refine.levels {
                println!(
                    "  L{}: {}x{} segs={} bundles={} fam_u={} fam_v={} improvement={} conf={} inliers={}",
                    lvl.level_index,
                    lvl.width,
                    lvl.height,
                    lvl.segments,
                    lvl.bundles,
                    lvl.family_u_count,
                    lvl.family_v_count,
                    format_opt(lvl.improvement),
                    format_opt(lvl.confidence),
                    format_opt(lvl.inlier_ratio),
                );
            }
        }
        None => println!("\nRefinement: disabled or no refinement result"),
    }
}

fn format_opt(val: Option<f32>) -> String {
    val.map(|v| format!("{:.3}", v))
        .unwrap_or_else(|| "-".to_string())
}
