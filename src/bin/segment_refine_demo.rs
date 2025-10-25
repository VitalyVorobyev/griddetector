//! Demonstration binary for the gradient-based segment refiner.
//!
//! The tool loads a grayscale image, detects coarse segments on the top
//! pyramid level, refines them down to the finest resolution, and emits a set
//! of visual and numeric artifacts:
//! - `pyramid_L{idx}.png` for every level produced by the pyramid builder.
//! - `segments_L{idx}.json` describing the segments present after each stage
//!   (including the raw LSD output on the coarsest level).
//! - `performance.json` capturing per-level timings and acceptance ratios.
//!
//! Artifacts enable quick visual inspection (with `tools/plot_coarse_segments.py`)
//! and allow benchmarking different parameter choices without enabling the full
//! grid-detector pipeline.

use grid_detector::edges::grad::{sobel_gradients, Grad};
use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::image::ImageView;
use grid_detector::pyramid::Pyramid;
use grid_detector::refine::segment::{self, RefineParams, ScaleMap, Segment as RefineSegment};
use grid_detector::segments::Segment as LsdSegment;
use grid_detector::segments::{lsd_extract_segments_with_options, LsdOptions};
use serde::Serialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let program = env::args()
        .next()
        .unwrap_or_else(|| "segment_refine_demo".to_string());
    let config = parse_args(&program)?;

    fs::create_dir_all(&config.out_dir)
        .map_err(|e| format!("Failed to create {}: {e}", config.out_dir.display()))?;

    let total_start = Instant::now();
    let load_start = Instant::now();
    let gray = load_grayscale_image(&config.image_path)?;
    let load_ms = load_start.elapsed().as_secs_f32() * 1000.0;

    let pyramid = Pyramid::build_u8(gray.as_view(), config.levels);
    save_pyramid_images(&pyramid, &config.out_dir)?;

    let gradients = build_gradients(&pyramid);

    let coarsest_index = pyramid
        .levels
        .len()
        .checked_sub(1)
        .ok_or_else(|| "Pyramid must contain at least one level".to_string())?;
    let lsd_start = Instant::now();
    let lsd_segments = detect_coarse_segments(&pyramid.levels[coarsest_index], &config);
    let lsd_ms = lsd_start.elapsed().as_secs_f32() * 1000.0;

    let lsd_settings = LsdSettings {
        magnitude_threshold: config.lsd_mag,
        angle_tolerance_deg: config.lsd_angle_deg,
        min_length: config.lsd_min_len,
        enforce_polarity: config.lsd_enforce_polarity,
        normal_span_limit: config.lsd_span_limit,
    };

    let coarsest_report = build_segments_report_coarse(
        coarsest_index,
        &pyramid.levels[coarsest_index],
        &lsd_segments,
    );
    write_json_file(
        &config
            .out_dir
            .join(format!("segments_L{coarsest_index}.json")),
        &coarsest_report,
    )?;

    let mut refine_params = build_refine_params(&config);
    // Ensure the normal search spacing stays positive even if overridden badly.
    if refine_params.delta_s <= 0.0 {
        refine_params.delta_s = 0.5;
    }
    if refine_params.delta_t <= 0.0 {
        refine_params.delta_t = 0.25;
    }

    let mut level_stats = Vec::new();
    let mut current_segments: Vec<RefineSegment> = lsd_segments
        .iter()
        .map(|seg| RefineSegment {
            p0: seg.p0,
            p1: seg.p1,
        })
        .collect();

    let mut refine_total_ms = 0.0f32;
    for coarse_idx in (1..pyramid.levels.len()).rev() {
        let finer_idx = coarse_idx - 1;
        let refine_start = Instant::now();
        let finer_level = &pyramid.levels[finer_idx];
        let grad = &gradients[finer_idx];
        let lvl_grad =
            segment::PyramidLevel {
                width: finer_level.w,
                height: finer_level.h,
                gx: grad.gx.as_slice().ok_or_else(|| {
                    format!("Gradient buffer at level {finer_idx} is not contiguous")
                })?,
                gy: grad.gy.as_slice().ok_or_else(|| {
                    format!("Gradient buffer at level {finer_idx} is not contiguous")
                })?,
            };
        let scale_map = LevelScaleMap::from_levels(&pyramid.levels[coarse_idx], finer_level);

        let mut refined_segments = Vec::with_capacity(current_segments.len());
        let mut records = Vec::with_capacity(current_segments.len());
        let mut accepted = 0usize;
        let mut score_sum = 0.0f32;

        for seg in &current_segments {
            let result = segment::refine_segment(&lvl_grad, *seg, &scale_map, &refine_params);
            if result.ok {
                accepted += 1;
                score_sum += result.score;
            }
            refined_segments.push(result.seg);
            records.push(SegmentRecord {
                p0: result.seg.p0,
                p1: result.seg.p1,
                length: result.seg.length(),
                score: Some(result.score),
                ok: Some(result.ok),
                inliers: Some(result.inliers),
                total: Some(result.total),
            });
        }

        let elapsed_ms = refine_start.elapsed().as_secs_f32() * 1000.0;
        refine_total_ms += elapsed_ms;
        let avg_score = if accepted > 0 {
            Some(score_sum / accepted as f32)
        } else {
            None
        };
        let acceptance_ratio = if current_segments.is_empty() {
            None
        } else {
            Some(accepted as f32 / current_segments.len() as f32)
        };

        let segment_file = SegmentsFile {
            level_index: finer_idx,
            width: finer_level.w,
            height: finer_level.h,
            segments: records,
            accepted: Some(accepted),
            acceptance_ratio,
        };
        write_json_file(
            &config.out_dir.join(format!("segments_L{finer_idx}.json")),
            &segment_file,
        )?;

        level_stats.push(LevelReport {
            coarse_level: coarse_idx,
            finer_level: finer_idx,
            segments_in: current_segments.len(),
            accepted,
            refine_ms: elapsed_ms,
            avg_score,
        });

        current_segments = refined_segments;
    }

    let total_ms = total_start.elapsed().as_secs_f32() * 1000.0;
    let performance = PerformanceReport {
        source_image: config.image_path.display().to_string(),
        image_width: gray.width(),
        image_height: gray.height(),
        pyramid_levels: pyramid.levels.len(),
        lsd: lsd_settings,
        total_segments: lsd_segments.len(),
        final_segments: current_segments.len(),
        timings_ms: TimingSummary {
            load: load_ms,
            lsd: lsd_ms,
            refine_total: refine_total_ms,
            total: total_ms,
        },
        levels: level_stats,
    };
    write_json_file(&config.out_dir.join("performance.json"), &performance)?;

    println!(
        "Artifacts written to {} (levels={}, segments={})",
        config.out_dir.display(),
        pyramid.levels.len(),
        lsd_segments.len(),
    );

    Ok(())
}

#[derive(Debug, Clone)]
struct DemoConfig {
    image_path: PathBuf,
    out_dir: PathBuf,
    levels: usize,
    lsd_mag: f32,
    lsd_angle_deg: f32,
    lsd_min_len: f32,
    lsd_enforce_polarity: bool,
    lsd_span_limit: Option<f32>,
    overrides: SegmentOverrides,
}

#[derive(Debug, Clone, Default)]
struct SegmentOverrides {
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

fn parse_args(program: &str) -> Result<DemoConfig, String> {
    let mut args = env::args().skip(1).peekable();
    let mut image_path: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut levels = 4usize;
    let mut lsd_mag = 0.08f32;
    let mut lsd_angle = 22.5f32;
    let mut lsd_min_len = 6.0f32;
    let mut lsd_enforce_polarity = false;
    let mut lsd_span_limit: Option<f32> = None;
    let mut overrides = SegmentOverrides::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{}", usage(program));
                std::process::exit(0);
            }
            "--image" => {
                image_path = Some(require_value(&mut args, "--image", program)?);
            }
            "--out-dir" => {
                out_dir = Some(require_value(&mut args, "--out-dir", program)?);
            }
            "--levels" => {
                levels = parse_number(&mut args, "--levels", program)?;
                if levels == 0 {
                    return Err("--levels must be >= 1".to_string());
                }
            }
            "--lsd-mag" => {
                lsd_mag = parse_number(&mut args, "--lsd-mag", program)?;
            }
            "--lsd-angle" => {
                lsd_angle = parse_number(&mut args, "--lsd-angle", program)?;
            }
            "--lsd-min-len" => {
                lsd_min_len = parse_number(&mut args, "--lsd-min-len", program)?;
            }
            "--lsd-enforce-polarity" => {
                lsd_enforce_polarity = true;
            }
            "--lsd-span-limit" => {
                lsd_span_limit = Some(parse_number(&mut args, "--lsd-span-limit", program)?);
            }
            "--seg-delta-s" => {
                overrides.delta_s = Some(parse_number(&mut args, "--seg-delta-s", program)?);
            }
            "--seg-w-perp" => {
                overrides.w_perp = Some(parse_number(&mut args, "--seg-w-perp", program)?);
            }
            "--seg-delta-t" => {
                overrides.delta_t = Some(parse_number(&mut args, "--seg-delta-t", program)?);
            }
            "--seg-pad" => {
                overrides.pad = Some(parse_number(&mut args, "--seg-pad", program)?);
            }
            "--seg-tau-mag" => {
                overrides.tau_mag = Some(parse_number(&mut args, "--seg-tau-mag", program)?);
            }
            "--seg-tau-ori" => {
                overrides.tau_ori_deg = Some(parse_number(&mut args, "--seg-tau-ori", program)?);
            }
            "--seg-huber" => {
                overrides.huber_delta = Some(parse_number(&mut args, "--seg-huber", program)?);
            }
            "--seg-max-iters" => {
                overrides.max_iters = Some(parse_number(&mut args, "--seg-max-iters", program)?);
            }
            "--seg-min-frac" => {
                overrides.min_inlier_frac =
                    Some(parse_number(&mut args, "--seg-min-frac", program)?);
            }
            other if other.starts_with('-') => {
                return Err(format!("Unknown option '{other}'\n{}", usage(program)));
            }
            other => {
                if image_path.is_none() {
                    image_path = Some(PathBuf::from(other));
                } else if out_dir.is_none() {
                    out_dir = Some(PathBuf::from(other));
                } else {
                    return Err(format!(
                        "Unexpected positional argument '{other}'\n{}",
                        usage(program)
                    ));
                }
            }
        }
    }

    let image_path = image_path.ok_or_else(|| usage(program))?;
    let out_dir = out_dir.unwrap_or_else(|| PathBuf::from("out/segment_demo"));

    Ok(DemoConfig {
        image_path,
        out_dir,
        levels,
        lsd_mag,
        lsd_angle_deg: lsd_angle,
        lsd_min_len,
        lsd_enforce_polarity,
        lsd_span_limit,
        overrides,
    })
}

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} --image input.png [options]\n\
Options:\n\
  --out-dir PATH           Output directory (default: out/segment_demo)\n\
  --levels N               Pyramid levels (default: 4)\n\
  --lsd-mag F              LSD magnitude threshold (default: 0.08)\n\
  --lsd-angle F            LSD angle tolerance in degrees (default: 22.5)\n\
  --lsd-min-len F          LSD minimum segment length (default: 6.0)\n\
  --lsd-enforce-polarity   Require consistent gradient polarity\n\
  --lsd-span-limit F       Maximum normal span for LSD (optional)\n\
  --seg-delta-s F          Override refinement delta_s\n\
  --seg-w-perp F           Override refinement w_perp\n\
  --seg-delta-t F          Override refinement delta_t\n\
  --seg-pad F              Override refinement pad\n\
  --seg-tau-mag F          Override refinement tau_mag\n\
  --seg-tau-ori F          Override refinement tau_ori_deg\n\
  --seg-huber F            Override refinement huber_delta\n\
  --seg-max-iters N        Override refinement max_iters\n\
  --seg-min-frac F         Override refinement min_inlier_frac"
    )
}

fn require_value(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
    program: &str,
) -> Result<PathBuf, String> {
    args.next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("{flag} expects a value\n{}", usage(program)))
}

fn parse_number<T>(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
    program: &str,
) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    args.next()
        .ok_or_else(|| format!("{flag} expects a value\n{}", usage(program)))?
        .parse::<T>()
        .map_err(|e| format!("Failed to parse {flag}: {e}"))
}

fn build_refine_params(config: &DemoConfig) -> RefineParams {
    let mut params = RefineParams::default();
    if let Some(v) = config.overrides.delta_s {
        params.delta_s = v;
    }
    if let Some(v) = config.overrides.w_perp {
        params.w_perp = v;
    }
    if let Some(v) = config.overrides.delta_t {
        params.delta_t = v;
    }
    if let Some(v) = config.overrides.pad {
        params.pad = v;
    }
    if let Some(v) = config.overrides.tau_mag {
        params.tau_mag = v;
    }
    if let Some(v) = config.overrides.tau_ori_deg {
        params.tau_ori_deg = v;
    }
    if let Some(v) = config.overrides.huber_delta {
        params.huber_delta = v;
    }
    if let Some(v) = config.overrides.max_iters {
        params.max_iters = v;
    }
    if let Some(v) = config.overrides.min_inlier_frac {
        params.min_inlier_frac = v;
    }
    params
}

fn build_gradients(pyramid: &Pyramid) -> Vec<Grad> {
    pyramid
        .levels
        .iter()
        .map(|level| sobel_gradients(level))
        .collect()
}

fn detect_coarse_segments(
    level: &grid_detector::image::ImageF32,
    config: &DemoConfig,
) -> Vec<LsdSegment> {
    let options = LsdOptions {
        enforce_polarity: config.lsd_enforce_polarity,
        normal_span_limit: config.lsd_span_limit,
    };
    let angle_tol = config.lsd_angle_deg.to_radians();
    lsd_extract_segments_with_options(
        level,
        config.lsd_mag,
        angle_tol,
        config.lsd_min_len,
        options,
    )
}

fn build_segments_report_coarse(
    level_index: usize,
    level: &grid_detector::image::ImageF32,
    segments: &[LsdSegment],
) -> SegmentsFile {
    let records = segments
        .iter()
        .map(|seg| SegmentRecord {
            p0: seg.p0,
            p1: seg.p1,
            length: seg.len,
            score: None,
            ok: None,
            inliers: None,
            total: None,
        })
        .collect();
    SegmentsFile {
        level_index,
        width: level.w,
        height: level.h,
        segments: records,
        accepted: None,
        acceptance_ratio: None,
    }
}

fn save_pyramid_images(pyramid: &Pyramid, out_dir: &Path) -> Result<(), String> {
    for (idx, level) in pyramid.levels.iter().enumerate() {
        let path = out_dir.join(format!("pyramid_L{idx}.png"));
        save_grayscale_f32(level, &path)?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct LevelScaleMap {
    sx: f32,
    sy: f32,
}

impl LevelScaleMap {
    fn from_levels(
        coarse: &grid_detector::image::ImageF32,
        fine: &grid_detector::image::ImageF32,
    ) -> Self {
        let sx = if coarse.w == 0 {
            1.0
        } else {
            fine.w as f32 / coarse.w as f32
        };
        let sy = if coarse.h == 0 {
            1.0
        } else {
            fine.h as f32 / coarse.h as f32
        };
        Self { sx, sy }
    }
}

impl ScaleMap for LevelScaleMap {
    fn up(&self, p_coarse: [f32; 2]) -> [f32; 2] {
        [p_coarse[0] * self.sx, p_coarse[1] * self.sy]
    }
}

#[derive(Serialize)]
struct SegmentRecord {
    p0: [f32; 2],
    p1: [f32; 2],
    length: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ok: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inliers: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total: Option<usize>,
}

#[derive(Serialize)]
struct SegmentsFile {
    level_index: usize,
    width: usize,
    height: usize,
    segments: Vec<SegmentRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    accepted: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    acceptance_ratio: Option<f32>,
}

#[derive(Serialize)]
struct LevelReport {
    coarse_level: usize,
    finer_level: usize,
    segments_in: usize,
    accepted: usize,
    refine_ms: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_score: Option<f32>,
}

#[derive(Serialize)]
struct TimingSummary {
    load: f32,
    lsd: f32,
    refine_total: f32,
    total: f32,
}

#[derive(Serialize)]
struct PerformanceReport {
    source_image: String,
    image_width: usize,
    image_height: usize,
    pyramid_levels: usize,
    lsd: LsdSettings,
    total_segments: usize,
    final_segments: usize,
    timings_ms: TimingSummary,
    levels: Vec<LevelReport>,
}

#[derive(Serialize)]
struct LsdSettings {
    magnitude_threshold: f32,
    angle_tolerance_deg: f32,
    min_length: f32,
    enforce_polarity: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    normal_span_limit: Option<f32>,
}
