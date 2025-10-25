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

use grid_detector::config::segment_refine_demo as seg_cfg;
use grid_detector::config::segments::LsdConfig;
use grid_detector::edges::grad::{sobel_gradients, Grad};
use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::image::ImageView;
use grid_detector::pyramid::Pyramid;
use grid_detector::refine::segment::{self, ScaleMap, Segment as RefineSegment};
use grid_detector::segments::Segment as LsdSegment;
use grid_detector::segments::{lsd_extract_segments_with_options, LsdOptions};
use serde::Serialize;
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config_path = env::args().nth(1).ok_or_else(usage)?;
    let config = seg_cfg::load_config(Path::new(&config_path))?;

    fs::create_dir_all(&config.output.dir)
        .map_err(|e| format!("Failed to create {}: {e}", config.output.dir.display()))?;

    let total_start = Instant::now();
    let load_start = Instant::now();
    let gray = load_grayscale_image(&config.input)?;
    let load_ms = load_start.elapsed().as_secs_f32() * 1000.0;

    let levels = config.pyramid.levels.max(1);
    let pyramid = if let Some(blur_levels_cfg) = config.pyramid.blur_levels {
        let blur_levels = blur_levels_cfg.min(levels.saturating_sub(1));
        Pyramid::build_u8_with_blur_levels(gray.as_view(), levels, blur_levels)
    } else {
        Pyramid::build_u8(gray.as_view(), levels)
    };
    save_pyramid_images(&pyramid, &config.output.dir)?;

    let gradients = build_gradients(&pyramid);

    let coarsest_index = pyramid
        .levels
        .len()
        .checked_sub(1)
        .ok_or_else(|| "Pyramid must contain at least one level".to_string())?;
    let lsd_start = Instant::now();
    let lsd_segments = detect_coarse_segments(&pyramid.levels[coarsest_index], &config.lsd);
    let lsd_ms = lsd_start.elapsed().as_secs_f32() * 1000.0;

    let lsd_settings = LsdSettings {
        magnitude_threshold: config.lsd.magnitude_threshold,
        angle_tolerance_deg: config.lsd.angle_tolerance_deg,
        min_length: config.lsd.min_length,
        enforce_polarity: config.lsd.enforce_polarity,
        normal_span_limit: if config.lsd.limit_normal_span {
            Some(config.lsd.normal_span_limit_px)
        } else {
            None
        },
    };

    let coarsest_report = build_segments_report_coarse(
        coarsest_index,
        &pyramid.levels[coarsest_index],
        &lsd_segments,
    );
    write_json_file(
        &config
            .output
            .dir
            .join(format!("segments_L{coarsest_index}.json")),
        &coarsest_report,
    )?;

    let mut refine_params = config.refine.resolve();
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
            &config
                .output
                .dir
                .join(format!("segments_L{finer_idx}.json")),
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
        source_image: config.input.display().to_string(),
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
    write_json_file(&config.output.dir.join("performance.json"), &performance)?;

    println!(
        "Artifacts written to {} (levels={}, segments={})",
        config.output.dir.display(),
        pyramid.levels.len(),
        lsd_segments.len(),
    );

    Ok(())
}

// Old CLI parsing removed: configuration now comes from JSON.

fn usage() -> String {
    "Usage: segment_refine_demo <config.json>".to_string()
}

// Legacy CLI helpers removed: config is now provided via JSON file.

fn build_gradients(pyramid: &Pyramid) -> Vec<Grad> {
    pyramid.levels.iter().map(sobel_gradients).collect()
}

fn detect_coarse_segments(
    level: &grid_detector::image::ImageF32,
    lsd_cfg: &LsdConfig,
) -> Vec<LsdSegment> {
    let options = LsdOptions {
        enforce_polarity: lsd_cfg.enforce_polarity,
        normal_span_limit: if lsd_cfg.limit_normal_span {
            Some(lsd_cfg.normal_span_limit_px)
        } else {
            None
        },
    };
    let angle_tol = lsd_cfg.angle_tolerance_deg.to_radians();
    lsd_extract_segments_with_options(
        level,
        lsd_cfg.magnitude_threshold,
        angle_tol,
        lsd_cfg.min_length,
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
