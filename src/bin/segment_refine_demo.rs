//! Demonstration binary for the gradient-based segment refiner.
//!
//! The tool loads a grayscale image, builds a pyramid, extracts coarse-level LSD
//! segments, refines them down the hierarchy, and emits a single JSON report
//! describing the refinement stages. Pyramid levels are saved as PNG files for
//! quick visual inspection.

use grid_detector::config::segment_refine_demo as seg_cfg;
use grid_detector::config::segments::LsdConfig;
use grid_detector::diagnostics::builders::convert_refined_segment;
use grid_detector::diagnostics::{
    PyramidStage, SegmentDescriptor, SegmentId, SegmentRefineLevel, SegmentRefineSample,
    SegmentRefineStage, TimingBreakdown,
};
use grid_detector::edges::grad::{sobel_gradients, Grad};
use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::image::ImageView;
use grid_detector::pyramid::{Pyramid, PyramidOptions};
use grid_detector::refine::segment::{
    self, PyramidLevel as SegmentGradientLevel, ScaleMap, Segment as SegmentSeed,
};
use grid_detector::segments::Segment;
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
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let levels = config.pyramid.levels.max(1);
    let pyramid_opts = if let Some(blur_levels_cfg) = config.pyramid.blur_levels {
        let blur_levels = blur_levels_cfg.min(levels.saturating_sub(1));
        PyramidOptions::new(levels).with_blur_levels(Some(blur_levels))
    } else {
        PyramidOptions::new(levels)
    };
    let pyr_start = Instant::now();
    let pyramid = Pyramid::build_u8(gray.as_view(), pyramid_opts);
    let pyramid_ms = pyr_start.elapsed().as_secs_f64() * 1000.0;
    save_pyramid_images(&pyramid, &config.output.dir)?;
    let pyramid_stage = PyramidStage::from_pyramid(&pyramid, pyramid_ms);

    let gradients = build_gradients(&pyramid);

    let coarsest_index = pyramid
        .levels
        .len()
        .checked_sub(1)
        .ok_or_else(|| "Pyramid must contain at least one level".to_string())?;
    let lsd_start = Instant::now();
    let lsd_segments = detect_coarse_segments(&pyramid.levels[coarsest_index], &config.lsd);
    let lsd_ms = lsd_start.elapsed().as_secs_f64() * 1000.0;

    let mut current_segments: Vec<Segment> = lsd_segments.clone();
    let mut segment_id_counter: u32 = 0;
    let mut initial_descriptors = Vec::with_capacity(current_segments.len());
    for seg in &current_segments {
        initial_descriptors.push(SegmentDescriptor::from_segment(
            SegmentId(segment_id_counter),
            seg,
        ));
        segment_id_counter += 1;
    }

    let mut refine_params = config.refine.resolve();
    if refine_params.delta_s <= 0.0 {
        refine_params.delta_s = 0.5;
    }
    if refine_params.delta_t <= 0.0 {
        refine_params.delta_t = 0.25;
    }

    let mut levels_report: Vec<SegmentRefineLevel> = Vec::new();
    let mut refine_total_ms = 0.0f64;

    for coarse_idx in (1..pyramid.levels.len()).rev() {
        let finer_idx = coarse_idx - 1;
        let refine_start = Instant::now();
        let finer_level = &pyramid.levels[finer_idx];
        let grad = &gradients[finer_idx];
        let gx = grad
            .gx
            .as_slice()
            .ok_or_else(|| format!("Gradient buffer at level {finer_idx} is not contiguous"))?;
        let gy = grad
            .gy
            .as_slice()
            .ok_or_else(|| format!("Gradient buffer at level {finer_idx} is not contiguous"))?;
        let grad_level = SegmentGradientLevel {
            width: finer_level.w,
            height: finer_level.h,
            gx,
            gy,
        };
        let scale_map = LevelScaleMap::from_levels(&pyramid.levels[coarse_idx], finer_level);

        let mut refined_segments = Vec::with_capacity(current_segments.len());
        let mut samples = Vec::with_capacity(current_segments.len());
        let mut accepted = 0usize;
        let mut score_sum = 0.0f32;

        for seg in &current_segments {
            let seed = SegmentSeed {
                p0: seg.p0,
                p1: seg.p1,
            };
            let result = segment::refine_segment(&grad_level, seed, &scale_map, &refine_params);
            let ok = result.ok;
            if ok {
                accepted += 1;
                if result.score.is_finite() {
                    score_sum += result.score;
                }
            }
            let score = if result.score.is_finite() {
                Some(result.score)
            } else {
                None
            };
            let inliers = if result.inliers > 0 {
                Some(result.inliers)
            } else {
                None
            };
            let total = if result.total > 0 {
                Some(result.total)
            } else {
                None
            };
            let updated = convert_refined_segment(seg, result);
            let descriptor =
                SegmentDescriptor::from_segment(SegmentId(segment_id_counter), &updated);
            segment_id_counter += 1;
            samples.push(SegmentRefineSample {
                segment: descriptor,
                score,
                ok: Some(ok),
                inliers,
                total,
            });
            refined_segments.push(updated);
        }

        let elapsed_ms = refine_start.elapsed().as_secs_f64() * 1000.0;
        refine_total_ms += elapsed_ms;
        let acceptance_ratio = if current_segments.is_empty() {
            None
        } else {
            Some(accepted as f32 / current_segments.len() as f32)
        };
        let avg_score = if accepted > 0 {
            Some(score_sum / accepted as f32)
        } else {
            None
        };

        levels_report.push(SegmentRefineLevel {
            coarse_level: coarse_idx,
            finer_level: finer_idx,
            elapsed_ms,
            segments_in: current_segments.len(),
            accepted,
            acceptance_ratio,
            avg_score,
            results: samples,
        });

        current_segments = refined_segments;
    }

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let mut timings = TimingBreakdown::with_total(total_ms);
    if load_ms > 0.0 {
        timings.push("load", load_ms);
    }
    if pyramid_ms > 0.0 {
        timings.push("pyramid", pyramid_ms);
    }
    if lsd_ms > 0.0 {
        timings.push("lsd", lsd_ms);
    }
    if refine_total_ms > 0.0 {
        timings.push("segment_refine", refine_total_ms);
    }

    let stage = SegmentRefineStage {
        pyramid: pyramid_stage,
        lsd_segments: Some(initial_descriptors),
        levels: levels_report,
        timings,
    };

    let output = SegmentRefineDemoOutput {
        image_width: gray.width(),
        image_height: gray.height(),
        stage,
    };
    write_json_file(&config.output.dir.join("report.json"), &output)?;

    println!(
        "Segment refinement report written to {}",
        config.output.dir.join("report.json").display()
    );

    Ok(())
}

fn usage() -> String {
    "Usage: segment_refine_demo <config.json>".to_string()
}

fn build_gradients(pyramid: &Pyramid) -> Vec<Grad> {
    pyramid.levels.iter().map(sobel_gradients).collect()
}

fn detect_coarse_segments(
    level: &grid_detector::image::ImageF32,
    lsd_cfg: &LsdConfig,
) -> Vec<Segment> {
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

fn save_pyramid_images(pyramid: &Pyramid, out_dir: &Path) -> Result<(), String> {
    for (idx, level) in pyramid.levels.iter().enumerate() {
        let path = out_dir.join(format!("pyramid_L{idx}.png"));
        save_grayscale_f32(level, &path)?;
    }
    Ok(())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SegmentRefineDemoOutput {
    image_width: usize,
    image_height: usize,
    stage: SegmentRefineStage,
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
