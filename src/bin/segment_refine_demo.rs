//! Demonstration binary for the gradient-based segment refiner.
//!
//! The demo mirrors the early detector pipeline stages:
//! 1. Build an image pyramid.
//! 2. Run LSD on the coarsest level.
//! 3. Build an orientation histogram and assign segment families.
//! 4. Refine the segments down the pyramid using only local gradients.
//! 5. Emit diagnostics identical to the pipeline (pyramid, LSD stage, refinement levels).

use grid_detector::config::segment_refine_demo as seg_cfg;
use grid_detector::detector::{DetectorWorkspace, LevelScaleMap};
use grid_detector::diagnostics::builders::{compute_family_counts, convert_refined_segment};
use grid_detector::diagnostics::{
    LsdStage, PyramidStage, SegmentRefineLevel, SegmentRefineSample, SegmentRefineStage,
    TimingBreakdown,
};
use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::image::{ImageF32, ImageView};
use grid_detector::lsd_vp::{analyze_families, FamilyAssignments};
use grid_detector::pyramid::{Pyramid, PyramidOptions};
use grid_detector::refine::segment::{
    self, PyramidLevel as SegmentGradientLevel, Segment as SegmentSeed,
};
use grid_detector::segments::{lsd_extract_segments, Segment};
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Run a closure while timing its execution and reporting the elapsed time. Should return the same
/// result as the closure.
fn run_with_timer<R, F: FnOnce() -> Result<R, String>>(f: F) -> Result<ResultWithTime<R>, String> {
    let start = Instant::now();
    let result = f();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("Completed in {:.2} ms", elapsed_ms);
    Ok(ResultWithTime { result, elapsed_ms })
}

fn main() {
    let run_perf = run_with_timer(|| {
        if let Err(err) = run() {
            eprintln!("Error: {err}");
            std::process::exit(1);
        } else {
            Ok(())
        }
    });
    println!(
        "Total execution time: {:.2} ms",
        match run_perf {
            Ok(r) => r.elapsed_ms,
            Err(_) => 0.0,
        }
    );
}

struct ResultWithTime<R> {
    result: Result<R, String>,
    elapsed_ms: f64,
}

fn run() -> Result<(), String> {
    let config_path = env::args().nth(1).ok_or_else(usage)?;
    let config = seg_cfg::load_config(Path::new(&config_path))?;

    fs::create_dir_all(&config.output.dir)
        .map_err(|e| format!("Failed to create {}: {e}", config.output.dir.display()))?;

    let total_start = Instant::now();
    let gray_perf = run_with_timer(|| load_grayscale_image(&config.input))?;
    let gray = gray_perf.result?;

    let levels = config.pyramid.levels.max(1);
    let pyramid_opts = PyramidOptions::new(levels).with_blur_levels(config.pyramid.blur_levels);
    let pyr_start = Instant::now();
    let pyramid = Pyramid::build_u8(gray.as_view(), pyramid_opts);
    let pyramid_ms = pyr_start.elapsed().as_secs_f64() * 1000.0;
    save_pyramid_images(&pyramid, &config.output.dir)?;
    let pyramid_stage = PyramidStage::from_pyramid(&pyramid, pyramid_ms);

    let mut workspace = DetectorWorkspace::new();
    workspace.reset(pyramid.levels.len());

    let coarsest_index = pyramid
        .levels
        .len()
        .checked_sub(1)
        .ok_or_else(|| "Pyramid must contain at least one level".to_string())?;
    let lsd_start = Instant::now();

    let scale = 1_f32 / (1 << pyramid.levels.len()) as f32;
    let coarse_level = &pyramid.levels[coarsest_index];
    let lsd_segments = lsd_extract_segments(coarse_level, config.lsd.with_scale(scale));
    let assignments = analyze_families(&lsd_segments, config.lsd.angle_tolerance_deg)
        .map_err(|err| format!("Orientation analysis failed: {err}"))?;
    let lsd_ms = lsd_start.elapsed().as_secs_f64() * 1000.0;
    let lsd_stage = build_lsd_stage(&assignments, lsd_ms);

    let mut current_segments: Vec<Segment> = lsd_segments.clone();
    let initial_segments = current_segments.clone();

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
        let grad = workspace.sobel_gradients(finer_idx, finer_level);
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
        let coarse_level = &pyramid.levels[coarse_idx];
        let scale_map = level_scale_map(coarse_level, finer_level);

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
            samples.push(SegmentRefineSample {
                segment: updated.clone(),
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
    if gray_perf.elapsed_ms > 0.0 {
        timings.push("load", gray_perf.elapsed_ms);
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
        lsd: Some(lsd_stage),
        lsd_segments: Some(initial_segments),
        levels: levels_report,
        timings,
    };

    let report_path = config.output.dir.join("report.json");
    write_json_file(&report_path, &stage)?;

    println!(
        "Segment refinement report written to {}",
        report_path.display()
    );

    Ok(())
}

fn usage() -> String {
    "Usage: segment_refine_demo <config.json>".to_string()
}

fn save_pyramid_images(pyramid: &Pyramid, out_dir: &Path) -> Result<(), String> {
    for (idx, level) in pyramid.levels.iter().enumerate() {
        let path = out_dir.join(format!("pyramid_L{idx}.png"));
        save_grayscale_f32(level, &path)?;
    }
    Ok(())
}

fn build_lsd_stage(assignments: &FamilyAssignments, elapsed_ms: f64) -> LsdStage {
    let dominant_angles_deg = [
        assignments.dominant_angles_rad[0].to_degrees(),
        assignments.dominant_angles_rad[1].to_degrees(),
    ];
    let family_counts = compute_family_counts(&assignments.families);
    LsdStage {
        elapsed_ms,
        confidence: assignments.confidence(),
        dominant_angles_deg,
        family_counts,
        segment_families: assignments.families.clone(),
        sample_ids: Vec::new(),
    }
}

fn level_scale_map(coarse: &ImageF32, fine: &ImageF32) -> LevelScaleMap {
    let sx = if coarse.w == 0 {
        2.0
    } else {
        fine.w as f32 / coarse.w as f32
    };
    let sy = if coarse.h == 0 {
        2.0
    } else {
        fine.h as f32 / coarse.h as f32
    };
    LevelScaleMap::new(sx, sy)
}
