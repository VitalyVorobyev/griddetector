//! Demonstration binary for the gradient-based segment refiner.
//!
//! The demo mirrors the early detector pipeline stages:
//! 1. Build an image pyramid.
//! 2. Run LSD on the coarsest level.
//! 3. Build an orientation histogram and assign segment families.
//! 4. Refine the segments down the pyramid using only local gradients.
//! 5. Emit diagnostics identical to the pipeline (pyramid, LSD stage, refinement levels).

use grid_detector::detector::{DetectorWorkspace, LevelScaleMap};
use grid_detector::diagnostics::builders::compute_family_counts;
use grid_detector::diagnostics::{
    LsdStage, PyramidStage, SegmentRefineLevel, SegmentRefineSample, SegmentRefineStage,
    TimingBreakdown,
};
use grid_detector::image::io::{
    load_grayscale_image, save_grayscale_f32, write_json_file, GrayImageU8,
};
use grid_detector::image::ImageF32;
use grid_detector::lsd_vp::{analyze_families, FamilyAssignments};
use grid_detector::pyramid::{Pyramid, PyramidOptions};
use grid_detector::refine::segment::roi::roi_to_int_bounds;
use grid_detector::refine::segment::RefineOptions as SegmentRefineParams;
use grid_detector::segments::LsdOptions;

#[cfg(feature = "profile_refine")]
use grid_detector::refine::segment::take_profile;
use grid_detector::refine::segment::{self, PyramidLevel as SegmentGradientLevel, ScaleMap};
use grid_detector::segments::{lsd_extract_segments_coarse, Segment};
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Deserialize)]
pub struct SegmentRefineDemoConfig {
    #[serde(rename = "input")]
    pub input: PathBuf,
    pub pyramid: PyramidOptions,
    #[serde(default)]
    pub lsd: LsdOptions,
    #[serde(default)]
    pub refine: SegmentRefineParams,
    #[serde(default)]
    pub performance_mode: bool,
    pub output: SegmentRefineDemoOutputConfig,
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


fn main() {
    run().unwrap_or_else(|err| {
        eprintln!("Error: {err}");
        std::process::exit(1);
    })
}

fn run() -> Result<(), String> {
    let config = load_config_from_args()?;
    ensure_output_dir(&config)?;
    let diagnostics_enabled = !config.performance_mode;

    let ResultWithTime {
        result: gray,
        elapsed_ms: load_ms,
    } = run_with_timer(|| load_grayscale_image(&config.input))?;

    let total_start = Instant::now();

    let PyramidBuildResult {
        pyramid,
        elapsed_ms: pyr_ms,
    } = self.build_pyramid(gray);

    let LsdResult {
        segments: segments,
        elapsed_ms: lsd_ms,
    } = lsd_extract_segments_coarse(&pyramid, self.params.lsd_params);

    let mut workspace = DetectorWorkspace::new();
    workspace.reset(pyramid.levels.len());

    let (levels_report, refine_total_ms) = run_refinement_levels(
        &pyramid,
        &mut workspace,
        &segments,
        &config.refine,
        diagnostics_enabled,
    )?;

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let mut timings = diagnostics_enabled.then(|| TimingBreakdown::with_total(total_ms));
    if load_ms > 0.0 {
        if let Some(t) = timings.as_mut() {
            t.push("load", load_ms);
        }
    }
    if pyramid_ms > 0.0 {
        if let Some(t) = timings.as_mut() {
            t.push("pyramid", pyramid_ms);
        }
        println!("Pyramid built in {:.2} ms", pyramid_ms);
    }
    if lsd_elapsed_ms > 0.0 {
        if let Some(t) = timings.as_mut() {
            t.push("lsd", lsd_elapsed_ms);
        }
        println!(
            "LSD detected {} segments in {:.2} ms",
            lsd_segments.len(),
            lsd_elapsed_ms
        );
    }
    let refined_levels = pyramid.levels.len().saturating_sub(1);
    if refine_total_ms > 0.0 {
        if let Some(t) = timings.as_mut() {
            t.push("segment_refine", refine_total_ms);
        }
        println!(
            "Refined {} segments over {} levels in {:.2} ms",
            lsd_segments.len(),
            refined_levels,
            refine_total_ms
        );
    }

    if diagnostics_enabled {
        let stage = SegmentRefineStage {
            lsd: lsd_stage,
            lsd_segments: Some(lsd_segments.clone()),
            levels: levels_report,
            timings: timings.expect("timings missing"),
        };

        let report_path = config.output.dir.join("report.json");
        write_json_file(&report_path, &stage)?;

        println!(
            "Segment refinement report written to {}",
            report_path.display()
        );
    } else {
        println!("Performance mode enabled: skipped diagnostics output");
    }

    println!("Total execution time: {:.2} ms", total_ms);

    Ok(())
}

fn usage() -> String {
    "Usage: segment_refine_demo <config.json>".to_string()
}

fn load_config_from_args() -> Result<SegmentRefineDemoConfig, String> {
    let config_path = env::args().nth(1).ok_or_else(usage)?;
    load_config(Path::new(&config_path))
}

fn ensure_output_dir(config: &SegmentRefineDemoConfig) -> Result<(), String> {
    fs::create_dir_all(&config.output.dir)
        .map_err(|e| format!("Failed to create {}: {e}", config.output.dir.display()))
}

fn save_pyramid_images(pyramid: &Pyramid, out_dir: &Path) -> Result<(), String> {
    for (idx, level) in pyramid.levels.iter().enumerate() {
        let path = out_dir.join(format!("pyramid_L{idx}.png"));
        save_grayscale_f32(level, &path)?;
    }
    Ok(())
}

fn run_lsd_demo(
    pyramid: &Pyramid,
    config: &SegmentRefineDemoConfig,
) -> Result<(Vec<Segment>, Option<LsdStage>, f64), String> {
    let coarsest_index = pyramid
        .levels
        .len()
        .checked_sub(1)
        .ok_or_else(|| "Pyramid must contain at least one level".to_string())?;
    let scale = 1_f32 / (1 << pyramid.levels.len()) as f32;
    let coarse_level = &pyramid.levels[coarsest_index];
    let ResultWithTime {
        result: (segments, assignments),
        elapsed_ms,
    } = run_with_timer(|| {
        let segments = lsd_extract_segments(coarse_level, config.lsd.with_scale(scale));
        if config.performance_mode {
            Ok((segments, None))
        } else {
            let assignments = analyze_families(&segments, config.lsd.angle_tolerance_deg)
                .map_err(|err| format!("Orientation analysis failed: {err}"))?;
            Ok((segments, Some(assignments)))
        }
    })?;
    let lsd_stage = assignments.map(|a| build_lsd_stage(&a, elapsed_ms));
    Ok((segments, lsd_stage, elapsed_ms))
}

fn run_refinement_levels(
    pyramid: &Pyramid,
    workspace: &mut DetectorWorkspace,
    source_segments: &[Segment],
    refine_params: &segment::RefineOptions,
    collect_levels: bool,
) -> Result<(Vec<SegmentRefineLevel>, f64), String> {
    let mut levels_report: Vec<SegmentRefineLevel> = Vec::new();
    let mut refine_total_ms = 0.0f64;
    let mut current_segments: Vec<Segment> = source_segments.to_vec();
    let full_width = pyramid.levels.first().map(|lvl| lvl.w).unwrap_or(0);

    for coarse_idx in (1..pyramid.levels.len()).rev() {
        let finer_idx = coarse_idx - 1;
        let refine_start = Instant::now();
        let finer_level = &pyramid.levels[finer_idx];
        let coarse_level = &pyramid.levels[coarse_idx];
        let scale_map = level_scale_map(coarse_level, finer_level);
        let level_params = refine_params.for_level(full_width, finer_level.w);
        let mut refined_segments = Vec::with_capacity(current_segments.len());
        let mut samples = collect_levels.then(|| Vec::with_capacity(current_segments.len()));
        let mut accepted = 0usize;
        let mut score_sum = 0.0f32;

        for seg in &current_segments {
            let grad_view = match segment::segment_roi_from_points(
                scale_map.up(seg.p0),
                scale_map.up(seg.p1),
                level_params.pad,
                finer_level.w,
                finer_level.h,
            )
            .and_then(|roi| roi_to_int_bounds(&roi, finer_level.w, finer_level.h))
            {
                Some(bounds) => workspace.scharr_gradients_window(finer_idx, finer_level, &bounds),
                None => workspace.scharr_gradients_full(finer_idx, finer_level),
            };
            let grad_level = SegmentGradientLevel {
                width: finer_level.w,
                height: finer_level.h,
                origin_x: grad_view.origin_x,
                origin_y: grad_view.origin_y,
                tile_width: grad_view.tile_width,
                tile_height: grad_view.tile_height,
                gx: grad_view.gx,
                gy: grad_view.gy,
                level_index: finer_idx,
            };
            let result = segment::refine_segment(&grad_level, seg, &scale_map, &level_params);
            let ok = result.ok;
            if ok {
                accepted += 1;
                if result.score.is_finite() {
                    score_sum += result.score;
                }
            }
            let score = result.score.is_finite().then_some(result.score);
            let inliers = (result.inliers > 0).then_some(result.inliers);
            let total = (result.total > 0).then_some(result.total);
            let updated = result.seg;
            if let Some(samples) = samples.as_mut() {
                samples.push(SegmentRefineSample {
                    segment: updated.clone(),
                    score,
                    ok: Some(ok),
                    inliers,
                    total,
                });
            }
            refined_segments.push(updated);
        }

        let elapsed_ms = refine_start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "Refinement for level {} took {:.2} ms",
            finer_idx, elapsed_ms
        );
        refine_total_ms += elapsed_ms;
        let acceptance_ratio = (!current_segments.is_empty())
            .then_some(accepted as f32 / current_segments.len() as f32);
        let avg_score = (accepted > 0).then_some(score_sum / accepted as f32);

        if let Some(samples) = samples {
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
        }

        current_segments = refined_segments;

        #[cfg(feature = "profile_refine")]
        dump_refine_profile(workspace);
    }

    Ok((levels_report, refine_total_ms))
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
        used_gradient_refinement: false,
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

#[cfg(feature = "profile_refine")]
fn dump_refine_profile(workspace: &DetectorWorkspace) {
    let profile = take_profile();
    if profile.is_empty() {
        return;
    }
    println!("Segment refine profile:");
    for entry in profile {
        if entry.roi_count == 0 && entry.bilinear_samples == 0 {
            continue;
        }
        let avg_roi = if entry.roi_count > 0 {
            entry.roi_area_px / entry.roi_count as f64
        } else {
            0.0
        };
        let grad_ms = workspace.gradient_time_ms(entry.level_index).unwrap_or(0.0);
        println!(
            "  L{}: grad_ms={:.2} roi_count={} avg_roi_px={:.1} bilinear_samples={}",
            entry.level_index, grad_ms, entry.roi_count, avg_roi, entry.bilinear_samples
        );
    }
}
