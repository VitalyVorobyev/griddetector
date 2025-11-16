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
use grid_detector::refine::segment;
use grid_detector::refine::segment::driver::{
    refine_segments_between_levels, ParallelRefineOptions,
};
#[cfg(feature = "profile_refine")]
use grid_detector::refine::segment::take_profile;
use grid_detector::segments::{lsd_extract_segments, Segment};
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Run a closure while timing its execution and reporting the elapsed time. Should return the same
/// result as the closure.
fn run_with_timer<R, F: FnOnce() -> Result<R, String>>(f: F) -> Result<ResultWithTime<R>, String> {
    let start = Instant::now();
    let result = f()?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(ResultWithTime { result, elapsed_ms })
}

fn main() {
    run().unwrap_or_else(|err| {
        eprintln!("Error: {err}");
        std::process::exit(1);
    })
}

struct ResultWithTime<R> {
    result: R,
    elapsed_ms: f64,
}

#[derive(Clone, Debug)]
struct LevelTiming {
    coarse_level: usize,
    finer_level: usize,
    elapsed_ms: f64,
    segments_in: usize,
}

#[derive(Clone, Debug)]
struct RunTimings {
    total_ms: f64,
    level_timings: Vec<LevelTiming>,
}

struct RefineRunResult {
    levels_report: Vec<SegmentRefineLevel>,
    timings: RunTimings,
}

fn run() -> Result<(), String> {
    let config = load_config_from_args()?;
    ensure_output_dir(&config)?;
    configure_thread_pool(config.timing.max_threads);
    let diagnostics_enabled = !config.performance_mode;

    let ResultWithTime {
        result: gray,
        elapsed_ms: load_ms,
    } = run_with_timer(|| load_grayscale_image(&config.input))?;

    let total_start = Instant::now();

    let (pyramid, pyramid_stage, pyramid_ms) = build_pyramid_stage(&gray, &config)?;
    if diagnostics_enabled {
        save_pyramid_images(&pyramid, &config.output.dir)?;
    }

    let mut workspace = DetectorWorkspace::new();
    let (lsd_segments, lsd_stage, lsd_elapsed_ms) = run_lsd_demo(&pyramid, &config)?;
    let refine_params = config.refine.resolve();
    let parallel_opts = config.timing.parallel_options();
    let warmup_runs = config.timing.warmup_runs();
    let measurement_runs = config.timing.measurement_runs();
    let total_runs = warmup_runs + measurement_runs;
    let mut measurement_timings: Vec<RunTimings> = Vec::new();
    let mut collected_levels = Vec::new();

    for run_idx in 0..total_runs {
        workspace.reset(pyramid.levels.len());
        let is_measurement = run_idx >= warmup_runs;
        let is_last_run = run_idx == total_runs.saturating_sub(1);
        let collect_levels = diagnostics_enabled && is_last_run;
        let log_timings = is_last_run || measurement_runs == 1;
        let RefineRunResult {
            levels_report,
            timings,
        } = run_refinement_levels(
            &pyramid,
            &mut workspace,
            &lsd_segments,
            &refine_params,
            collect_levels,
            log_timings,
            parallel_opts,
        )?;
        if collect_levels {
            collected_levels = levels_report;
        }
        if is_measurement {
            measurement_timings.push(timings);
        }
    }

    let refine_total_ms = if measurement_timings.is_empty() {
        0.0
    } else {
        measurement_timings.iter().map(|t| t.total_ms).sum::<f64>()
            / (measurement_timings.len() as f64)
    };

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
        if measurement_timings.len() > 1 {
            println!(
                "Refined {} segments over {} levels: avg {:.2} ms over {} runs",
                lsd_segments.len(),
                refined_levels,
                refine_total_ms,
                measurement_timings.len()
            );
            print_refine_timing_summary(&measurement_timings);
        } else {
            println!(
                "Refined {} segments over {} levels in {:.2} ms",
                lsd_segments.len(),
                refined_levels,
                refine_total_ms
            );
        }
    }

    if diagnostics_enabled {
        let stage = SegmentRefineStage {
            pyramid: pyramid_stage.expect("pyramid stage missing"),
            lsd: lsd_stage,
            lsd_segments: Some(lsd_segments.clone()),
            levels: collected_levels,
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

fn load_config_from_args() -> Result<seg_cfg::SegmentRefineDemoConfig, String> {
    let config_path = env::args().nth(1).ok_or_else(usage)?;
    seg_cfg::load_config(Path::new(&config_path))
}

fn ensure_output_dir(config: &seg_cfg::SegmentRefineDemoConfig) -> Result<(), String> {
    fs::create_dir_all(&config.output.dir)
        .map_err(|e| format!("Failed to create {}: {e}", config.output.dir.display()))
}

fn build_pyramid_stage(
    gray: &GrayImageU8,
    config: &seg_cfg::SegmentRefineDemoConfig,
) -> Result<(Pyramid, Option<PyramidStage>, f64), String> {
    let levels = config.pyramid.levels.max(1);
    let pyramid_opts = PyramidOptions::new(levels).with_blur_levels(config.pyramid.blur_levels);
    let ResultWithTime {
        result: pyramid,
        elapsed_ms,
    } = run_with_timer(|| {
        let pyramid = Pyramid::build_u8(gray.as_view(), pyramid_opts);
        Ok(pyramid)
    })?;
    let stage =
        (!config.performance_mode).then(|| PyramidStage::from_pyramid(&pyramid, elapsed_ms));
    Ok((pyramid, stage, elapsed_ms))
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
    config: &seg_cfg::SegmentRefineDemoConfig,
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
    refine_params: &segment::RefineParams,
    collect_levels: bool,
    log_timings: bool,
    parallel: ParallelRefineOptions,
) -> Result<RefineRunResult, String> {
    let mut levels_report: Vec<SegmentRefineLevel> = Vec::new();
    let mut refine_total_ms = 0.0f64;
    let mut current_segments: Vec<Segment> = source_segments.to_vec();
    let full_width = pyramid.levels.first().map(|lvl| lvl.w).unwrap_or(0);
    let mut level_timings = Vec::new();

    for coarse_idx in (1..pyramid.levels.len()).rev() {
        let finer_idx = coarse_idx - 1;
        let refine_start = Instant::now();
        let finer_level = &pyramid.levels[finer_idx];
        let coarse_level = &pyramid.levels[coarse_idx];
        let scale_map = level_scale_map(coarse_level, finer_level);
        let level_params = refine_params.for_level(full_width, finer_level.w);
        let mut accepted = 0usize;
        let mut score_sum = 0.0f32;
        let segments_in = current_segments.len();
        let results = refine_segments_between_levels(
            workspace,
            finer_level,
            finer_idx,
            &scale_map,
            &level_params,
            current_segments,
            parallel,
        );
        let mut refined_segments = Vec::with_capacity(results.len());
        let mut samples = collect_levels.then(|| Vec::with_capacity(results.len()));

        for result in results {
            if result.ok {
                accepted += 1;
                if result.score.is_finite() {
                    score_sum += result.score;
                }
            }
            let updated = result.seg;
            if let Some(samples) = samples.as_mut() {
                let score = result.score.is_finite().then_some(result.score);
                let inliers = (result.inliers > 0).then_some(result.inliers);
                let total = (result.total > 0).then_some(result.total);
                samples.push(SegmentRefineSample {
                    segment: updated.clone(),
                    score,
                    ok: Some(result.ok),
                    inliers,
                    total,
                });
            }
            refined_segments.push(updated);
        }

        let elapsed_ms = refine_start.elapsed().as_secs_f64() * 1000.0;
        if log_timings {
            println!(
                "Refinement for level {} took {:.2} ms",
                finer_idx, elapsed_ms
            );
        }
        refine_total_ms += elapsed_ms;
        level_timings.push(LevelTiming {
            coarse_level: coarse_idx,
            finer_level: finer_idx,
            elapsed_ms,
            segments_in,
        });
        let acceptance_ratio = (segments_in > 0).then_some(accepted as f32 / segments_in as f32);
        let avg_score = (accepted > 0).then_some(score_sum / accepted as f32);

        if let Some(samples) = samples {
            levels_report.push(SegmentRefineLevel {
                coarse_level: coarse_idx,
                finer_level: finer_idx,
                elapsed_ms,
                segments_in,
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

    Ok(RefineRunResult {
        levels_report,
        timings: RunTimings {
            total_ms: refine_total_ms,
            level_timings,
        },
    })
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

fn configure_thread_pool(max_threads: Option<usize>) {
    #[cfg(feature = "parallel")]
    {
        if let Some(limit) = max_threads {
            if limit > 0 {
                if let Err(err) = rayon::ThreadPoolBuilder::new()
                    .num_threads(limit)
                    .build_global()
                {
                    eprintln!("Warning: unable to configure Rayon thread pool: {err}");
                }
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    let _ = max_threads;
}

fn print_refine_timing_summary(runs: &[RunTimings]) {
    if runs.len() <= 1 {
        return;
    }
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct LevelAggregate {
        coarse_level: usize,
        finer_level: usize,
        sum_ms: f64,
        sum_segments: f64,
        count: usize,
    }

    let mut per_level: BTreeMap<usize, LevelAggregate> = BTreeMap::new();
    for run in runs {
        for level in &run.level_timings {
            let entry = per_level
                .entry(level.finer_level)
                .or_insert_with(LevelAggregate::default);
            entry.coarse_level = level.coarse_level;
            entry.finer_level = level.finer_level;
            entry.sum_ms += level.elapsed_ms;
            entry.sum_segments += level.segments_in as f64;
            entry.count += 1;
        }
    }

    println!("  Per-level averages:");
    for (_, agg) in per_level {
        if agg.count == 0 {
            continue;
        }
        let avg_ms = agg.sum_ms / agg.count as f64;
        let avg_segments = (agg.sum_segments / agg.count as f64).round() as usize;
        println!(
            "    L{}→L{} avg {:.2} ms (segments≈{})",
            agg.coarse_level, agg.finer_level, avg_ms, avg_segments
        );
    }
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
