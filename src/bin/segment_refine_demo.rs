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
    StageTiming, TimingBreakdown,
};
use grid_detector::image::io::{
    load_grayscale_image, save_grayscale_f32, write_json_file, GrayImageU8,
};
use grid_detector::image::{ImageF32, ImageView};
use grid_detector::lsd_vp::{analyze_families, FamilyAssignments};
use grid_detector::pyramid::{Pyramid, PyramidOptions};
use grid_detector::refine::segment::{self, PyramidLevel as SegmentGradientLevel, RefineProfiler};
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
    result: R,
    elapsed_ms: f64,
}

fn run() -> Result<(), String> {
    let config = load_config_from_args()?;
    ensure_output_dir(&config)?;
    let total_start = Instant::now();

    let ResultWithTime {
        result: gray,
        elapsed_ms: load_ms,
    } = run_with_timer(|| load_grayscale_image(&config.input))?;

    let (pyramid, pyramid_stage, pyramid_ms) = build_pyramid_stage(&gray, &config)?;
    save_pyramid_images(&pyramid, &config.output.dir)?;

    let mut workspace = DetectorWorkspace::new();
    workspace.reset(pyramid.levels.len());

    let (lsd_segments, lsd_stage) = run_lsd_demo(&pyramid, &config)?;
    let refine_params = config.refine.resolve();
    let (levels_report, refine_total_ms) =
        run_refinement_levels(&pyramid, &mut workspace, &lsd_segments, &refine_params)?;

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let mut timings = TimingBreakdown::with_total(total_ms);
    if load_ms > 0.0 {
        timings.push("load", load_ms);
    }
    if pyramid_ms > 0.0 {
        timings.push("pyramid", pyramid_ms);
    }
    if lsd_stage.elapsed_ms > 0.0 {
        timings.push("lsd", lsd_stage.elapsed_ms);
    }
    if refine_total_ms > 0.0 {
        timings.push("segment_refine", refine_total_ms);
    }

    let stage = SegmentRefineStage {
        pyramid: pyramid_stage,
        lsd: Some(lsd_stage),
        lsd_segments: Some(lsd_segments.clone()),
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
) -> Result<(Pyramid, PyramidStage, f64), String> {
    let levels = config.pyramid.levels.max(1);
    let pyramid_opts = PyramidOptions::new(levels).with_blur_levels(config.pyramid.blur_levels);
    let ResultWithTime {
        result: pyramid,
        elapsed_ms,
    } = run_with_timer(|| {
        let pyramid = Pyramid::build_u8(gray.as_view(), pyramid_opts);
        Ok(pyramid)
    })?;
    let stage = PyramidStage::from_pyramid(&pyramid, elapsed_ms);
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
) -> Result<(Vec<Segment>, LsdStage), String> {
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
        let assignments = analyze_families(&segments, config.lsd.angle_tolerance_deg)
            .map_err(|err| format!("Orientation analysis failed: {err}"))?;
        Ok((segments, assignments))
    })?;
    let lsd_stage = build_lsd_stage(&assignments, elapsed_ms);
    Ok((segments, lsd_stage))
}

fn run_refinement_levels(
    pyramid: &Pyramid,
    workspace: &mut DetectorWorkspace,
    source_segments: &[Segment],
    refine_params: &segment::RefineParams,
) -> Result<(Vec<SegmentRefineLevel>, f64), String> {
    let mut levels_report: Vec<SegmentRefineLevel> = Vec::new();
    let mut refine_total_ms = 0.0f64;
    let mut current_segments: Vec<Segment> = source_segments.to_vec();

    for coarse_idx in (1..pyramid.levels.len()).rev() {
        let finer_idx = coarse_idx - 1;
        let refine_start = Instant::now();
        let finer_level = &pyramid.levels[finer_idx];
        let grad = workspace.scharr_gradients(finer_idx, finer_level);
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
            let mut profiler = SampleProfiler::default();
            let result =
                segment::refine_segment(&grad_level, seg, &scale_map, refine_params, &mut profiler);
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
            samples.push(SegmentRefineSample {
                segment: updated.clone(),
                score,
                ok: Some(ok),
                inliers,
                total,
                timings: profiler.into_timings(),
            });
            refined_segments.push(updated);
        }

        let elapsed_ms = refine_start.elapsed().as_secs_f64() * 1000.0;
        refine_total_ms += elapsed_ms;
        let acceptance_ratio = (!current_segments.is_empty())
            .then_some(accepted as f32 / current_segments.len() as f32);
        let avg_score = (accepted > 0).then_some(score_sum / accepted as f32);

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
    }
}

#[derive(Default)]
struct SampleProfiler {
    timings: Vec<StageTiming>,
}

impl SampleProfiler {
    fn into_timings(self) -> Option<Vec<StageTiming>> {
        if self.timings.is_empty() {
            None
        } else {
            Some(self.timings)
        }
    }
}

impl RefineProfiler for SampleProfiler {
    fn record(&mut self, stage: segment::ProfileStage, elapsed_ms: f32) {
        self.timings
            .push(StageTiming::new(stage.label(), elapsed_ms as f64));
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
