//! Demonstration binary for the gradient-based segment refiner.
//!
//! The demo mirrors the early detector pipeline stages:
//! 1. Build an image pyramid.
//! 2. Run LSD on the coarsest level.
//! 3. Build an orientation histogram and assign segment families.
//! 4. Refine the segments down the pyramid using only local gradients.
//! 5. Emit diagnostics identical to the pipeline (pyramid, LSD stage, refinement levels).

use grid_detector::config::segment_refine_demo as seg_cfg;
use grid_detector::detector::DetectorWorkspace;
use grid_detector::diagnostics::builders::{compute_family_counts, run_segment_refine_levels};
use grid_detector::diagnostics::{LsdStage, PyramidStage, SegmentRefineStage, TimingBreakdown};
use grid_detector::image::io::{
    load_grayscale_image, save_grayscale_f32, write_json_file, GrayImageU8,
};
use grid_detector::lsd_vp::{analyze_families, FamilyAssignments};
use grid_detector::pyramid::{Pyramid, PyramidOptions};
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

fn run() -> Result<(), String> {
    let config = load_config_from_args()?;
    ensure_output_dir(&config)?;

    let ResultWithTime {
        result: gray,
        elapsed_ms: load_ms,
    } = run_with_timer(|| load_grayscale_image(&config.input))?;

    let total_start = Instant::now();

    let (pyramid, pyramid_stage, pyramid_ms) = build_pyramid_stage(&gray, &config)?;
    save_pyramid_images(&pyramid, &config.output.dir)?;

    let mut workspace = DetectorWorkspace::new();
    workspace.reset(pyramid.levels.len());

    let (lsd_segments, lsd_stage) = run_lsd_demo(&pyramid, &config)?;
    let refine_params = config.refine.resolve();
    let refine_run =
        run_segment_refine_levels(&pyramid, &mut workspace, &lsd_segments, &refine_params);
    let refine_total_ms = refine_run.elapsed_ms;
    let levels_report = refine_run.levels;

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let mut timings = TimingBreakdown::with_total(total_ms);
    if load_ms > 0.0 {
        timings.push("load", load_ms);
    }
    if pyramid_ms > 0.0 {
        timings.push("pyramid", pyramid_ms);
        println!("Pyramid built in {:.2} ms", pyramid_ms);
    }
    if lsd_stage.elapsed_ms > 0.0 {
        timings.push("lsd", lsd_stage.elapsed_ms);
        println!(
            "LSD detected {} segments in {:.2} ms",
            lsd_segments.len(),
            lsd_stage.elapsed_ms
        );
    }
    if refine_total_ms > 0.0 {
        timings.push("segment_refine", refine_total_ms);
        println!(
            "Refined {} segments over {} levels in {:.2} ms",
            lsd_segments.len(),
            levels_report.len(),
            refine_total_ms
        );
    }

    #[cfg(feature = "profile_refine")]
    dump_refine_profile(&workspace);

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
