//! Demonstration binary for VP fitting, outlier rejection, and homography refinement.
//!
//! The tool runs the following stages on a grayscale image:
//! 1. Build an image pyramid and extract LSD-like segments on the coarsest level.
//! 2. Fit the coarse vanishing-point hypothesis and classify segments as inliers/outliers.
//! 3. Re-fit the model from inliers and optionally run a single-level homography refinement.
//! 4. Emit JSON containing per-segment labels, fitted models, and performance metrics.
//! 5. Save the coarsest pyramid level as a PNG to ease downstream visualisation.
//!
//! Use `tools/plot_vp_outlier_demo.py` to visualise the JSON output.

use grid_detector::config::segments::LsdConfig as LsdStageConfig;
use grid_detector::config::vp_outlier_demo as cfg;
use grid_detector::detector::outliers::classify_segments_with_details;
use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::lsd_vp::{Engine as LsdVpEngine, FamilyLabel};
use grid_detector::pyramid::{Pyramid, PyramidOptions};
use grid_detector::refine::{RefineLevel, Refiner};
use grid_detector::segments::{
    bundle_segments, lsd_extract_segments_with_options, LsdOptions, Segment,
};
use nalgebra::Matrix3;
use serde::Serialize;
use std::env;
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
    let config = cfg::load_config(Path::new(&config_path))?;

    std::fs::create_dir_all(&config.output.dir)
        .map_err(|e| format!("Failed to create {}: {e}", config.output.dir.display()))?;

    let total_start = Instant::now();

    let load_start = Instant::now();
    let gray = load_grayscale_image(&config.input)?;
    let load_ms = elapsed_ms(load_start);
    let full_width = gray.width();
    let full_height = gray.height();

    let levels = config.pyramid.levels.max(1);
    let mut pyramid_opts = PyramidOptions::new(levels);
    if let Some(blur_levels_cfg) = config.pyramid.blur_levels {
        let blur_levels = blur_levels_cfg.min(levels.saturating_sub(1));
        pyramid_opts = pyramid_opts.with_blur_levels(Some(blur_levels));
    }

    let pyramid_start = Instant::now();
    let pyramid = Pyramid::build_u8(gray.as_view(), pyramid_opts);
    let pyramid_ms = elapsed_ms(pyramid_start);

    let coarse_index = pyramid
        .levels
        .len()
        .checked_sub(1)
        .ok_or_else(|| "Pyramid must contain at least one level".to_string())?;
    let coarse_level = &pyramid.levels[coarse_index];
    save_grayscale_f32(coarse_level, &config.output.coarsest_path())?;
    println!(
        "[pyramid] levels={} | coarse=L{} ({}x{}) saved -> {}",
        pyramid.levels.len(),
        coarse_index,
        coarse_level.w,
        coarse_level.h,
        config.output.coarsest_path().display()
    );

    let scale_x = if coarse_level.w > 0 {
        full_width as f32 / coarse_level.w as f32
    } else {
        1.0
    };
    let scale_y = if coarse_level.h > 0 {
        full_height as f32 / coarse_level.h as f32
    } else {
        1.0
    };
    let scale_matrix = Matrix3::new(scale_x, 0.0, 0.0, 0.0, scale_y, 0.0, 0.0, 0.0, 1.0);
    let inv_scale_matrix = Matrix3::new(
        scale_x.recip(),
        0.0,
        0.0,
        0.0,
        scale_y.recip(),
        0.0,
        0.0,
        0.0,
        1.0,
    );

    let lsd_options = build_lsd_options(&config.lsd);
    let lsd_start = Instant::now();
    let segments = lsd_extract_segments_with_options(
        coarse_level,
        config.lsd.magnitude_threshold,
        config.lsd.angle_tolerance_deg.to_radians(),
        config.lsd.min_length,
        lsd_options,
    );
    let lsd_ms = elapsed_ms(lsd_start);
    println!(
        "[lsd] segments={} | mag_thresh={:.3} angle_tol_deg={:.1} min_len={:.1} | {:.1} ms",
        segments.len(),
        config.lsd.magnitude_threshold,
        config.lsd.angle_tolerance_deg,
        config.lsd.min_length,
        lsd_ms
    );
    if segments.is_empty() {
        return Err(
            "LSD stage returned no segments – adjust magnitude_threshold or angle_tolerance"
                .to_string(),
        );
    }

    let engine_params = config.resolve_engine();
    let engine = LsdVpEngine {
        mag_thresh: engine_params.magnitude_threshold,
        angle_tol_deg: engine_params.angle_tolerance_deg,
        min_len: engine_params.min_length,
    };

    let vp_start = Instant::now();
    let detailed = engine
        .infer_with_segments(coarse_level, segments.clone())
        .ok_or_else(|| {
            format!(
                "VP engine failed to produce a hypothesis (segments={}): tighten thresholds or check the input",
                segments.len()
            )
        })?;
    let vp_fit_ms = elapsed_ms(vp_start);
    let initial_h_coarse = detailed.hypothesis.hmtx0;
    let initial_h_full = scale_matrix * initial_h_coarse;
    let dominant_angles_deg = [
        detailed.dominant_angles_rad[0].to_degrees(),
        detailed.dominant_angles_rad[1].to_degrees(),
    ];
    let family_counts = count_families(&detailed.families);
    println!(
        "[vp] conf={:.3} | θ_u={:.1}° θ_v={:.1}° | fam_u={} fam_v={} | {:.1} ms",
        detailed.hypothesis.confidence,
        dominant_angles_deg[0],
        dominant_angles_deg[1],
        family_counts[0],
        family_counts[1],
        vp_fit_ms
    );

    let lsd_vp_params = config.resolve_lsd_vp_params();
    let filter_params = config.outlier.resolve();

    let classify_start = Instant::now();
    let (decisions, diag_core) = classify_segments_with_details(
        &segments,
        &initial_h_coarse,
        &filter_params,
        &lsd_vp_params,
    );
    let outlier_ms = elapsed_ms(classify_start);

    // Build segment reports and inlier set from core decisions
    let mut inlier_segments: Vec<Segment> = Vec::new();
    let mut reports: Vec<SegmentReport> = Vec::with_capacity(decisions.len());
    let mut rejected_angle = 0usize;
    let mut rejected_residual = 0usize;
    for d in &decisions {
        let seg = &segments[d.index];
        if d.inlier {
            inlier_segments.push(seg.clone());
        }
        if d.rejection == Some("angle") {
            rejected_angle += 1;
        } else if d.rejection == Some("residual") {
            rejected_residual += 1;
        }
        reports.push(SegmentReport {
            p0: seg.p0,
            p1: seg.p1,
            len: seg.len,
            strength: seg.strength,
            family_engine: detailed.families[d.index].map(family_label_to_str),
            family: d.family.map(family_label_to_str),
            angle_diff_deg: d.angle_diff_rad.map(|a| a.to_degrees()),
            residual_px: d.residual_px,
            rejection: d.rejection,
            inlier: d.inlier,
        });
    }

    let total = decisions.len();
    let inliers_cnt = inlier_segments.len();
    let outliers_cnt = total.saturating_sub(inliers_cnt);
    println!(
        "[filter] total={} inliers={} outliers={} | rej_angle={} rej_residual={} | angle_thr_deg={:.1} residual_thr_px={:.2} | {:.1} ms",
        total,
        inliers_cnt,
        outliers_cnt,
        rejected_angle,
        rejected_residual,
        (lsd_vp_params.angle_tol_deg + filter_params.angle_margin_deg),
        filter_params.line_residual_thresh_px,
        outlier_ms
    );

    let inlier_fit_start = Instant::now();
    let inlier_detailed = if inlier_segments.len() >= 6 {
        engine.infer_with_segments(coarse_level, inlier_segments.clone())
    } else {
        None
    };
    let inlier_fit_ms = elapsed_ms(inlier_fit_start);
    if let Some(m) = &inlier_detailed {
        let da = [
            m.dominant_angles_rad[0].to_degrees(),
            m.dominant_angles_rad[1].to_degrees(),
        ];
        let counts = count_families(&m.families);
        println!(
            "[vp-inliers] conf={:.3} | θ_u={:.1}° θ_v={:.1}° | fam_u={} fam_v={} | {:.1} ms",
            m.hypothesis.confidence, da[0], da[1], counts[0], counts[1], inlier_fit_ms
        );
    } else {
        println!(
            "[vp-inliers] skipped (inliers={})",
            inlier_segments.len()
        );
    }

    let base_h_coarse = inlier_detailed
        .as_ref()
        .map(|d| d.hypothesis.hmtx0.clone())
        .unwrap_or_else(|| initial_h_coarse.clone());
    let base_h_full = scale_matrix * base_h_coarse;

    let bundling_params = config.bundling.resolve();
    let refine_params = config.refine.resolve();
    let refiner = Refiner::new(refine_params.clone());

    let (rescaled_inliers, scale_applied) =
        rescale_segments_to_full(&inlier_segments, scale_x, scale_y);

    let bundling_start = Instant::now();
    let bundles = if !rescaled_inliers.is_empty() {
        bundle_segments(
            &rescaled_inliers,
            bundling_params.orientation_tol_deg.to_radians(),
            bundling_params.merge_dist_px,
            bundling_params.min_weight,
        )
    } else {
        Vec::new()
    };
    let bundling_ms = if rescaled_inliers.is_empty() {
        0.0
    } else {
        elapsed_ms(bundling_start)
    };
    if !rescaled_inliers.is_empty() {
        println!(
            "[bundling] bundles={} from={} | ori_tol_deg={:.1} merge_dist_px={:.2} min_weight={:.1} | {:.1} ms",
            bundles.len(),
            rescaled_inliers.len(),
            bundling_params.orientation_tol_deg,
            bundling_params.merge_dist_px,
            bundling_params.min_weight,
            bundling_ms
        );
    } else {
        println!("[bundling] skipped (no inlier segments)");
    }

    let enough_bundles =
        bundles.len() >= refine_params.min_bundles_per_family * 2 && !bundles.is_empty();

    let refine_level = if enough_bundles {
        Some(RefineLevel {
            level_index: 0,
            width: full_width,
            height: full_height,
            segments: rescaled_inliers.len(),
            bundles: bundles.as_slice(),
        })
    } else {
        None
    };

    let (refine_result, refine_ms) = if let Some(level) = refine_level {
        let refine_start = Instant::now();
        let result = refiner.refine(base_h_full.clone(), std::slice::from_ref(&level));
        (result, elapsed_ms(refine_start))
    } else {
        (None, 0.0)
    };
    if let Some(res) = &refine_result {
        println!(
            "[refine] conf={:.3} inlier_ratio={:.3} levels_used={} | {:.1} ms",
            res.confidence, res.inlier_ratio, res.levels_used, refine_ms
        );
    } else {
        println!("[refine] skipped (not enough bundles)");
    }

    let total_ms = elapsed_ms(total_start);

    let performance = PerformanceReport {
        load_ms,
        pyramid_ms,
        lsd_ms,
        vp_fit_ms,
        outlier_ms,
        inlier_fit_ms: if inlier_segments.is_empty() {
            None
        } else {
            Some(inlier_fit_ms)
        },
        bundling_ms: if bundles.is_empty() {
            None
        } else {
            Some(bundling_ms)
        },
        refine_ms: refine_result.as_ref().map(|_| refine_ms),
        total_ms,
    };

    let initial_model = ModelReport {
        confidence: detailed.hypothesis.confidence,
        coarse_h: matrix_to_array(&initial_h_coarse),
        full_h: matrix_to_array(&initial_h_full),
        dominant_angles_deg: Some(dominant_angles_deg),
        family_counts: Some(family_counts),
    };

    let inlier_model = inlier_detailed.as_ref().map(|d| ModelReport {
        confidence: d.hypothesis.confidence,
        coarse_h: matrix_to_array(&d.hypothesis.hmtx0),
        full_h: matrix_to_array(&(scale_matrix * d.hypothesis.hmtx0)),
        dominant_angles_deg: Some([
            d.dominant_angles_rad[0].to_degrees(),
            d.dominant_angles_rad[1].to_degrees(),
        ]),
        family_counts: Some(count_families(&d.families)),
    });

    let refined_model = refine_result.map(|res| RefinedModelReport {
        confidence: res.confidence,
        inlier_ratio: res.inlier_ratio,
        levels_used: res.levels_used,
        coarse_h: matrix_to_array(&(inv_scale_matrix * res.h_refined)),
        full_h: matrix_to_array(&res.h_refined),
        levels: res.level_reports,
    });

    let bundling_report = if bundles.is_empty() {
        None
    } else {
        Some(BundleSummary {
            bundles: bundles.len(),
            orientation_tol_deg: bundling_params.orientation_tol_deg,
            merge_dist_px: bundling_params.merge_dist_px,
            min_weight: bundling_params.min_weight,
            source_segments: rescaled_inliers.len(),
            scale_applied,
        })
    };

    let report = DemoReport {
        source_image: config.input.display().to_string(),
        image_size: [full_width, full_height],
        pyramid_levels: pyramid.levels.len(),
        coarse_level_index: coarse_index,
        coarse_size: [coarse_level.w, coarse_level.h],
        scale_to_full: [scale_x, scale_y],
        lsd: LsdStageReport {
            magnitude_threshold: config.lsd.magnitude_threshold,
            angle_tolerance_deg: config.lsd.angle_tolerance_deg,
            min_length: config.lsd.min_length,
            enforce_polarity: config.lsd.enforce_polarity,
            normal_span_limit_px: if config.lsd.limit_normal_span {
                Some(config.lsd.normal_span_limit_px)
            } else {
                None
            },
        },
        outlier_thresholds: OutlierThresholdReport {
            angle_threshold_deg: lsd_vp_params.angle_tol_deg + filter_params.angle_margin_deg,
            angle_margin_deg: filter_params.angle_margin_deg,
            residual_threshold_px: filter_params.line_residual_thresh_px,
        },
        performance_ms: performance,
        initial_model,
        inlier_model,
        refined_model,
        bundling: bundling_report,
        classification: ClassificationStats {
            total,
            inliers: inliers_cnt,
            outliers: outliers_cnt,
            rejected_angle,
            rejected_residual,
            degenerate_segments: diag_core.skipped_degenerate,
            angle_threshold_deg: lsd_vp_params.angle_tol_deg + filter_params.angle_margin_deg,
            residual_threshold_px: filter_params.line_residual_thresh_px,
        },
        segments: reports,
    };

    write_json_file(&config.output.result_path(), &report)?;

    println!(
        "[io] wrote JSON report -> {} | total {:.1} ms",
        config.output.result_path().display(),
        total_ms
    );

    Ok(())
}

fn usage() -> String {
    format!(
        "Usage: vp_outlier_demo <config.json>\n\
         Example config: {{\"input\": \"image.png\", \"output\": {{\"dir\": \"out\", \"coarsest_image\": \"L_coarse.png\", \"result_json\": \"report.json\"}}}}"
    )
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn build_lsd_options(cfg: &LsdStageConfig) -> LsdOptions {
    let mut opts = LsdOptions::default();
    opts.enforce_polarity = cfg.enforce_polarity;
    opts.normal_span_limit = if cfg.limit_normal_span {
        Some(cfg.normal_span_limit_px)
    } else {
        None
    };
    opts
}

fn count_families(families: &[Option<FamilyLabel>]) -> [usize; 2] {
    let mut counts = [0usize, 0usize];
    for fam in families {
        match fam {
            Some(FamilyLabel::U) => counts[0] += 1,
            Some(FamilyLabel::V) => counts[1] += 1,
            None => {}
        }
    }
    counts
}

fn family_label_to_str(label: FamilyLabel) -> &'static str {
    match label {
        FamilyLabel::U => "u",
        FamilyLabel::V => "v",
    }
}

fn rescale_segments_to_full(segments: &[Segment], sx: f32, sy: f32) -> (Vec<Segment>, [f32; 2]) {
    let mut scaled = Vec::with_capacity(segments.len());
    for seg in segments {
        let mut p0 = [seg.p0[0] * sx, seg.p0[1] * sy];
        let mut p1 = [seg.p1[0] * sx, seg.p1[1] * sy];
        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let mut len = (dx * dx + dy * dy).sqrt().max(1e-6);
        let dir = [dx / len, dy / len];

        if !len.is_finite() {
            len = seg.len;
            p0 = seg.p0;
            p1 = seg.p1;
        }

        let mut a = seg.line[0] / sx;
        let mut b = seg.line[1] / sy;
        let mut c = seg.line[2];
        let norm = (a * a + b * b).sqrt().max(1e-6);
        a /= norm;
        b /= norm;
        c /= norm;

        let strength = seg.avg_mag * len;

        scaled.push(Segment {
            p0,
            p1,
            dir,
            len,
            line: [a, b, c],
            avg_mag: seg.avg_mag,
            strength,
        });
    }
    (scaled, [sx, sy])
}

fn matrix_to_array(m: &Matrix3<f32>) -> [[f32; 3]; 3] {
    [
        [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
        [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
        [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
    ]
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DemoReport {
    source_image: String,
    image_size: [usize; 2],
    pyramid_levels: usize,
    coarse_level_index: usize,
    coarse_size: [usize; 2],
    scale_to_full: [f32; 2],
    lsd: LsdStageReport,
    outlier_thresholds: OutlierThresholdReport,
    performance_ms: PerformanceReport,
    initial_model: ModelReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    inlier_model: Option<ModelReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refined_model: Option<RefinedModelReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bundling: Option<BundleSummary>,
    classification: ClassificationStats,
    segments: Vec<SegmentReport>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LsdStageReport {
    magnitude_threshold: f32,
    angle_tolerance_deg: f32,
    min_length: f32,
    enforce_polarity: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    normal_span_limit_px: Option<f32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OutlierThresholdReport {
    angle_threshold_deg: f32,
    angle_margin_deg: f32,
    residual_threshold_px: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PerformanceReport {
    load_ms: f64,
    pyramid_ms: f64,
    lsd_ms: f64,
    vp_fit_ms: f64,
    outlier_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    inlier_fit_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bundling_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refine_ms: Option<f64>,
    total_ms: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelReport {
    confidence: f32,
    coarse_h: [[f32; 3]; 3],
    full_h: [[f32; 3]; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    dominant_angles_deg: Option<[f32; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    family_counts: Option<[usize; 2]>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RefinedModelReport {
    confidence: f32,
    inlier_ratio: f32,
    levels_used: usize,
    coarse_h: [[f32; 3]; 3],
    full_h: [[f32; 3]; 3],
    levels: Vec<grid_detector::diagnostics::RefinementLevelDiagnostics>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BundleSummary {
    bundles: usize,
    orientation_tol_deg: f32,
    merge_dist_px: f32,
    min_weight: f32,
    source_segments: usize,
    scale_applied: [f32; 2],
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ClassificationStats {
    total: usize,
    inliers: usize,
    outliers: usize,
    rejected_angle: usize,
    rejected_residual: usize,
    degenerate_segments: usize,
    angle_threshold_deg: f32,
    residual_threshold_px: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SegmentReport {
    p0: [f32; 2],
    p1: [f32; 2],
    len: f32,
    strength: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    family_engine: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    family: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    angle_diff_deg: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    residual_px: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rejection: Option<&'static str>,
    inlier: bool,
}

// ClassificationOutcome no longer used; classification is derived directly
