use crate::types::GridResult;
use nalgebra::Matrix3;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct BundleEntryDiagnostics {
    pub center: [f32; 2],
    pub line: [f32; 3],
    pub weight: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct BundleDiagnostics {
    pub level_index: usize,
    pub bundles: Vec<BundleEntryDiagnostics>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SegmentFilterDiagnostics {
    pub total: usize,
    pub kept: usize,
    pub rejected: usize,
    pub kept_u: usize,
    pub kept_v: usize,
    pub skipped_degenerate: usize,
    pub angle_threshold_deg: f32,
    pub residual_threshold_px: f32,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct PyramidLevelDiagnostics {
    pub level: usize,
    pub width: usize,
    pub height: usize,
    pub mean_intensity: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct LsdSegmentDiagnostics {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub len: f32,
    pub strength: f32,
    pub family: Option<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LsdDiagnostics {
    pub segments_total: usize,
    pub dominant_angles_deg: [f32; 2],
    pub family_u_count: usize,
    pub family_v_count: usize,
    pub confidence: f32,
    pub elapsed_ms: f64,
    pub segments_sample: Vec<LsdSegmentDiagnostics>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RefinementLevelDiagnostics {
    pub level_index: usize,
    pub width: usize,
    pub height: usize,
    pub segments: usize,
    pub bundles: usize,
    pub family_u_count: usize,
    pub family_v_count: usize,
    pub improvement: Option<f32>,
    pub confidence: Option<f32>,
    pub inlier_ratio: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RefinementDiagnostics {
    pub levels_used: usize,
    pub aggregated_confidence: f32,
    pub final_inlier_ratio: f32,
    pub levels: Vec<RefinementLevelDiagnostics>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProcessingDiagnostics {
    pub input_width: usize,
    pub input_height: usize,
    pub pyramid_levels: Vec<PyramidLevelDiagnostics>,
    pub pyramid_build_ms: f64,
    pub lsd_ms: f64,
    pub segment_filter: Option<SegmentFilterDiagnostics>,
    pub outlier_filter_ms: f64,
    pub bundling_ms: f64,
    pub segment_refine_ms: f64,
    pub refine_ms: f64,
    pub refinement_passes: usize,
    pub lsd: Option<LsdDiagnostics>,
    pub refinement: Option<RefinementDiagnostics>,
    pub bundling: Option<Vec<BundleDiagnostics>>,
    pub homography: Matrix3<f32>,
    pub total_latency_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct DetailedResult {
    pub result: GridResult,
    pub diagnostics: ProcessingDiagnostics,
}

fn format_opt(val: Option<f32>) -> String {
    val.map(|v| format!("{:.3}", v))
        .unwrap_or_else(|| "-".to_string())
}

impl DetailedResult {
    pub fn print_text_summary(&self) {
        let res = &self.result;
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

        let diag = &self.diagnostics;
        println!("\nPyramid (built in {:.3} ms)", diag.pyramid_build_ms);
        for lvl in &diag.pyramid_levels {
            println!(
                "  L{}: {}x{} mean={:.4}",
                lvl.level, lvl.width, lvl.height, lvl.mean_intensity
            );
        }

        println!(
        "\nTimings (ms): pyramid={:.3} lsd={:.3} filter={:.3} bundling={:.3} seg_refine={:.3} refine={:.3} total={:.3}",
        diag.pyramid_build_ms,
        diag.lsd_ms,
        diag.outlier_filter_ms,
        diag.bundling_ms,
        diag.segment_refine_ms,
        diag.refine_ms,
        diag.total_latency_ms
    );

        if let Some(filter) = &diag.segment_filter {
            println!(
            "Segment filter: kept={}/{} (fam_u={} fam_v={} skipped_degenerate={}) angle_thresh={:.1} residual_thresh={:.2}px elapsed_ms={:.3}",
            filter.kept,
            filter.total,
            filter.kept_u,
            filter.kept_v,
            filter.skipped_degenerate,
            filter.angle_threshold_deg,
            filter.residual_threshold_px,
            filter.elapsed_ms,
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
                "\nRefinement: passes={} levels={} aggregated_confidence={:.3} inlier_ratio={:.3}",
                diag.refinement_passes,
                refine.levels_used,
                refine.aggregated_confidence,
                refine.final_inlier_ratio
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
}
