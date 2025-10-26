use crate::diagnostics::{
    BundlingStage, LsdStage, OutlierFilterStage, PyramidStage, RefinementStage, SegmentDescriptor,
    TimingBreakdown,
};
use crate::types::{GridResult, Pose};
use serde::Serialize;

/// Result produced by [`GridDetector::process_with_diagnostics`](crate::GridDetector).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectionReport {
    pub grid: GridResult,
    pub trace: PipelineTrace,
}

impl DetectionReport {
    pub fn print_text_summary(&self) {
        let res = &self.grid;
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

        if let Some(coarse) = &self.trace.coarse_homography {
            println!(
                "\nCoarse homography (pre-refine):\n    [{:.4} {:.4} {:.4}]\n    [{:.4} {:.4} {:.4}]\n    [{:.4} {:.4} {:.4}]",
                coarse[0][0],
                coarse[0][1],
                coarse[0][2],
                coarse[1][0],
                coarse[1][1],
                coarse[1][2],
                coarse[2][0],
                coarse[2][1],
                coarse[2][2]
            );
        }

        if let Some(pyramid) = &self.trace.pyramid {
            println!("\nPyramid (built in {:.3} ms)", pyramid.elapsed_ms);
            for lvl in &pyramid.levels {
                println!(
                    "  L{}: {:>4}x{:>4} mean={:.4}",
                    lvl.level_index, lvl.width, lvl.height, lvl.mean_intensity
                );
            }
        }

        if !self.trace.timings.stages.is_empty() {
            print!("\nTimings (ms): total={:.3}", self.trace.timings.total_ms);
            for stage in &self.trace.timings.stages {
                print!("\n{:>19}={:8.3}", stage.label, stage.elapsed_ms);
            }
            println!();
        }

        if let Some(lsd) = &self.trace.lsd {
            println!(
                "\nLSD stage: segments={} fam_u={} fam_v={} unassigned={} conf={:.3} elapsed_ms={:.3}",
                self.trace.segments.len(),
                lsd.family_counts.family_u,
                lsd.family_counts.family_v,
                lsd.family_counts.unassigned,
                lsd.confidence,
                lsd.elapsed_ms
            );
            println!(
                "  dominant_angles_deg=[{:.1}, {:.1}]",
                lsd.dominant_angles_deg[0], lsd.dominant_angles_deg[1]
            );
        } else {
            println!("\nLSD stage: no viable hypothesis");
        }

        if let Some(outlier) = &self.trace.outlier_filter {
            println!(
                "\nSegment filter: kept={}/{} (fam_u={} fam_v={} degenerate={}) angle_thresh={:.1} margin={:.1} residual_thresh={:.2}px elapsed_ms={:.3}",
                outlier.kept,
                outlier.total,
                outlier.kept_u,
                outlier.kept_v,
                outlier.degenerate_segments,
                outlier.thresholds.angle_threshold_deg,
                outlier.thresholds.angle_margin_deg,
                outlier.thresholds.residual_threshold_px,
                outlier.elapsed_ms
            );
        } else {
            println!("\nSegment filter: skipped");
        }

        if let Some(refine) = &self.trace.refinement {
            if let Some(outcome) = &refine.outcome {
                println!(
                    "\nRefinement: passes={} levels_used={} conf={:.3} inlier_ratio={:.3} elapsed_ms={:.3}",
                    refine.passes,
                    outcome.levels_used,
                    outcome.confidence,
                    outcome.inlier_ratio,
                    refine.elapsed_ms
                );
                for lvl in &outcome.iterations {
                    println!(
                        "  L{}: {}x{} segs={} bundles={} fam_u={} fam_v={} improvement={} conf={} inliers={}",
                        lvl.level_index,
                        lvl.width,
                        lvl.height,
                        lvl.segments,
                        lvl.bundles,
                        lvl.family_u_count,
                        lvl.family_v_count,
                        format_optional(lvl.improvement),
                        format_optional(lvl.confidence),
                        format_optional(lvl.inlier_ratio)
                    );
                }
            } else {
                println!("\nRefinement: attempted but no valid update");
            }
        } else {
            println!("\nRefinement: disabled or no refinement result");
        }
    }
}

fn format_optional(val: Option<f32>) -> String {
    val.map(|v| format!("{:.3}", v))
        .unwrap_or_else(|| "-".to_string())
}

/// End-to-end trace describing the internal execution of the detector.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PipelineTrace {
    pub input: InputDescriptor,
    pub timings: TimingBreakdown,
    pub segments: Vec<SegmentDescriptor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pyramid: Option<PyramidStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lsd: Option<LsdStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outlier_filter: Option<OutlierFilterStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bundling: Option<BundlingStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refinement: Option<RefinementStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coarse_homography: Option<[[f32; 3]; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pose: Option<PoseStage>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InputDescriptor {
    pub width: usize,
    pub height: usize,
    pub pyramid_levels: usize,
}

/// Camera pose recovered from the refined homography, if available.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PoseStage {
    pub rotation: [[f32; 3]; 3],
    pub translation: [f32; 3],
}

impl PoseStage {
    pub fn from_pose(pose: &Pose) -> Self {
        Self {
            rotation: [
                [pose.r[(0, 0)], pose.r[(0, 1)], pose.r[(0, 2)]],
                [pose.r[(1, 0)], pose.r[(1, 1)], pose.r[(1, 2)]],
                [pose.r[(2, 0)], pose.r[(2, 1)], pose.r[(2, 2)]],
            ],
            translation: [pose.t[0], pose.t[1], pose.t[2]],
        }
    }
}
