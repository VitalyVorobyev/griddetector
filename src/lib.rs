#![doc = include_str!("../README.md")]

// Public modules (stable-ish surface)
pub mod detector;
pub mod diagnostics;
pub mod homography;
pub mod image;
pub mod types;

// “Expert” modules – still public, but considered unstable internals.
// (You can tighten or feature-gate these later.)
pub mod angle;
pub mod edges;
pub mod lsd_vp;
pub mod pyramid;
pub mod refine;
pub mod segments;

// --- High-level re-exports -------------------------------------------------

// Main entry points: detector + results.
pub use crate::detector::{DetectorWorkspace, GridDetector, GridParams};
pub use crate::types::GridResult;

// High-level diagnostics returned by the detector.
pub use crate::diagnostics::{DetectionReport, PipelineTrace};

// Convenience homography helpers that are generally useful.
pub use crate::homography::{apply_homography_points, rescale_homography_image_space};

// --- Prelude ---------------------------------------------------------------

/// Small prelude for quick experiments.
///
/// ```no_run
/// use grid_detector::prelude::*;
/// use nalgebra::Matrix3;
///
/// # fn main() {
/// let (w, h) = (640usize, 480usize);
/// let gray = vec![0u8; w * h];
/// let img = ImageU8 { w, h, stride: w, data: &gray };
///
/// let mut det = GridDetector::new(GridParams {
///     kmtx: Matrix3::identity(),
///     ..Default::default()
/// });
///
/// let report = det.process(img);
/// println!("found={} latency_ms={:.3}", report.grid.found, report.grid.latency_ms);
/// # }
/// ```
pub mod prelude {
    pub use crate::image::ImageU8;
    pub use crate::{GridDetector, GridParams, GridResult};
}

// --- Stage-level diagnostics API (for tools & advanced users) --------------

pub mod stages {
    // Stage runners / builders.
    pub use crate::diagnostics::builders::{
        run_lsd_stage, run_outlier_stage, LsdStageOutput, OutlierStageOutput,
    };

    // Structured diagnostics types.
    pub use crate::diagnostics::{
        BundlingLevel, BundlingStage, FamilyIndexing, GradientRefineStage, GridIndexingStage,
        GridLineIndex, InputDescriptor, LsdStage, OutlierFilterStage, PyramidLevelReport,
        PyramidStage, RefinementIteration, RefinementOutcome, RefinementStage, SegmentClass,
        SegmentId, SegmentRefineLevel, SegmentRefineSample, SegmentRefineStage, SegmentSample,
        StageTiming, TimingBreakdown,
    };
}
