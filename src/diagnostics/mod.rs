//! Unified diagnostics data model exposed by the detector and supporting demos.
//!
//! The module is split into focused submodules to keep individual report
//! structures manageable. `DetectionReport` is the main entry point returned by
//! the detector, bundling both the coarse result (`GridResult`) and a detailed
//! `PipelineTrace` describing every stage the pipeline executed.

pub mod bundles;
pub mod gradient_refine;
// pub mod grid_indexing;
pub mod lsd;
// pub mod outliers;
// pub mod pipeline;
pub mod refine;
pub mod segment_refine;
// pub mod segments;
pub mod timing;

pub use crate::segments::SegmentId;
pub use bundles::{BundleDescriptor, BundlingLevel, BundlingStage};
pub use gradient_refine::GradientRefineStage;
// pub use grid_indexing::{FamilyIndexing, GridIndexingStage, GridLineIndex};
pub use lsd::{FamilyCounts, LsdStage};
// pub use outliers::{OutlierFilterStage, OutlierThresholds};
// pub use pipeline::{DetectionReport, InputDescriptor, PipelineTrace};
pub use refine::{RefinementIteration, RefinementOutcome, RefinementStage};
pub use segment_refine::{SegmentRefineLevel, SegmentRefineSample, SegmentRefineStage};
// pub use segments::{SegmentClass, SegmentSample};
pub use timing::{StageTiming, TimingBreakdown};
