//! Unified diagnostics data model exposed by the detector and supporting demos.
//!
//! The module is split into focused submodules to keep individual report
//! structures manageable. `DetectionReport` is the main entry point returned by
//! the detector, bundling both the coarse result (`GridResult`) and a detailed
//! `PipelineTrace` describing every stage the pipeline executed.

pub mod builders;
pub mod bundles;
pub mod lsd;
pub mod outliers;
pub mod pipeline;
pub mod pyramid;
pub mod refine;
pub mod segment_refine;
pub mod segments;
pub mod timing;

pub use bundles::{BundleDescriptor, BundlingLevel, BundlingStage};
pub use lsd::{FamilyCounts, LsdStage};
pub use outliers::{OutlierFilterStage, OutlierThresholds};
pub use pipeline::{DetectionReport, InputDescriptor, PipelineTrace, PoseStage};
pub use pyramid::{PyramidLevelReport, PyramidStage};
pub use refine::{RefinementIteration, RefinementOutcome, RefinementStage};
pub use segment_refine::{SegmentRefineLevel, SegmentRefineSample, SegmentRefineStage};
pub use segments::{SegmentClass, SegmentDescriptor, SegmentId, SegmentSample};
pub use timing::{StageTiming, TimingBreakdown};
