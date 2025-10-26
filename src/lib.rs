#![doc = include_str!("../README.md")]

pub mod angle;
pub mod config;
pub mod detector;
pub mod diagnostics;
pub mod edges;
pub mod image;
pub mod lsd_vp;
pub mod pyramid;
pub mod refine;
pub mod segments;
pub mod types;

pub use crate::detector::{GridDetector, GridParams};
pub use crate::diagnostics::builders::{
    run_lsd_stage, run_outlier_stage, LsdStageOutput, OutlierStageOutput,
};
pub use crate::diagnostics::{
    BundlingLevel, BundlingStage, DetectionReport, InputDescriptor, LsdStage, OutlierFilterStage,
    PipelineTrace, PoseStage, PyramidLevelReport, PyramidStage, RefinementIteration,
    RefinementOutcome, RefinementStage, SegmentClass, SegmentDescriptor, SegmentId,
    SegmentRefineLevel, SegmentRefineSample, SegmentRefineStage, SegmentSample, StageTiming,
    TimingBreakdown,
};
pub use crate::types::{GridResult, Pose};
