//! Lightweight instrumentation hooks for the gradient-driven segment refiner.

/// High-level stages reported by the gradient refiner when profiling is enabled.
#[derive(Clone, Copy, Debug)]
pub enum ProfileStage {
    Upsample,
    Roi,
    SupportSampling,
    LineFit,
    Endpoints,
    Total,
}

impl ProfileStage {
    /// Human-readable label for diagnostics.
    pub fn label(self) -> &'static str {
        match self {
            ProfileStage::Upsample => "upsample",
            ProfileStage::Roi => "roi",
            ProfileStage::SupportSampling => "support_sampling",
            ProfileStage::LineFit => "line_fit",
            ProfileStage::Endpoints => "endpoints",
            ProfileStage::Total => "total",
        }
    }
}

/// Sink used by the refiner to emit stage durations.
pub trait RefineProfiler {
    fn record(&mut self, stage: ProfileStage, elapsed_ms: f32);
}

/// Default no-op profiler used by the production pipeline.
#[derive(Default)]
pub struct NoopProfiler;

impl RefineProfiler for NoopProfiler {
    #[inline]
    fn record(&mut self, _stage: ProfileStage, _elapsed_ms: f32) {}
}
