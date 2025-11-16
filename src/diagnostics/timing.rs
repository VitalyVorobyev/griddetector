use std::time::Instant;
use serde::{Deserialize, Serialize};

/// Timing entry describing a single stage of the pipeline or one of the helper
/// routines executed by the demos.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StageTiming {
    pub label: String,
    pub elapsed_ms: f64,
}

impl StageTiming {
    pub fn new(label: impl Into<String>, elapsed_ms: f64) -> Self {
        Self {
            label: label.into(),
            elapsed_ms,
        }
    }
}

/// Aggregated timing trace for the detector run.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimingBreakdown {
    pub total_ms: f64,
    pub stages: Vec<StageTiming>,
}

impl TimingBreakdown {
    pub fn with_total(total_ms: f64) -> Self {
        Self {
            total_ms,
            stages: Vec::new(),
        }
    }

    pub fn push(&mut self, label: impl Into<String>, elapsed_ms: f64) {
        self.stages.push(StageTiming::new(label, elapsed_ms));
    }
}

pub struct ResultWithTime<R> {
    result: R,
    elapsed_ms: f64,
}

/// Run a closure while timing its execution and reporting the elapsed time. Should return the same
/// result as the closure.
pub fn run_with_timer<R, F: FnOnce() -> Result<R, String>>(f: F) -> Result<ResultWithTime<R>, String> {
    let start = Instant::now();
    let result = f()?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(ResultWithTime { result, elapsed_ms })
}
