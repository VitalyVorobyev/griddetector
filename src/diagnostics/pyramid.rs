use crate::image::traits::ImageView;
use crate::pyramid::Pyramid;
use serde::{Deserialize, Serialize};

/// Statistics for a single level of the image pyramid.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PyramidLevelReport {
    pub level_index: usize,
    pub width: usize,
    pub height: usize,
    pub mean_intensity: f32,
}

/// Pyramid construction details captured by the detector or demo utilities.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PyramidStage {
    pub elapsed_ms: f64,
    pub levels: Vec<PyramidLevelReport>,
}

impl PyramidStage {
    pub fn from_pyramid(pyramid: &Pyramid, elapsed_ms: f64) -> Self {
        let levels = pyramid
            .levels
            .iter()
            .enumerate()
            .map(|(level, lvl)| {
                let sum: f32 = if let Some(slice) = lvl.as_slice() {
                    slice.iter().copied().sum()
                } else {
                    lvl.rows().map(|r| r.iter().copied().sum::<f32>()).sum()
                };
                let denom = (lvl.w * lvl.h).max(1) as f32;
                PyramidLevelReport {
                    level_index: level,
                    width: lvl.w,
                    height: lvl.h,
                    mean_intensity: sum / denom,
                }
            })
            .collect();
        Self { elapsed_ms, levels }
    }
}
