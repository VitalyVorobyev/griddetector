use crate::detector::params::BundlingParams;
use crate::detector::scaling::{rescale_bundle_to_full_res, LevelScaling};
use crate::segments::{self, Bundle, Segment};
use nalgebra::Matrix3;
use std::time::Instant;

const EPS: f32 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrategyFrame {
    ImageSpace,
    Rectified,
}

#[derive(Clone, Copy, Debug)]
pub struct BundleThresholds {
    pub orientation_tol_rad: Option<f32>,
    pub merge_distance: f32,
    pub min_weight: f32,
}

#[derive(Debug)]
pub struct BundleOutcome {
    pub bundles: Vec<Bundle>,
    pub elapsed_ms: f64,
    pub frame: StrategyFrame,
    pub applied_scale: Option<[f32; 2]>,
}

pub trait BundleStrategy {
    fn frame(&self) -> StrategyFrame;
    fn thresholds(&self, scaling: &LevelScaling) -> BundleThresholds;
    fn bundle_native(
        &self,
        segments: &[Segment],
        thresholds: &BundleThresholds,
        coarse_h: Option<&Matrix3<f32>>,
    ) -> Vec<Bundle>;

    fn bundle(
        &self,
        segments: &[Segment],
        scaling: &LevelScaling,
        coarse_h: Option<&Matrix3<f32>>,
    ) -> BundleOutcome {
        let thresholds = self.thresholds(scaling);
        let start = Instant::now();
        let bundles = self.bundle_native(segments, &thresholds, coarse_h);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let bundles = bundles
            .into_iter()
            .map(|b| {
                rescale_bundle_to_full_res(b, scaling.scale_x_to_full, scaling.scale_y_to_full)
            })
            .collect();
        BundleOutcome {
            bundles,
            elapsed_ms,
            frame: self.frame(),
            applied_scale: Some([scaling.scale_x_to_full, scaling.scale_y_to_full]),
        }
    }
}

pub struct RectifiedBundleStrategy<'a> {
    params: &'a BundlingParams,
}

impl<'a> RectifiedBundleStrategy<'a> {
    pub fn new(params: &'a BundlingParams) -> Self {
        Self { params }
    }
}

impl<'a> BundleStrategy for RectifiedBundleStrategy<'a> {
    fn frame(&self) -> StrategyFrame {
        StrategyFrame::Rectified
    }

    fn thresholds(&self, scaling: &LevelScaling) -> BundleThresholds {
        let (dist, weight) = adapt_thresholds(self.params, scaling);
        BundleThresholds {
            orientation_tol_rad: None,
            merge_distance: dist,
            min_weight: weight,
        }
    }

    fn bundle_native(
        &self,
        segments: &[Segment],
        thresholds: &BundleThresholds,
        coarse_h: Option<&Matrix3<f32>>,
    ) -> Vec<Bundle> {
        let Some(h) = coarse_h else {
            return Vec::new();
        };
        crate::lsd_vp::bundling::bundle_rectified(
            segments,
            h,
            thresholds.merge_distance,
            thresholds.min_weight,
        )
    }
}

pub struct ImageSpaceBundleStrategy<'a> {
    params: &'a BundlingParams,
}

impl<'a> ImageSpaceBundleStrategy<'a> {
    pub fn new(params: &'a BundlingParams) -> Self {
        Self { params }
    }
}

impl<'a> BundleStrategy for ImageSpaceBundleStrategy<'a> {
    fn frame(&self) -> StrategyFrame {
        StrategyFrame::ImageSpace
    }

    fn thresholds(&self, scaling: &LevelScaling) -> BundleThresholds {
        let (dist, weight) = adapt_thresholds(self.params, scaling);
        BundleThresholds {
            orientation_tol_rad: Some(self.params.orientation_tol_deg.to_radians()),
            merge_distance: dist,
            min_weight: weight,
        }
    }

    fn bundle_native(
        &self,
        segments: &[Segment],
        thresholds: &BundleThresholds,
        _coarse_h: Option<&Matrix3<f32>>,
    ) -> Vec<Bundle> {
        let orient = thresholds.orientation_tol_rad.unwrap_or(0.0);
        segments::bundle_segments(
            segments,
            orient,
            thresholds.merge_distance,
            thresholds.min_weight,
        )
    }
}

pub struct BundleStack<'a> {
    params: &'a BundlingParams,
    rectified: RectifiedBundleStrategy<'a>,
    image_space: ImageSpaceBundleStrategy<'a>,
}

impl<'a> BundleStack<'a> {
    pub fn new(params: &'a BundlingParams) -> Self {
        Self {
            params,
            rectified: RectifiedBundleStrategy::new(params),
            image_space: ImageSpaceBundleStrategy::new(params),
        }
    }

    pub fn params(&self) -> &'a BundlingParams {
        self.params
    }

    pub fn bundle_level(
        &self,
        segments: &[Segment],
        scaling: &LevelScaling,
        coarse_h: Option<&Matrix3<f32>>,
    ) -> BundleOutcome {
        if coarse_h.is_some() {
            let outcome = self.rectified.bundle(segments, scaling, coarse_h);
            if !outcome.bundles.is_empty() || segments.is_empty() {
                return outcome;
            }
        }
        self.image_space.bundle(segments, scaling, coarse_h)
    }
}

pub fn adapt_thresholds(params: &BundlingParams, scaling: &LevelScaling) -> (f32, f32) {
    let dist = params.merge_dist_px * scaling.mean_scale_from_full;
    let weight = params.min_weight * scaling.mean_scale_from_full;
    (dist.max(EPS), weight.max(EPS))
}

/// Bundle coarse segments on the coarsest pyramid level and rescale to full resolution.
///
/// This is a convenience wrapper used by the coarse-only pipeline path.
/// Returns both the diagnostic stage and the produced bundles.
pub fn bundle_coarsest(
    bundler: &BundleStack<'_>,
    pyramid: &crate::pyramid::Pyramid,
    coarse_h: Option<&Matrix3<f32>>,
    segments: &[Segment],
    full_width: usize,
    full_height: usize,
) -> Option<(crate::diagnostics::BundlingStage, Vec<Bundle>)> {
    if segments.is_empty() {
        return None;
    }
    let h = coarse_h?;
    let level_index = pyramid.levels.len().checked_sub(1)?;
    let level = &pyramid.levels[level_index];
    let scaling = LevelScaling::from_dimensions(level.w, level.h, full_width, full_height);

    let outcome = bundler.bundle_level(segments, &scaling, Some(h));
    let elapsed_ms = outcome.elapsed_ms;
    let bundles_full = outcome.bundles;
    let params = bundler.params();

    let level_descriptor = crate::diagnostics::BundlingLevel {
        level_index,
        width: level.w,
        height: level.h,
        bundles: bundles_full
            .iter()
            .map(|b| crate::diagnostics::BundleDescriptor {
                center: b.center,
                line: b.line,
                weight: b.weight,
            })
            .collect(),
    };

    let stage = crate::diagnostics::BundlingStage {
        elapsed_ms,
        segment_refine_ms: 0.0,
        orientation_tol_deg: params.orientation_tol_deg,
        merge_distance_px: params.merge_dist_px,
        min_weight: params.min_weight,
        source_segments: segments.len(),
        scale_applied: outcome.applied_scale,
        levels: vec![level_descriptor],
    };

    Some((stage, bundles_full))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_scale() {
        let params = BundlingParams {
            orientation_tol_deg: 20.0,
            merge_dist_px: 2.0,
            min_weight: 4.0,
        };
        let scaling = LevelScaling::from_dimensions(100, 100, 400, 400);
        let (dist, weight) = adapt_thresholds(&params, &scaling);
        assert!((dist - 0.5).abs() < 1e-6);
        assert!((weight - 1.0).abs() < 1e-6);
    }
}
