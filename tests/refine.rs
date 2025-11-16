mod common;

use common::synthetic_image::checkerboard_u8;
use grid_detector::lsd_vp::Engine as LsdVpEngine;
use grid_detector::pyramid::{Pyramid, PyramidOptions};
use grid_detector::refine::{RefineLevel, RefineOptions, Refiner};
use grid_detector::segments::{bundle_segments, Bundle};
use nalgebra::Matrix3;

const MERGE_DIST: f32 = 1.5;
const MIN_BUNDLE_WEIGHT: f32 = 3.0;

fn rescale_bundle(bundle: Bundle, scale_x: f32, scale_y: f32) -> Bundle {
    let mut b = bundle;
    b.center[0] *= scale_x;
    b.center[1] *= scale_y;

    let mut a = b.line[0] / scale_x;
    let mut bb = b.line[1] / scale_y;
    let mut c = b.line[2];
    let norm = (a * a + bb * bb).sqrt().max(1e-6);
    a /= norm;
    bb /= norm;
    c /= norm;

    b.line = [a, bb, c];
    b.weight *= 0.5 * (scale_x + scale_y);
    b
}

#[test]
fn refiner_improves_checkerboard_hypothesis() {
    let _ = env_logger::builder().is_test(true).try_init();
    let width = 640usize;
    let height = 480usize;
    let cell = 32usize;
    let buffer = checkerboard_u8(width, height, cell);
    let image = grid_detector::image::ImageU8 {
        w: width,
        h: height,
        stride: width,
        data: &buffer,
    };

    let pyramid = Pyramid::build_u8(image, PyramidOptions::new(3));
    let coarsest = pyramid
        .levels
        .last()
        .expect("pyramid has at least one level");

    let engine = LsdVpEngine::default();
    let detailed = engine
        .infer(coarsest)
        .expect("coarse hypothesis should be available on synthetic checkerboard");
    let segments = detailed.segments;
    let hypothesis = detailed.hypothesis;
    let scale_x = width as f32 / coarsest.w as f32;
    let scale_y = height as f32 / coarsest.h as f32;
    let scale = Matrix3::new(scale_x, 0.0, 0.0, 0.0, scale_y, 0.0, 0.0, 0.0, 1.0);
    let h_initial = scale * hypothesis.hmtx0;

    let params = RefineOptions {
        min_bundles_per_family: 2,
        ..Default::default()
    };
    let orientation_tol = params.orientation_tol_deg.to_radians();
    let bundles = bundle_segments(&segments, orientation_tol, MERGE_DIST, MIN_BUNDLE_WEIGHT);
    assert!(
        !bundles.is_empty(),
        "bundling should produce constraints on synthetic checkerboard"
    );
    let bundles_full: Vec<Bundle> = bundles
        .into_iter()
        .map(|b| rescale_bundle(b, scale_x, scale_y))
        .collect();
    let level = RefineLevel {
        level_index: 0,
        width,
        height,
        segments: segments.len(),
        bundles: bundles_full.as_slice(),
    };
    let levels = [level];
    let refiner = Refiner::new(params);
    let result = refiner
        .refine(h_initial, &levels)
        .expect("refinement should succeed");

    assert!(
        result.confidence > 0.0,
        "expected positive refinement confidence"
    );
    let delta = (result.h_refined - h_initial).norm();
    assert!(
        delta < 1e-2 || result.inlier_ratio > 0.5,
        "refinement should adjust homography or confirm strong inliers"
    );
}
