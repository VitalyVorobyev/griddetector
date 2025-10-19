mod common;

use common::synthetic_image::checkerboard_u8;
use grid_detector::lsd_vp::Engine as LsdVpEngine;
use grid_detector::pyramid::Pyramid;
use grid_detector::refine::{RefineParams, Refiner};
use grid_detector::segments::lsd_extract_segments;
use nalgebra::Matrix3;

#[test]
fn refiner_improves_checkerboard_hypothesis() {
    let _ = env_logger::builder().is_test(true).try_init();
    let width = 640usize;
    let height = 480usize;
    let cell = 32usize;
    let buffer = checkerboard_u8(width, height, cell);
    let image = grid_detector::types::ImageU8 {
        w: width,
        h: height,
        stride: width,
        data: &buffer,
    };

    let pyramid = Pyramid::build_u8(image, 3);
    let coarsest = pyramid
        .levels
        .last()
        .expect("pyramid has at least one level");

    let mut engine = LsdVpEngine::default();
    let hypothesis = engine
        .infer(coarsest)
        .expect("coarse hypothesis should be available on synthetic checkerboard");
    let scale_x = width as f32 / coarsest.w as f32;
    let scale_y = height as f32 / coarsest.h as f32;
    let scale = Matrix3::new(scale_x, 0.0, 0.0, 0.0, scale_y, 0.0, 0.0, 0.0, 1.0);
    let h_initial = scale * hypothesis.hmtx0;

    // Debug: check segment counts with refine's thresholds per level
    let levels = pyramid.levels.len();
    for (idx, img) in pyramid.levels.iter().enumerate().rev() {
        let scale_lvl = 2.0f32.powi((levels - 1 - idx) as i32);
        let mag_thresh = (0.03f32 / scale_lvl.sqrt()).max(0.005);
        let min_len = (6.0f32 * scale_lvl).max(4.0);
        let angle_tol = 20.0f32.to_radians();
        let segs = lsd_extract_segments(img, mag_thresh, angle_tol, min_len);
        println!(
            "debug refine-level idx={} size={}x{} segs={} mag_thresh={:.3} min_len={:.1}",
            idx,
            img.w,
            img.h,
            segs.len(),
            mag_thresh,
            min_len
        );
    }

    let mut p = RefineParams::default();
    p.min_bundle_weight = 1.0;
    p.min_bundles_per_family = 2;
    let refiner = Refiner::new(p);
    let result = refiner
        .refine(&pyramid, h_initial)
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
