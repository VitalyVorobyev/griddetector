mod common;

use common::synthetic_image::checkerboard_u8;
use grid_detector::lsd_vp::Engine as LsdVpEngine;
use grid_detector::pyramid::Pyramid;
use grid_detector::refine::{RefineParams, Refiner};
use nalgebra::Matrix3;

#[test]
fn refiner_improves_checkerboard_hypothesis() {
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
    let coarsest = pyramid.levels.last().expect("pyramid has at least one level");

    let mut engine = LsdVpEngine::default();
    let hypothesis = engine
        .infer(coarsest)
        .expect("coarse hypothesis should be available on synthetic checkerboard");
    let scale_x = width as f32 / coarsest.w as f32;
    let scale_y = height as f32 / coarsest.h as f32;
    let scale = Matrix3::new(scale_x, 0.0, 0.0, 0.0, scale_y, 0.0, 0.0, 0.0, 1.0);
    let h_initial = scale * hypothesis.hmtx0;

    let refiner = Refiner::new(RefineParams::default());
    let result = refiner
        .refine(&pyramid, h_initial)
        .expect("refinement should succeed");

    assert!(result.confidence > 0.0, "expected positive refinement confidence");
    let delta = (result.h_refined - h_initial).norm();
    assert!(
        delta < 1e-2 || result.inlier_ratio > 0.5,
        "refinement should adjust homography or confirm strong inliers"
    );
}
