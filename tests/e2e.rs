mod common;

use common::synthetic_image::checkerboard_u8;
use grid_detector::image::ImageU8;
use grid_detector::{GridDetector, GridParams};
use nalgebra::Matrix3;

#[test]
fn checkerboard_image_triggers_detection() {
    let width = 640usize;
    let height = 480usize;
    let cell = 32usize;
    let buffer = checkerboard_u8(width, height, cell);

    let image = ImageU8 {
        w: width,
        h: height,
        stride: width,
        data: &buffer,
    };

    let params = GridParams {
        pyramid_levels: 3,
        confidence_thresh: 0.2,
        kmtx: Matrix3::identity(),
        ..Default::default()
    };
    let mut detector = GridDetector::new(params);
    let result = detector.process(image);

    assert!(
        result.grid.found,
        "expected detector to find the grid, confidence={:.3}",
        result.grid.confidence
    );
    assert!(
        result.grid.confidence >= 0.2,
        "confidence below configured threshold: {:.3}",
        result.grid.confidence
    );
    assert_eq!(result.grid.pose.is_some(), result.grid.found);
}
