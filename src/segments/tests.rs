use super::extractor::{LsdExtractor, LsdResult};
use super::*;
use crate::image::ImageF32;

fn step_image(width: usize, height: usize, split_x: usize) -> ImageF32 {
    let mut img = ImageF32::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let v = if x < split_x { 0.0 } else { 1.0 };
            img.set(x, y, v);
        }
    }
    img
}

#[test]
fn lsd_extractor_finds_vertical_segment() {
    let img = step_image(32, 32, 16);
    let LsdResult { segments: segs, .. } =
        LsdExtractor::from_image(&img, LsdOptions::default()).extract_full_scan();
    assert!(
        !segs.is_empty(),
        "expected at least one segment on a vertical edge"
    );
    let longest = segs
        .iter()
        .max_by(|a, b| a.length_sq().partial_cmp(&b.length_sq()).unwrap())
        .unwrap();
    assert!(
        longest.direction()[1].abs() > longest.direction()[0].abs(),
        "expected vertical-oriented segment, got dir={:?}",
        longest.direction()
    );
    assert!(
        longest.length() >= 8.0,
        "expected a reasonably long segment, got len={}",
        longest.length()
    );
}

#[test]
fn lsd_extractor_rejects_flat_image() {
    let img = ImageF32::new(16, 16);
    let LsdResult { segments: segs, .. } =
        LsdExtractor::from_image(&img, LsdOptions::default()).extract_full_scan();
    assert!(
        segs.is_empty(),
        "no segments should be detected in a flat image, got {:?}",
        segs
    );
}
