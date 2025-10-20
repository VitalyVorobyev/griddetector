use super::extractor::LsdExtractor;
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
    let segs = LsdExtractor::new(
        &img,
        0.1,
        std::f32::consts::FRAC_PI_6,
        4.0,
        None,
        LsdOptions::default(),
    )
    .extract();
    assert!(
        !segs.is_empty(),
        "expected at least one segment on a vertical edge"
    );
    let longest = segs
        .iter()
        .max_by(|a, b| a.len.partial_cmp(&b.len).unwrap())
        .unwrap();
    assert!(
        longest.dir[1].abs() > longest.dir[0].abs(),
        "expected vertical-oriented segment, got dir={:?}",
        longest.dir
    );
    assert!(
        longest.len >= 8.0,
        "expected a reasonably long segment, got len={}",
        longest.len
    );
}

#[test]
fn lsd_extractor_rejects_flat_image() {
    let img = ImageF32::new(16, 16);
    let segs = LsdExtractor::new(
        &img,
        0.05,
        std::f32::consts::FRAC_PI_4,
        2.0,
        None,
        LsdOptions::default(),
    )
    .extract();
    assert!(
        segs.is_empty(),
        "no segments should be detected in a flat image, got {:?}",
        segs
    );
}
