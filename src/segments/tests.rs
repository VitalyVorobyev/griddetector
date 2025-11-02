use super::extractor::LsdExtractor;
use super::*;
use crate::image::ImageF32;
use nalgebra::Vector3;

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

fn make_test_segment(angle: f32) -> Segment {
    let dir = [angle.cos(), angle.sin()];
    let center = [0.0f32, 0.0f32];
    let half_len = 5.0f32;
    let p0 = [center[0] - dir[0] * half_len, center[1] - dir[1] * half_len];
    let p1 = [center[0] + dir[0] * half_len, center[1] + dir[1] * half_len];
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let len = (dx * dx + dy * dy).sqrt();
    let normal = [-dir[1], dir[0]];
    let c = -(normal[0] * center[0] + normal[1] * center[1]);
    Segment {
        id: SegmentId(0),
        p0,
        p1,
        dir,
        len,
        line: Vector3::new(normal[0], normal[1], c),
        avg_mag: 1.0,
        strength: 1.0,
    }
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
fn bundle_segments_merges_across_orientation_bins() {
    let orientation_tol = 0.2;
    let dist_tol = 0.5;
    let min_weight = 0.0;

    let mut nbins = (std::f32::consts::PI / orientation_tol).ceil() as usize;
    nbins = nbins.clamp(8, 90);
    let bin_width = std::f32::consts::PI / nbins as f32;

    let boundary = bin_width;
    let delta = 0.05 * bin_width;
    let angle_a = boundary - delta;
    let angle_b = boundary + delta;

    let idx_a = {
        let th = crate::angle::normalize_half_pi(angle_a);
        let mut idx = (th / bin_width).floor() as isize;
        if idx < 0 {
            idx += nbins as isize;
        }
        (idx as usize).min(nbins - 1)
    };
    let idx_b = {
        let th = crate::angle::normalize_half_pi(angle_b);
        let mut idx = (th / bin_width).floor() as isize;
        if idx < 0 {
            idx += nbins as isize;
        }
        (idx as usize).min(nbins - 1)
    };
    assert_ne!(
        idx_a, idx_b,
        "segments should start in different orientation bins"
    );

    let seg_a = make_test_segment(angle_a);
    let seg_b = make_test_segment(angle_b);
    let segments = vec![seg_a, seg_b];

    let bundles = bundle_segments(&segments, orientation_tol, dist_tol, min_weight);
    assert_eq!(bundles.len(), 1, "segments near bin boundary should merge");
    assert!(
        (bundles[0].weight - 2.0).abs() < 1e-3,
        "expected merged bundle weight to reflect both segments"
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
