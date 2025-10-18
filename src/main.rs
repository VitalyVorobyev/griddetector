use grid_detector::types::ImageU8;
use grid_detector::{GridDetector, GridParams};
use nalgebra::Matrix3;

fn main() {
    // Demo stub: creates a fake 8-bit image buffer and runs the detector
    let w = 640usize;
    let h = 480usize;
    let stride = w; // tightly packed
    let gray = vec![0u8; w * h];
    let img = ImageU8 {
        w,
        h,
        stride,
        data: &gray,
    };

    let mut det = GridDetector::new(GridParams {
        kmtx: Matrix3::identity(),
        ..Default::default()
    });
    let res = det.process(img);
    println!("found={} latency_ms={:.3}", res.found, res.latency_ms);
}
