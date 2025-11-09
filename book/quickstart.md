# Quick Start

Add via Git (until published on crates.io):

```sh
cargo add griddetector --git https://github.com/VitalyVorobyev/griddetector
```

Minimal example:

```rust
use grid_detector::image::ImageU8;
use grid_detector::{GridDetector, GridParams};
use nalgebra::Matrix3;

let (w, h) = (640usize, 480usize);
let gray = vec![0u8; w * h];
let img = ImageU8 { w, h, stride: w, data: &gray };

let mut det = GridDetector::new(GridParams { kmtx: Matrix3::identity(), ..Default::default() });
let res = det.process(img);

println!("found={} latency_ms={:.3}", res.grid.found, res.grid.latency_ms);
```

Run the demo:

```sh
cargo run --release --bin grid_demo -- data/sample.png --save-debug out/run1
```

* (Example mirrors the README usage and demo pattern.)
* [oai_citation:13‡GitHub](https://github.com/VitalyVorobyev/griddetector)
