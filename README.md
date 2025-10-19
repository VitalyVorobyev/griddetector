# griddetector

[![CI](https://github.com/VitalyVorobyev/griddetector/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/griddetector/actions/workflows/ci.yml)
[![Release](https://github.com/VitalyVorobyev/griddetector/actions/workflows/release.yml/badge.svg)](https://github.com/VitalyVorobyev/griddetector/actions/workflows/release.yml)
[![Security Audit](https://github.com/VitalyVorobyev/griddetector/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/griddetector/actions/workflows/audit.yml)

Edge-based grid/chessboard detector written in Rust. It builds an image pyramid, extracts line segments (LSD‑like), groups them into two dominant line families to estimate vanishing points, composes a coarse homography, and recovers camera pose.

This repo contains both a library crate (`grid_detector`) and a tiny demo binary (`grid_demo`). The public API is intentionally small and focused.

## Features

- Image pyramid with separable 5‑tap Gaussian and 2× decimation
- Fast gradients: Sobel (default) and Scharr kernels
- Lightweight LSD‑like region growing with PCA fitting and significance tests
- Vanishing‑point based coarse homography hypothesis (H₀)
- Pose recovery from homography and intrinsics (R,t)
- Pure Rust, `nalgebra` for linear algebra; optional `rayon` feature for parallelism

## Status

Early stage, under active development. APIs may change. Expect rough edges and ongoing improvements to the LSD/VP pipeline and the refinement stage.

## Quick Start

Build the demo binary:

```
cargo run --release --bin grid_demo
```

Add the library to another project (until on crates.io, use Git):

```
cargo add griddetector --git https://github.com/yourname/griddetector
```

Minimal usage example:

```rust
use grid_detector::image::ImageU8;
use grid_detector::{GridDetector, GridParams};
use nalgebra::Matrix3;

fn main() {
    // Provide your grayscale 8‑bit image buffer
    let (w, h) = (640usize, 480usize);
    let gray = vec![0u8; w * h];
    let img = ImageU8 { w, h, stride: w, data: &gray };

    // Configure detector (set your camera intrinsics)
    let mut det = GridDetector::new(GridParams { kmtx: Matrix3::identity(), ..Default::default() });
    let res = det.process(img);
    println!("found={} latency_ms={:.3}", res.found, res.latency_ms);
}
```

To enable optional parallelism:

```
cargo run --release --features parallel --bin grid_demo
```

## Modules Overview

- `image`: lightweight image views/owners plus I/O helpers (see doc/image.md)
- `pyramid`: grayscale `ImageU8` → multi‑level `ImageF32` pyramid
- `edges`: Sobel/Scharr gradient utilities and orientation quantization
- `segments`: LSD‑like region growing and PCA line fitting
- `lsd_vp`: coarse vanishing‑point engine that returns an H₀ hypothesis
- `detector`: end‑to‑end pipeline wrapper that builds the pyramid, runs the engine, and recovers pose
- `types`: result and pose structs (`GridResult`, `Pose`)

## Roadmap

- Robust refinement from H₀ to metric homography with spacing constraints
- Better outlier rejection and edge bundling
- Confidence scoring and quality metrics
- Example images, tests, and benchmarks

## Contributing

Issues and PRs are welcome. Please open an issue to discuss significant changes.

## Continuous Integration

- **CI** (`ci.yml`): runs on pushes and PRs targeting `main`, checking formatting, Clippy lints (all targets & features), docs, tests, and `cargo package`.
- **Release** (`release.yml`): fires on tags matching `v*`, revalidates the workspace, builds release binaries, packages the crate, and publishes GitHub release assets (binary tarball, `.crate`, checksums).
- **Security Audit** (`audit.yml`): scheduled weekly (Mondays 05:00 UTC) and on manual dispatch, running `cargo audit` against the latest advisory DB.

## License

MIT. See `LICENSE`.
