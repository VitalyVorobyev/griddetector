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

```sh
cargo run --release --bin grid_demo
```

Coarsest-level inspection helpers:

```sh
cargo run --release --bin coarse_edges config/coarse_edges.json
cargo run --release --bin coarse_segments config/coarse_segments.json
```

LSD options in `config/coarse_segments.json`:

- `magnitude_threshold`, `angle_tolerance_deg`, `min_length`
- Optional guards:
  - `enforce_polarity` (bool): prevent merging opposite‑polarity parallel edges.
  - `limit_normal_span` (bool) and `normal_span_limit_px` (float): reject thick regions by capping perpendicular span.

See [doc/segments.md](doc/segments.md) for details and tuning tips.

Add the library to another project (until on crates.io, use Git):

```sh
cargo add griddetector --git https://github.com/VitalyVorobyev/griddetector
```

Minimal usage example:

```rust
use grid_detector::image::ImageU8;
use grid_detector::{GridDetector, GridParams};
use nalgebra::Matrix3;

// Provide your grayscale 8‑bit image buffer
let (w, h) = (640usize, 480usize);
let gray = vec![0u8; w * h];
let img = ImageU8 { w, h, stride: w, data: &gray };

// Configure detector (set your camera intrinsics)
let mut det = GridDetector::new(GridParams { kmtx: Matrix3::identity(), ..Default::default() });
let res = det.process(img);
println!("found={} latency_ms={:.3}", res.found, res.latency_ms);
```

To enable optional parallelism:

```sh
cargo run --release --features parallel --bin grid_demo
```

## Modules Overview

- `image`: lightweight image views/owners plus I/O helpers (see doc/image.md)
- `pyramid`: grayscale `ImageU8` → multi‑level `ImageF32` pyramid ([doc/pyramid.md](doc/pyramid.md))
- `edges`: Sobel/Scharr gradients and NMS ([doc/edges.md](doc/edges.md))
- `segments`: LSD‑like region growing and PCA line fitting ([doc/segments.md](doc/segments.md))
- `lsd_vp`: coarse vanishing‑point hypothesis ([doc/lsd_vp.md](doc/lsd_vp.md))
- `refine`: coarse‑to‑fine homography refinement ([doc/refine.md](doc/refine.md))
- `detector`: end-to-end pipeline wrapper that builds the pyramid, runs the engine, and recovers pose
- `types`: result and pose structs (`GridResult`, `Pose`)

## Documentation

- Image module: [doc/image.md](doc/image.md)
- Pyramid: [doc/pyramid.md](doc/pyramid.md)
- Edges: [doc/edges.md](doc/edges.md)
- Segments: [doc/segments.md](doc/segments.md)
- LSD→VP: [doc/lsd_vp.md](doc/lsd_vp.md)
- Refinement: [doc/refine.md](doc/refine.md)
- Roadmap: [doc/roadmap.md](doc/roadmap.md)
- Tools: `tools/plot_coarse_edges.py`, `tools/plot_coarse_segments.py`

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
