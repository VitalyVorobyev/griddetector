# griddetector

[![CI](https://github.com/VitalyVorobyev/griddetector/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/griddetector/actions/workflows/ci.yml)
[![Release](https://github.com/VitalyVorobyev/griddetector/actions/workflows/release.yml/badge.svg)](https://github.com/VitalyVorobyev/griddetector/actions/workflows/release.yml)
[![Security Audit](https://github.com/VitalyVorobyev/griddetector/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/griddetector/actions/workflows/audit.yml)

Edge-based grid/chessboard detector written in Rust. It builds an image pyramid, extracts line segments (LSD‑like), groups them into two dominant line families to estimate vanishing points, composes a coarse homography, filters outliers, refines across the pyramid, and recovers camera pose.

This repo contains both a library crate (`grid_detector`) and a tiny demo binary (`grid_demo`). The public API is intentionally small and focused.

## Pipeline Overview

1. Pyramid
   - Convert 8‑bit grayscale to float and build a multi‑level pyramid with 2× decimation per level.
   - Optional blur limiting: apply the 5‑tap Gaussian only to the first `N` downscale steps.

2. LSD→VP (coarsest level)
   - Detect long, coherent line segments with an LSD‑like extractor (orientation growth → PCA fit → significance tests).
   - Build an orientation histogram in [0, π); pick two dominant peaks and estimate the family vanishing points.
   - Compose a coarse projective basis `H0 = [vpu | vpv | x0]` (anchor `x0` at image centre).

3. Segment Outlier Filtering
   - Reject coarse segments outside an angular margin beyond the LSD tolerance.
   - Gate segments whose lines pass far from the family vanishing point (homography residual).

4. Coarse‑to‑Fine Refinement
   - For each level from coarse → fine:
     - Refine segments at the next finer level using gradient support (normal probing, orthogonal fit, endpoint search).
     - Bundle near‑collinear constraints; thresholds can be level‑invariant or full‑resolution‑invariant.
   - Run a Huber‑weighted IRLS update for the vanishing points and anchor; stop early on negligible Frobenius improvement.

5. Reporting
   - Return the refined homography and confidence; surface detailed diagnostics (pyramid, LSD, filtering, bundling, refinement).

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

LSD→VP coarse hypothesis demo (writes the coarsest-level image and a JSON report with
segment families and the estimated projective basis):

```sh
cargo run --release --bin lsd_vp_demo config/lsd_vp_demo_sample.json
```

VP outlier classification and single-level refinement demo (produces per-segment labels,
coarse/refined models, and timing metrics):

```sh
cargo run --release --bin vp_outlier_demo config/vp_outlier_demo_sample.json
```

Dump intermediate artifacts (pyramid levels, diagnostics, bundles) for debugging:

```sh
cargo run --release --bin grid_demo -- data/sample.png --save-debug out/debug_run
```

Visualize the demo output (requires `matplotlib`, `numpy`, `Pillow`):

```sh
python tools/plot_lsd_vp.py \
    --image out/lsd_vp_coarsest.png \
    --result out/lsd_vp_result.json \
    --save out/lsd_vp_overlay.png
```

Visualize VP outlier demo output (requires `matplotlib`, `numpy`, `Pillow`):

```sh
python tools/plot_vp_outlier_demo.py \
    --image out/vp_outlier_demo/coarsest.png \
    --result out/vp_outlier_demo/result.json \
    --save out/vp_outlier_demo/overlay.png
```

LSD options in `config/coarse_segments.json`:

- `magnitude_threshold`, `angle_tolerance_deg`, `min_length`
- Optional guards:
  - `enforce_polarity` (bool): prevent merging opposite‑polarity parallel edges.
  - `normal_span_limit_px` (float, optional): reject thick regions by capping perpendicular span. Omit to disable.

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
- `lsd_vp`: coarse vanishing-point hypothesis ([doc/lsd_vp.md](doc/lsd_vp.md))
- `refine`: coarse‑to‑fine homography refinement ([doc/refine.md](doc/refine.md))
- `detector`: end‑to‑end pipeline wrapper with outlier filtering, multi‑level segment refinement, bundling, IRLS, and pose recovery
- `types`: result and pose structs (`GridResult`, `Pose`)

## Configuration

The `grid_demo` accepts a JSON config (see `config/sample_config.json`). Relevant `grid` entries:

```json
{
  "grid": {
    "pyramid_levels": 4,
    "pyramid_blur_levels": null,
    "confidence_thresh": 0.35,
    "enable_refine": true,
    "refinement_schedule": { "passes": 1, "improvement_thresh": 0.0005 },
    "lsd_vp": { "mag_thresh": 0.05, "angle_tol_deg": 22.5, "min_len": 4.0 },
    "outlier_filter": { "angle_margin_deg": 8.0, "line_residual_thresh_px": 1.5 },
    "bundling": {
      "orientation_tol_deg": 22.5,
      "merge_dist_px": 1.5,
      "min_weight": 3.0,
      "scale_mode": "full_res"
    }
  }
}
```

Tips:
- Start by tuning `lsd_vp.mag_thresh` and `lsd_vp.min_len` for your scale.
- Use `bundling.scale_mode = "full_res"` to keep thresholds consistent across levels.
- Allow `refinement_schedule.passes = 2` when the coarse hypothesis is weak.

## Documentation

- Image module: [doc/image.md](doc/image.md)
- Pyramid: [doc/pyramid.md](doc/pyramid.md)
- Edges: [doc/edges.md](doc/edges.md)
- Segments: [doc/segments.md](doc/segments.md)
- LSD→VP: [doc/lsd_vp.md](doc/lsd_vp.md)
- Refinement: [doc/refine.md](doc/refine.md)
- Roadmap: [doc/roadmap.md](doc/roadmap.md)
- Tools: `tools/plot_coarse_edges.py`, `tools/plot_coarse_segments.py`, `tools/plot_lsd_vp.py`, `tools/plot_vp_outlier_demo.py`

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
