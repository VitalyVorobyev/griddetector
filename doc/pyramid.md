# Image Pyramid

Builds a grayscale image pyramid using a separable 5‑tap Gaussian followed by 2× decimation at each level.

## Goals

- Accelerate coarse hypothesis generation (run LSD→VP on small images).
- Support coarse‑to‑fine refinement with robust initialisation.

## Algorithm

1. Level 0 (L0)
   - Convert 8‑bit grayscale to `f32` in `[0, 1]`.

2. For each next level (Lk → Lk+1)
   - Optionally apply separable 1D Gaussian blur with kernel `[1, 4, 6, 4, 1]/16`:
     - Horizontal pass, then vertical pass.
     - Borders use clamping (replicate) via index saturation.
   - Decimate by 2×: pick every other pixel `[2x, 2y]` from the blurred image.

## Properties

- Value range stays within `[0, 1]`.
- Time per level is `O(W×H)`; total cost across levels is ~4/3 of the base image size.
- Memory layout: row‑major, `stride == width` for compact access.

## Usage

```rust
use grid_detector::pyramid::{Pyramid, PyramidOptions};
use grid_detector::image::ImageU8;

let gray = ImageU8 { w, h, stride: w, data: &buffer };
let options = PyramidOptions::new(4);
let pyr = Pyramid::build_u8(gray, options);
// use pyr.levels.last() for coarse processing
```

## Practical Notes

- For 1280×1024 inputs, 3–5 levels balance speed and detection stability.
- The blur is intentionally small and separable for performance; consider SIMD/parallelisation (`rayon`) if profiling shows this as a bottleneck.
- If aliasing is observed at fine textures, slightly increase the blur strength or add a pre‑blur at L0.

## Blur control

- By default, blur is applied before every 2× downscale step (`PyramidOptions::blur_levels = None`).
- Limit blurring to only the first N downscale steps by setting `PyramidOptions::new(levels).with_blur_levels(Some(n))`. Use `Some(0)` to skip blur entirely.
- CLI tools (`coarse_edges`, `coarse_segments`, `lsd_vp_demo`, `grid_demo`) expose this via their `pyramid.blur_levels` configuration entry.
