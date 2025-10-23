Current State

Pyramid: Functional 5-tap Gaussian + 2× decimation, returns ImageF32 levels (src/pyramid.rs:1).
Gradients: Sobel/Scharr with magnitude + orientation quantization; per-pixel Grad computed (src/edges.rs:11).
LSD-like segments: Region growing on gradient orientation, PCA fit, simple significance test, returns normal-form lines (src/segments.rs:114). Includes basic unit tests.
VP + coarse H0: Orientation histogram → two dominant line families → VP estimation → coarse projective basis H0 and confidence (src/lsd_vp.rs:35).
Detector: Multi-level pipeline (pyramid, LSD→VP, per-level segment refinement, bundling, homography IRLS, pose recovery) with confidence gating (src/detector.rs:1).
Key Gaps

No grid indexing (u,v lattice), origin detection, or coverage/quality metrics.
Metric upgrade absent: homography not constrained by equal spacing or non-planar grids.
Outlier handling is minimal (no robust fit/RANSAC; bundle family assignment deterministic).
Parallelism optional feature not used; frequent allocations in hot paths.
Sparse tests; no integration/benchmarks and no example images.
Roadmap

Milestone 1: Solid Baseline

Add simple integration test using a synthetic chessboard image generator.
Expose grid lattice quality metrics in `GridResult` (coverage, reprojection RMSE).
Document new configuration knobs (LSD/segment refinement/bundling) with examples.

Milestone 2: Metric Upgrade

Extend refinement with grid spacing constraints (affine/metric upgrade).
Infer lattice indices (u,v), origin detection, and coverage estimation.
Provide hooks for non-planar grids / local warps (beyond single homography).
Milestone 3: Corner Extraction & Grid Lines

Derive per-cell grid lines/corners from the metric upgrade, exposing precise grid coordinates.
Track per-line confidence and residuals; surface optional smoothing for wavy boards/non-planar grids.
Plumb lattice metadata through the CLI/debug outputs for downstream calibration tooling.
Milestone 4: Robustness + Outliers

RANSAC/IRLS for VP estimation (line intersections) before refinement.
Improve orientation histogram (smoothing, peak separation checks, fallback strategy).
Strengthen LSD extraction: adaptive mag_thresh per level (e.g., percentile), raise significance checks, and line endpoint trimming.
Milestone 5: Performance

Preallocate buffers and reuse across frames; keep a detector workspace.
Parallelize gradients + region seeding across tiles with rayon feature.
Optional std::simd for Sobel and Gaussian; measure wins before committing.
Add Criterion benchmarks for: pyramid, gradients, LSD, VP, full pipeline.
Milestone 6: API, Tests, Docs

Document GridParams parameters and defaults.
Add more unit tests (edges/pyramid/VP), integration tests on synthetic + sample photos.
Extend GridResult quality metrics (reproj_rmse, coverage, final confidence) and document semantics.
