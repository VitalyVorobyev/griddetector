Current State

Pyramid: Functional 5-tap Gaussian + 2× decimation, returns ImageF32 levels (src/pyramid.rs:1).
Gradients: Sobel/Scharr with magnitude + orientation quantization; per-pixel Grad computed (src/edges.rs:11).
LSD-like segments: Region growing on gradient orientation, PCA fit, simple significance test, returns normal-form lines (src/segments.rs:114). Includes basic unit tests.
VP + coarse H0: Orientation histogram → two dominant line families → VP estimation → coarse projective basis H0 and confidence (src/lsd_vp.rs:35).
Detector: Builds pyramid, runs low-res engine once, placeholder refinement, pose-from-H implemented; found always false, unused params (src/detector.rs:43,57,68).
Key Gaps

No coarse-to-fine refinement of H0 to a metric homography honoring equal grid spacing.
H0 is built at the last pyramid level and not rescaled to original coordinates.
No grid indexing (u,v lattice), origin detection, or coverage/quality metrics.
Outlier handling is minimal (no robust fit/RANSAC; no line bundling/merging).
Parallelism optional feature not used; frequent allocations in hot paths.
Params mismatch: canny_low/high unused; confidence not propagated to found.
Sparse tests; no integration/benchmarks and no example images.
Roadmap

Milestone 1: Solid Baseline

Set found when H0 is valid with confidence >= τ (e.g., 0.5) and propagate confidence to result (src/detector.rs:52,68).
Rescale H0 from lowest pyramid level to full-res coordinates by composing with scale matrices.
Add simple integration test using a synthetic chessboard image generator.
Expose a debug flag to return extracted segments and VPs for inspection.
Milestone 2: Coarse-to-Fine Refinement

Introduce refine.rs: robustly refine from H0 at low-res → mid → full. Energy: distance of detected edge bundles to projected grid lines with Huber loss.
Bundle/merge collinear segments; weight by length and gradient magnitude.
Add stopping criteria, confidence update, and failure fallback to last good H0.
Milestone 3: Grid Lattice + Metric Homography

From refined projective basis, estimate metric homography by enforcing equal spacing along u and v.
Infer integer lattice: project edges to u/v, detect 1D peaks (via 1D Hough/peak finding) → offsets + integer indexing → set origin_uv, visible_range, coverage.
Flip found=true when both families have ≥ min inliers and spacing stability is good.
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

Clean GridParams (remove or implement canny_*), document parameters and defaults.
Add more unit tests (edges/pyramid/VP), integration tests on synthetic + sample photos.
Extend GridResult quality metrics (reproj_rmse, coverage, final confidence) and document semantics.
