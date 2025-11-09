## Refinement Overview

The `refine` module exposes two cooperating passes that tighten the detector’s
coarse hypotheses:

1. `segment::refine_segment` upsamples coarse line segments to the next finer
   pyramid level and re-centres them on strong gradient support. The result is a
   set of clean, polarity-consistent edge measurements with accurate extent.
2. `homography::Refiner` runs a Huber-weighted IRLS update on the bundled
   segments, refining the coarse projective basis delivered by the LSD→VP stage.

The detector typically alternates between these passes per level: refine the
segments, rebuild bundles, then update the homography.

- **Segment filtering** – Before entering refinement, the detector rejects
  coarse segments whose orientation deviates too far from the initial
  vanishing-point directions or whose lines miss the corresponding vanishing
  points by more than a configurable residual. This keeps the IRLS stage
  better conditioned and surfaces useful diagnostics.
- **Refinement schedule** – Multiple IRLS passes can be attempted when the
  Frobenius improvement exceeds a configurable threshold. Each pass reuses the
  previously bundled constraints, enabling a light-weight consistency check
  without rebuilding the entire pyramid.

### Segment Refinement (`segment`)

- **Inputs**: gradient buffers for level `l`, a coarse segment from level
  `l+1`, a `ScaleMap` that lifts coordinates, and `segment::SegmentRefineParams`.
- **Stages**:
  1. Upscale the coarse endpoints to level `l`.
  2. Evaluate a handful of seeds along the predicted segment and snap the best
     one onto the local edge by running a tiny (±2–3 px) search along the
     gradient normal.
  3. Grow the support forward and backward along the tangent, repeatedly
     predicting the next point, refining along the normal, and stopping when the
     gradient weakens or the track drifts off the edge.
  4. Fit a line to all refined samples, then project the extreme points onto the
     line to obtain subpixel endpoints.
- **Outputs**: refined segment geometry, mean gradient magnitude along the
  support, support count, and diagnostics (normal refinements, tangent steps,
  gradient samples).
- **Parameters**: normal search radius/step, tangent step and iteration cap,
  gradient magnitude gate, displacement tolerances, minimum support size, and
  number of seed anchors.

### Homography Refinement (`homography`)

Refines an initial projective basis `H0` into a stable homography by bundling
the gradient-aligned segments from the previous step and re-estimating
vanishing points across the pyramid.

- **Inputs**: bundled line constraints per level (in the current image
  coordinates) and an initial homography `H0`.
- **Stages**:
  1. **Family Assignment** – Compute the vanishing directions implied by the
     current homography and assign bundles to the `u`/`v` families based on
     angular proximity (within `orientation_tol_deg`).
  2. **Huber-Weighted VP Estimation** – For each family, solve the normal
     equations on `ax + by + c ≈ 0`, weighting by bundle strength and the Huber
     schedule (`δ / |residual|` beyond the inlier region). Iterate until
     convergence or the iteration cap.
  3. **Anchor Update** – Intersect the strongest `u` and `v` bundles to obtain
     a stable translation anchor column.
  4. **Coarse-to-Fine Progression** – Repeat from coarse → fine, measuring
     Frobenius improvement; stop early on negligible change. The last good
     homography and confidence accumulate across levels.
- **Outputs**: refined homography `h_refined`, aggregated `confidence`,
  last-level `inlier_ratio`, number of `levels_used`, and per-level diagnostics.
- **Parameters** (`homography::RefineParams`): orientation tolerance, Huber
  delta, maximum IRLS iterations, and minimum bundles per family.

### Detector Integration

`GridDetector::process_with_diagnostics` now drives both refinement stages across
all pyramid levels. After the LSD+outlier stage, the coarsest-level segments are
refined purely from image gradients (no homography prior) using
`segment::refine_segment`. Vanishing points and the coarse homography are then
re-estimated from these gradient-tightened segments before continuing with the
coarse→fine bundling cascade. Subsequent levels refine segments toward full
resolution and feed the resulting bundles to the homography IRLS pass (fine →
coarse). The CLI accepts `--save-debug <dir>` to dump the generated pyramid
levels, per-level bundles, and refinement diagnostics for inspection.

### Module Layout

- `src/refine/segment.rs` – gradient-driven line refinement.
- `src/refine/homography.rs` – bundle-to-homography IRLS refiner.
- `src/refine/{anchor,families,irls,types}.rs` – helpers shared by the
  homography pass.
- `src/refine/mod.rs` – façade re-exporting both refinement entry points.

### Notes & Extensions

- Assumes lens distortion is compensated before refinement.
- Bundling/ROI selection happens upstream in the detector.
- To extend: consider RANSAC-based family assignment, add metric constraints
  (grid spacing) for affine/metric upgrade, or introduce temporal smoothing for
  video sequences.
