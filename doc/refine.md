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

### Segment Refinement (`segment`)

- **Inputs**: gradient buffers for level `l`, a coarse segment from level
  `l+1`, a `ScaleMap` that lifts coordinates, and `segment::RefineParams`.
- **Stages**:
  1. Upscale endpoints and build a padded ROI.
  2. Sample gradients along the segment normal, keeping peaks above the
     magnitude threshold; weight them with a Huber loss.
  3. Fit a weighted line via orthogonal regression; iterate a few times until
     the carrier stabilises.
  4. Scan along the refined line to locate the longest contiguous run of
     inlier pixels whose gradient polarity/orientation matches the line normal.
  5. Promote those endpoints as the refined segment and compute an alignment
     score. Reject if support is too weak.
- **Outputs**: refined segment, average normal-projected gradient score,
  inlier counts, and an `ok` flag.
- **Parameters**: sampling spacing (`delta_s`), normal search half-width
  (`w_perp`), gradient magnitude/orientation thresholds, Huber delta, maximum
  iterations, and minimum inlier fraction.

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
