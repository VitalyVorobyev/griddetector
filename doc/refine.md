## Coarse‑to‑Fine Homography Refinement

Refines an initial projective basis `H0` (from LSD→VP) into a stable homography by bundling edges and re‑estimating vanishing points with Huber‑weighted IRLS across an image pyramid.

### Inputs

- Pyramid levels (`ImageF32`), coarse‐to‐fine.
- Initial homography `H0` already rescaled to the current image coordinates.

### Stages

1. Segment Extraction
   - Use the LSD‑like extractor at each level with level‑aware thresholds.

2. Edge Bundling
   - Merge collinear, nearby segments into bundles.
   - Each bundle stores a normalized line `ax+by+c=0`, a representative center, and a weight `strength = len×avg_mag`.

3. Family Assignment
   - From `H`, compute directions to the vanishing points relative to the translation anchor.
   - Assign bundles to `u` or `v` families based on angular proximity (with a tolerance), ensuring minimal family support.

4. Huber‑Weighted VP Estimation (IRLS)
   - For each family, solve normal equations on `ax+by+c≈0` with weights from bundle strength and Huber weights (`δ/|residual|` beyond the inlier region).
   - Update vanishing points; iterate a few times or until convergence.

5. Anchor Update
   - Estimate a stable anchor as the intersection of the heaviest `u` and `v` bundles.

6. Coarse‑to‑Fine Progression
   - Apply steps 1–5 from coarse → fine, using the previous level’s `H` as initialization.
   - Early stop on small relative update; accumulate a confidence score and keep the last good `H` for fallback.

### Outputs

- `h_refined`: refined homography.
- `confidence`: accumulated across levels based on inlier support.
- `inlier_ratio`: last‑level inlier ratio.
- `levels_used`: how many levels contributed.

### Parameters (`RefineParams`)

- `base_mag_thresh`, `base_min_len`: scale with level size (coarse→smaller mag, fine→longer min_len).
- `orientation_tol_deg`: bundle/assignment tolerance.
- `merge_dist_px`: line proximity for bundling.
- `huber_delta`: inlier threshold for Huber loss.
- `max_iterations`: IRLS iterations per level.
- `min_bundle_weight`, `min_bundles_per_family`: robustness gates.

### Notes

- Assumes rectified input imagery (no lens distortion).
- Confidence blending with the coarse hypothesis is handled in the detector (`combine_confidence`).
- Metric constraints (equal spacing for square/rectangular grids) are not enforced here; they belong to a later refinement stage.

### Extensions

- Replace family assignment with VP RANSAC before IRLS for stronger outlier rejection.
- Add grid‑spacing residuals to transition from projective to metric homography.
- Introduce temporal smoothing across frames for video input.

