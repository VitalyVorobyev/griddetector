## Coarse‑to‑Fine Homography Refinement

Refines an initial projective basis `H0` (from LSD→VP) into a stable homography by bundling edges and re‑estimating vanishing points with Huber‑weighted IRLS across an image pyramid.

### Inputs

- Bundled line constraints per level, expressed in the image coordinates of `H0`.
- Initial homography `H0` already rescaled to the current image coordinates.

### Stages

1. Family Assignment
   - From `H`, compute directions to the vanishing points relative to the translation anchor.
   - Assign bundles to `u` or `v` families based on angular proximity (with a tolerance), ensuring minimal family support.

2. Huber‑Weighted VP Estimation (IRLS)
   - For each family, solve normal equations on `ax+by+c≈0` with weights from bundle strength and Huber weights (`δ/|residual|` beyond the inlier region).
   - Update vanishing points; iterate a few times or until convergence.

3. Anchor Update
   - Estimate a stable anchor as the intersection of the heaviest `u` and `v` bundles.

4. Coarse‑to‑Fine Progression
   - Apply steps 1–3 from coarse → fine, using the previous level’s `H` as initialization.
   - Early stop on small relative update; accumulate a confidence score and keep the last good `H` for fallback.

### Outputs

- `h_refined`: refined homography.
- `confidence`: accumulated across levels based on inlier support.
- `inlier_ratio`: last‑level inlier ratio.
- `levels_used`: how many levels contributed.

### Parameters (`RefineParams`)

- `orientation_tol_deg`: angular tolerance when assigning bundles to the two families.
- `huber_delta`: inlier threshold for the Huber loss.
- `max_iterations`: IRLS iterations per level.
- `min_bundles_per_family`: robustness gate to avoid ill-conditioned updates.

### Notes

- Assumes rectified input imagery (no lens distortion).
- Bundling and ROI selection happen in the detector before refinement.
- Metric constraints (equal spacing for square/rectangular grids) are not enforced here; they belong to a later refinement stage.

### Extensions

- Replace family assignment with VP RANSAC before IRLS for stronger outlier rejection.
- Add grid‑spacing residuals to transition from projective to metric homography.
- Introduce temporal smoothing across frames for video input.
