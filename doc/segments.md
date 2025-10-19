# Segment Extraction (LSD‑like)

This module implements a lightweight, LSD‑inspired line segment extractor optimized for edge‑based grid/chessboard detection and multi‑scale refinement.

## Overview

Given a grayscale image level (from the pyramid), we:

1. Compute gradients via Sobel/Scharr to obtain `gx`, `gy`, and magnitude.
2. Region‑grow from seeds using orientation consistency of gradient normals, while enforcing a magnitude threshold.
3. Fit a principal direction to the region with a 2×2 covariance eigen‑decomposition (PCA).
4. Project region points onto the principal axis to form endpoints.
5. Emit a normalized line in ax+by+c=0 form and basic significance metrics.

The output segments layer is intentionally small and clean; it targets long, coherent edges that drive VP estimation and refinement.

## Data and Parameters

- `mag_thresh`: minimum gradient magnitude to seed/grow a region (level scale).
- `angle_tol`: angular tolerance (radians) against the seed normal during growth.
- `min_len`: minimum acceptable segment length (level pixels).

Each `Segment` carries:

- `p0`, `p1`: endpoints along the principal axis.
- `dir`: unit tangent direction.
- `len`: length along the tangent.
- `line`: normalized normal form `ax+by+c=0`.
- `avg_mag`: average gradient magnitude in the region.
- `strength`: `len * avg_mag` (used as a weight later).

## Algorithm Details

1. Orientation Normalization: use angles modulo π so opposite directions are treated the same for grid lines (`angle::normalize_half_pi`).
2. Seed Selection: iterate pixels once; if not used and above `mag_thresh`, start a region with its normalized gradient orientation as the seed.
3. Region Growth: visit 8‑neighbors; add pixels whose magnitude exceeds `mag_thresh` and whose orientation is within `angle_tol` of the seed.
4. PCA Fit: maintain online sums for centroid and covariance, then eigen‑decompose to obtain the major eigenvector as the tangent.
5. Endpoints: project member pixels onto the tangent axis; min/max define `p0`/`p1`.
6. Normal Form: a unit normal is `n = [−ty, tx]`, with `c = −n·center`.
7. Validation: require region size ≥ N, `len ≥ min_len`, and an aligned‑fraction threshold (default ~0.6) before emitting.

## Complexity

- Each pixel is visited at most once during region growth ⇒ O(W·H) per level.
- Eigen‑decomposition is 2×2 and cheap; endpoints are linear in region size.

## Tips

- Scale thresholds by level; e.g., reduce `mag_thresh` slightly at coarser levels and increase `min_len` at finer ones.
- For noisy inputs, pre‑blur (pyramid already uses a 5‑tap separable Gaussian).

## Extending

- Add endpoint trimming using orthogonal distance thresholds.
- Merge small adjacent regions with consistent orientation to reduce fragmentation.

