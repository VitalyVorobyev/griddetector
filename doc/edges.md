# Edges: Gradients and Non‑Maximum Suppression

This document describes the edge processing blocks used in the detector: gradient computation (Sobel/Scharr) and a simple non‑maximum suppression (NMS) that yields sparse edge elements for visualization and coarse processing.

## Goals

- Provide fast, cache‑friendly gradient maps (gx, gy, magnitude) with an orientation proxy.
- Offer a minimal NMS that is easy to tune and serialize for tooling.
- Keep boundary behavior explicit and robust.

## Gradients

- Kernels: 3×3 Sobel and Scharr pairs
  - Sobel X/Y
    - X: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    - Y: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
  - Scharr X/Y (better rotational symmetry)
    - X: [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
    - Y: [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]
- Borders: indices are clamped (replicated), effectively mirroring the nearest valid pixel.
- Outputs:
  - `gx`, `gy`: horizontal and vertical derivatives.
  - `mag`: Euclidean magnitude `sqrt(gx^2 + gy^2)`.
  - `ori_q8`: 8‑bin quantized orientation (π‑periodic) to support fast orientation tests downstream.
- Complexity: O(W·H); memory: three float buffers + 1 byte/pixel.

## Non‑Maximum Suppression (NMS)

- Direction: estimated from `atan2(gy, gx)`, quantized to 4 canonical bins (0°, 45°, 90°, 135°).
- Neighborhood: the two neighbors along the chosen direction are compared to the current pixel.
- Criterion: keep pixel if `mag > neighbor1 && mag > neighbor2 && mag >= threshold`.
- Borders: the 1‑pixel frame is skipped to avoid out‑of‑bounds reads (gradients themselves already clamp borders).
- Output: sparse list of `EdgeElement { x, y, magnitude, direction }` serialized to JSON when needed.

## Usage Hints

- Threshold selection: for images normalized to [0, 1], typical coarse thresholds are 0.05–0.20 depending on noise and blur. In a pyramid, scale thresholds down as levels get coarser.
- Scharr vs Sobel: Scharr reduces directional bias and can be beneficial on fine textures; Sobel is cheaper and sufficient for many tasks.
- Orientation handling: many grid tasks are π‑periodic (unsigned direction). Use `ori_q8` or normalize raw angles modulo π for comparisons.

## Rationale and Trade‑offs

- We intentionally keep NMS simple (no hysteresis) as downstream logic already aggregates support (LSD‑like segments, VPs, refinement).
- The code favors row‑based access to improve cache locality and enable compiler vectorization.
- For best performance on large images, consider parallelizing over rows when the `parallel` feature is enabled elsewhere in the crate.

## References

- Sobel operator: Irwin Sobel and Gary Feldman (1968).
- Scharr operator: H. Scharr (2000), “Optimal Operator for Derivative Estimation in Two Dimensions”.
