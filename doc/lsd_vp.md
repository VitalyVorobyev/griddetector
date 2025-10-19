# LSD→VP Coarse Hypothesis

The LSD→VP engine produces a coarse projective basis `H0` from line segments at a single (usually coarsest) pyramid level. It is designed to be fast and robust enough to seed later coarse‑to‑fine refinement.

## Inputs

- An image level (`ImageF32`) from the pyramid.
- Detected line segments from the lightweight LSD‑like extractor.

## Steps

1. Orientation Histogram
   - For each segment, compute its tangent angle in `[0, π)` and accumulate a histogram weighted by segment length.
   - Pick the two dominant peaks, enforcing a minimum separation (≈2× tolerance).

2. Family Assignment
   - Soft‑assign each segment to the closer peak if within the angular tolerance.
   - Require basic support: both families should have enough segments.

3. Vanishing Point Estimation
   - Each segment contributes a normalized line constraint `ax + by + c ≈ 0`.
   - Solve a weighted least squares for `(x, y)` using the per‑segment length as weight.
   - If the normal matrix is near‑singular (e.g., truly parallel lines), fall back to a point at infinity in the average tangent direction.

4. Hypothesis Composition
   - Form `H0 = [vpu | vpv | x0]` where `x0` is the image center in homogeneous coordinates.
   - This `H0` lives in the pyramid level’s coordinates; rescale to the full image size when returning to the caller.

5. Confidence
   - Heuristic score combining: support in each family (capped), and angular separation between the two peaks.

## Output

- `H0`: coarse projective basis for the grid.
- `confidence`: 0..1 score for downstream gating and blending with refinement.

## Notes and Tips

- Use the coarsest level for speed and stability; refinement should handle details across levels.
- Ensure segment extraction thresholds are appropriate for the downscaled level.
- For perfectly axis‑aligned checkerboards (no perspective), the fallback VP at infinity ensures a usable `H0`.

