# LSD→VP Coarse Hypothesis

The LSD→VP engine produces a coarse projective basis $H_0$ from line segments at a single (usually coarsest) pyramid level. It is designed to be fast and robust enough to seed later coarse‑to‑fine refinement.

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
   - Each segment contributes a normalized line constraint $ax + by + c ≈ 0$.
   - Solve a weighted least squares for $(x, y)$ using the per‑segment length as weight.
   - If the normal matrix is near‑singular (e.g., truly parallel lines), fall back to a point at infinity in the average tangent direction.

4. Hypothesis Composition
   - Form $H_0 = [vp_u | vp_v | x_0]$ where $x0$ is the image center in homogeneous coordinates.
   - This $H_0$ lives in the pyramid level’s coordinates; rescale to the full image size when returning to the caller.

5. Confidence
   - Heuristic score combining: support in each family (capped), and angular separation between the two peaks.

## Output

- `H0`: coarse projective basis for the grid.
- `confidence`: 0..1 score for downstream gating and blending with refinement.

## Notes and Tips

- Use the coarsest level for speed and stability; refinement should handle details across levels.
- Ensure segment extraction thresholds are appropriate for the downscaled level.
- For perfectly axis‑aligned checkerboards (no perspective), the fallback VP at infinity ensures a usable `H0`.

# Projective geometry insight

Think of a rectified grid plane with homogeneous coordinates $X=(u,v,1)^\top$. Choose the projective basis on that plane as:

* $U_\infty=(1,0,0)^\top$: the point at infinity along the u-axis (all u-parallel lines meet here),
* $V_\infty=(0,1,0)^\top$: the point at infinity along the v-axis,
* $O=(0,0,1)^\top$: the finite origin of the grid.

In the image:
* the image of $U_\infty$ is the vanishing point $v_u$,
* the image of $V_\infty$ is the vanishing point $v_v$,
* the image of $O$ is some finite point $x_0$ (e.g., where you want the grid origin to land).

A planar projective map (homography) is fully determined by where it sends a projective basis. If we build

$$
H = \big[\, v_u \;\; v_v \;\; x_0 \,\big] \;\;\in \mathbb{R}^{3\times 3},
$$

then by construction it satisfies

$$
H\,U_\infty \sim v_u,\quad
H\,V_\infty \sim v_v,\quad
H\,O \sim x_0,
$$

and for any rectified point $X=(u,v,1)^\top$,

$$
x \;\sim\; H\,X \;=\; u\,v_u \;+\; v\,v_v \;+\; 1\cdot x_0.
$$

That is exactly a rectified→image map: points with increasing $u$ move along rays toward $v_u$; points with increasing $v$ move toward $v_v$. Lines of constant $u$ map to image lines through $v_v$, and lines of constant $v$ map to image lines through $v_u$ — which is what you see in perspective images of a rectilinear grid.

Consequences:
* To push rectified points to the image, use $x \sim H\,X$.
* To pull image points back to the rectified frame, use $X \sim H^{-1}x$.
* For lines, use duals: image → rectified is $\ell_{\text{rect}} \sim H^{\mathsf T}\ell_{\text{img}}$, rectified → image is $\ell_{\text{img}} \sim H^{-\mathsf T}\ell_{\text{rect}}$.

This matches (a) how vanishing points arise as images of points at infinity along grid directions, and (b) why using $[v_u\,|\,v_v\,|\,x_0]$ makes $H$ a rectified→image transform by definition.