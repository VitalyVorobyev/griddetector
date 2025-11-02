# VP & Coarse Homography

**Inputs:** dominant orientation families and their segment sets.  
**Outputs:** vanishing points `vp_u`, `vp_v` and coarse homography `H0`.

Algorithm sketch:
1. Estimate `vp_u` and `vp_v` with robust line intersection (Huber weights on angular residuals).
2. Handle degeneracies (vp at infinity) explicitly.
3. Compose `H0` from `{vp_u, vp_v, x0}`; ensure a consistent normalization and coordinate convention.

Knobs:
- `lsd_vp.angle_tol_deg`, `lsd_vp.min_len`, etc.

Diagnostics to log:
- VP locations, family sizes, residual histograms, and `H0` conditioning.
