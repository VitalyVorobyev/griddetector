# Concepts & Data Types

- `ImageU8` / `ImageF32`: lightweight image views/owners.
- `Segment`: sub-pixel endpoints + normal/orientation + length.
- `Bundle`: a group of near-collinear segments within a family.
- `GridParams`: detector knobs (levels, thresholds, schedule, intrinsics).
- `GridResult`: `found`, `H` (3×3), `confidence`, bundles, timings.
- Coordinates: image pixels (x→right, y→down) unless stated; intrinsics `K` map to normalized camera plane when provided.
