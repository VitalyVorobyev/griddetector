# Image Module: Views, Buffers, Iterators, and I/O

This document explains the image layer used by the detector, covering owned and borrowed image types, iteration strategies, and I/O helpers.

## Goals

- Keep hot paths simple and fast with cache‑friendly row access.
- Make ownership and mutability explicit (borrowed views vs. owned buffers).
- Provide a contiguous fast path for whole‑image operations.

## Types

- `ImageU8<'a>` (view)
  - Borrowed read‑only view over 8‑bit grayscale data
  - Fields: `w`, `h`, `stride`, `data: &'a [u8]`
  - Use case: input image for pyramid L0 conversion
- `ImageF32` (owned)
  - Owned, mutable float buffer for processing
  - Row‑major layout; `stride == w`
  - Use case: pyramid levels, gradients, temporary buffers

## Traits and Iterators

- `ImageView` (read‑only)
  - `row(y) -> &[Pixel]` and `rows()` iterator
  - `as_slice()` and `pixels()` when `stride == w`
- `ImageViewMut`
  - `row_mut(y) -> &mut [Pixel]` and `rows_mut()` iterator
  - `as_mut_slice()` and `pixels_mut()` when `stride == w`
- Design: prefer row iteration for stencil operations; use flat iterators for global reductions and whole‑image transforms.

## I/O Helpers

- `load_grayscale_image(path) -> GrayImageU8`
  - Loads an image via `image` crate and converts to 8‑bit gray
  - Convert to `ImageU8` with `.as_view()`
- `save_grayscale_f32(&ImageF32, path)` and `save_grayscale_u8(&GrayImageU8, path)`
  - Save to PNG; values are clamped appropriately
- `write_json_file(path, &value)`
  - Pretty‑prints a serializable object to disk; ensures parent directories exist

## Best Practices

- Convert input to `ImageF32` early (pyramid L0), then keep everything in float.
- Use `rows()`/`row_mut()` in kernels for better cache locality; avoid per‑pixel `get/set` in hot loops.
- When `is_contiguous()`, prefer `pixels()`/`pixels_mut()` for whole‑image passes.
- Avoid implementing `Iterator` on image types; provide iterator‑producing methods instead.

## Future Extensions

- Optional unsafe accessors (`get_unchecked`, `row_unchecked`) for peak performance where benchmarks justify it.
- Owned 8‑bit buffer type if in‑place u8 processing becomes common.
- Parallel row iteration (gated behind a `parallel` feature) for large images.
