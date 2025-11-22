use super::filters::{SeparableFilter, StaticSeparableFilter};
use super::options::PyramidOptions;
use crate::image::{ImageF32, ImageU8, ImageView, ImageViewMut};

use serde::Serialize;
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct Pyramid {
    pub levels: Vec<ImageF32>,
}

impl Pyramid {
    pub fn build_f32(image: ImageF32, options: PyramidOptions) -> Self {
        assert!(options.levels >= 1, "pyramid requires at least one level");
        let mut levels = Vec::with_capacity(options.levels);
        levels.push(image);
        if options.levels == 1 {
            return Self { levels };
        }

        let blur_limit = options.blur_levels.min(options.levels.saturating_sub(1));
        let mut horiz_cache = Vec::new();
        let mut cached_rows = Vec::new();
        for lvl in 1..options.levels {
            let prev = levels.last().expect("previous level available");
            let (nw, nh) = (prev.w.div_ceil(2), prev.h.div_ceil(2));
            let mut down = ImageF32::new(nw, nh);
            let use_blur = lvl <= blur_limit;
            if use_blur {
                downsample_with_filter(
                    prev,
                    &mut down,
                    options.filter,
                    &mut horiz_cache,
                    &mut cached_rows,
                );
            } else {
                downsample_without_filter(prev, &mut down);
            }
            levels.push(down);
        }

        Self { levels }
    }

    /// Build a pyramid from an 8-bit grayscale input using the provided options.
    pub fn build_u8(gray: ImageU8<'_>, options: PyramidOptions) -> Self {
        assert!(options.levels >= 1, "pyramid requires at least one level");
        let image_l0 = convert_l0(gray);

        Pyramid::build_f32(image_l0, options)
    }

    pub fn scale_for_level(&self, level: &ImageF32) -> f32 {
        if let Some(index) = self.levels.iter().position(|l| std::ptr::eq(l, level)) {
            1.0 / (2u32.pow(index as u32) as f32)
        } else {
            1.0
        }
    }
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct PyramidResult {
    #[serde(skip)]
    pub pyramid: Pyramid,
    pub elapsed_ms: f64,
    pub elapsed_convert_l0_ms: f64,
}

pub fn build_pyramid(gray: ImageU8<'_>, options: PyramidOptions) -> PyramidResult {
    let l0_start = Instant::now();
    let l0_image = convert_l0(gray);
    let elapsed_convert_l0_ms = l0_start.elapsed().as_secs_f64() * 1000.0;

    let pyr_start = Instant::now();
    let pyramid = Pyramid::build_f32(l0_image, options);
    let elapsed_ms = pyr_start.elapsed().as_secs_f64() * 1000.0;

    PyramidResult {
        pyramid,
        elapsed_ms,
        elapsed_convert_l0_ms,
    }
}

fn convert_l0(gray: ImageU8<'_>) -> ImageF32 {
    let mut out = ImageF32::new(gray.w, gray.h);
    for y in 0..gray.h {
        let src = gray.row(y);
        let dst = out.row_mut(y);
        for x in 0..gray.w {
            dst[x] = src[x] as f32 / 255.0;
        }
    }
    out
}

fn downsample_without_filter(src: &ImageF32, dst: &mut ImageF32) {
    if src.w == 0 || src.h == 0 {
        return;
    }
    let max_sx = src.w.saturating_sub(1);
    let max_sy = src.h.saturating_sub(1);
    let mut sy = 0usize;
    for y in 0..dst.h {
        let src_row = src.row(sy.min(max_sy));
        let dst_row = dst.row_mut(y);
        let mut sx = 0usize;
        for dst_px in dst_row {
            *dst_px = src_row[sx.min(max_sx)];
            sx = sx.saturating_add(2);
        }
        sy = sy.saturating_add(2);
    }
}

fn downsample_with_filter(
    src: &ImageF32,
    dst: &mut ImageF32,
    filter: StaticSeparableFilter,
    horiz_cache: &mut Vec<f32>,
    cached_rows: &mut Vec<isize>,
) {
    if src.w == 0 || src.h == 0 || dst.w == 0 || dst.h == 0 {
        return;
    }
    let taps = filter.taps();
    assert!(
        !taps.is_empty(),
        "filter must provide at least one tap for downsampling"
    );
    let radius = taps.len() / 2;
    let taps_len = taps.len();
    let cache_width = dst.w;

    horiz_cache.resize(cache_width * taps_len, 0.0);
    cached_rows.resize(taps_len, -1);

    for y in 0..dst.h {
        let center_sy = (y * 2) as isize;
        for ky in 0..taps_len {
            let offset = ky as isize - radius as isize;
            let sy = clamp_index(center_sy + offset, src.h) as isize;
            if cached_rows[ky] != sy {
                let src_row = src.row(sy as usize);
                let cache_row = &mut horiz_cache[ky * cache_width..(ky + 1) * cache_width];
                filter_row_downsample(src_row, cache_row, taps, radius);
                cached_rows[ky] = sy;
            }
        }
        let dst_row = dst.row_mut(y);
        for x in 0..cache_width {
            let mut acc = 0.0f32;
            for ky in 0..taps_len {
                acc += taps[ky] * horiz_cache[ky * cache_width + x];
            }
            dst_row[x] = acc;
        }
    }
}

fn filter_row_downsample(row: &[f32], out: &mut [f32], taps: &[f32], radius: usize) {
    if row.is_empty() || out.is_empty() {
        return;
    }
    let max_x = row.len();
    let mut sx = 0isize;
    for dst_px in out {
        let mut acc = 0.0f32;
        for (k, &tap) in taps.iter().enumerate() {
            let offset = k as isize - radius as isize;
            let idx = clamp_index(sx + offset, max_x);
            acc += tap * row[idx];
        }
        *dst_px = acc;
        sx = sx.saturating_add(2);
    }
}

fn clamp_index(idx: isize, upper: usize) -> usize {
    if upper == 0 {
        return 0;
    }
    if idx < 0 {
        0
    } else if (idx as usize) >= upper {
        upper - 1
    } else {
        idx as usize
    }
}
