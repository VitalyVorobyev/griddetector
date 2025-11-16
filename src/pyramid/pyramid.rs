use super::filters::apply as apply_filter;
use super::options::PyramidOptions;
use crate::image::{ImageF32, ImageU8, ImageView, ImageViewMut};


#[derive(Clone, Debug)]
pub struct Pyramid {
    pub levels: Vec<ImageF32>,
}

impl Pyramid {
    /// Build a pyramid from an 8-bit grayscale input using the provided options.
    pub fn build_u8(gray: ImageU8<'_>, options: PyramidOptions) -> Self {
        assert!(options.levels >= 1, "pyramid requires at least one level");
        let mut levels = Vec::with_capacity(options.levels);
        levels.push(convert_l0(gray));

        let blur_limit = options.blur_levels;
        for lvl in 1..options.levels {
            let prev = levels.last().expect("previous level available");
            let use_blur = lvl <= blur_limit;
            let filtered = if use_blur {
                Some(apply_filter(options.filter, prev))
            } else {
                None
            };
            let src_img = filtered.as_ref().unwrap_or(prev);

            let (nw, nh) = (prev.w.div_ceil(2), prev.h.div_ceil(2));
            let mut down = ImageF32::new(nw, nh);
            for y in 0..nh {
                let dst_row = down.row_mut(y);
                let sy = (y * 2).min(src_img.h - 1);
                let src_row = src_img.row(sy);
                for (x, dst_px) in dst_row.iter_mut().enumerate() {
                    let sx = (x * 2).min(src_img.w - 1);
                    *dst_px = src_row[sx];
                }
            }
            levels.push(down);
        }

        Self { levels }
    }

    pub fn scale_for_level(&self, level: &ImageF32) -> f32 {
        if let Some(index) = self.levels.iter().position(|l| std::ptr::eq(l, level)) {
            1.0 / (2u32.pow(index as u32) as f32)
        } else {
            1.0
        }
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
