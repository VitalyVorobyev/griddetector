use crate::types::{ImageF32, ImageU8};

#[derive(Clone, Debug)]
pub struct Pyramid {
    pub levels: Vec<ImageF32>, // grayscale float 0..1
}

impl Pyramid {
    pub fn build_u8(gray: ImageU8, levels: usize) -> Self {
        let mut out = Vec::with_capacity(levels);
        // L0: convert to f32 [0,1]
        let mut l0 = ImageF32::new(gray.w, gray.h);
        for y in 0..gray.h {
            for x in 0..gray.w {
                l0.set(x, y, gray.get(x, y) as f32 / 255.0);
            }
        }
        out.push(l0);
        for lvl in 1..levels {
            let prev = &out[lvl - 1];
            let (nw, nh) = ((prev.w + 1) / 2, (prev.h + 1) / 2);
            let mut tmp = ImageF32::new(prev.w, prev.h);
            gaussian5x5_sep(prev, &mut tmp);
            let mut down = ImageF32::new(nw, nh);
            // 2x decimation (pick every other pixel)
            for y in 0..nh {
                for x in 0..nw {
                    down.set(x, y, tmp.get(x * 2.min(prev.w - 1), y * 2.min(prev.h - 1)));
                }
            }
            out.push(down);
        }
        Self { levels: out }
    }
}

/// Simple 5-tap separable Gaussian (approx sigma≈1)
fn gaussian5x5_sep(inp: &ImageF32, out: &mut ImageF32) {
    // 1D kernel [1,4,6,4,1]/16 applied separably
    let w = inp.w;
    let h = inp.h;
    let mut tmp = ImageF32::new(w, h);
    // horizontal
    for y in 0..h {
        for x in 0..w {
            let xm1 = x.saturating_sub(1);
            let xm2 = x.saturating_sub(2);
            let xp1 = (x + 1).min(w - 1);
            let xp2 = (x + 2).min(w - 1);
            let v = (inp.get(xm2, y)
                + 4.0 * inp.get(xm1, y)
                + 6.0 * inp.get(x, y)
                + 4.0 * inp.get(xp1, y)
                + inp.get(xp2, y))
                * (1.0 / 16.0);
            tmp.set(x, y, v);
        }
    }
    // vertical
    for y in 0..h {
        let ym1 = y.saturating_sub(1);
        let ym2 = y.saturating_sub(2);
        let yp1 = (y + 1).min(h - 1);
        let yp2 = (y + 2).min(h - 1);
        for x in 0..w {
            let v = (tmp.get(x, ym2)
                + 4.0 * tmp.get(x, ym1)
                + 6.0 * tmp.get(x, y)
                + 4.0 * tmp.get(x, yp1)
                + tmp.get(x, yp2))
                * (1.0 / 16.0);
            out.set(x, y, v);
        }
    }
}
