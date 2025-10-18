use crate::types::ImageF32;

#[derive(Clone, Debug)]
pub struct Grad {
    pub gx: ImageF32,
    pub gy: ImageF32,
    pub mag: ImageF32,
    pub ori_q8: Vec<u8>, // per-pixel quantized orientation in 8 bins
}

pub fn sobel_gradients(l: &ImageF32) -> Grad {
    let w = l.w;
    let h = l.h;
    let mut gx = ImageF32::new(w, h);
    let mut gy = ImageF32::new(w, h);
    // Sobel 3x3
    for y in 0..h {
        let ym1 = y.saturating_sub(1);
        let yp1 = (y + 1).min(h - 1);
        for x in 0..w {
            let xm1 = x.saturating_sub(1);
            let xp1 = (x + 1).min(w - 1);
            let tl = l.get(xm1, ym1);
            let tc = l.get(x, ym1);
            let tr = l.get(xp1, ym1);
            let ml = l.get(xm1, y);
            let mc = l.get(x, y);
            let mr = l.get(xp1, y);
            let bl = l.get(xm1, yp1);
            let bc = l.get(x, yp1);
            let br = l.get(xp1, yp1);
            let gxv = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br; // [-1 0 1; -2 0 2; -1 0 1]
            let gyv = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br; // [-1 -2 -1; 0 0 0; 1 2 1]
            gx.set(x, y, gxv);
            gy.set(x, y, gyv);
        }
    }
    let mut mag = ImageF32::new(w, h);
    let mut ori_q8 = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let gxx = gx.get(x, y);
            let gyy = gy.get(x, y);
            let m = (gxx * gxx + gyy * gyy).sqrt();
            mag.set(x, y, m);
            let ang = gyy.atan2(gxx); // [-pi,pi]
                                      // quantize to 8 bins (0..7) ignoring sign symmetry for now
            let mut bin =
                (((ang + std::f32::consts::PI) * (4.0 / std::f32::consts::PI)) as i32) & 7; // *8/(2pi)
            if bin < 0 {
                bin += 8;
            }
            ori_q8[i] = bin as u8;
        }
    }
    Grad {
        gx,
        gy,
        mag,
        ori_q8,
    }
}
