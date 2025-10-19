use crate::types::ImageF32;

type Kernel3 = [[f32; 3]; 3];

const SOBEL_KERNEL_X: Kernel3 = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
const SOBEL_KERNEL_Y: Kernel3 = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

const SCHARR_KERNEL_X: Kernel3 = [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]];
const SCHARR_KERNEL_Y: Kernel3 = [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]];

#[derive(Clone, Debug)]
pub struct Grad {
    pub gx: ImageF32,
    pub gy: ImageF32,
    pub mag: ImageF32,
    pub ori_q8: Vec<u8>, // per-pixel quantized orientation in 8 bins
}

#[inline]
fn quantize_orientation(angle: f32) -> u8 {
    let wrapped = (angle + std::f32::consts::PI).rem_euclid(2.0 * std::f32::consts::PI);
    ((wrapped * (4.0 / std::f32::consts::PI)).floor() as i32 & 7) as u8
}

fn gradients_with_kernels(l: &ImageF32, kernel_x: &Kernel3, kernel_y: &Kernel3) -> Grad {
    let w = l.w;
    let h = l.h;
    let mut gx = ImageF32::new(w, h);
    let mut gy = ImageF32::new(w, h);
    let mut mag = ImageF32::new(w, h);
    let mut ori_q8 = vec![0u8; w * h];

    if w == 0 || h == 0 {
        return Grad {
            gx,
            gy,
            mag,
            ori_q8,
        };
    }

    for y in 0..h {
        let y_idx = [y.saturating_sub(1), y, (y + 1).min(h - 1)];
        for x in 0..w {
            let x_idx = [x.saturating_sub(1), x, (x + 1).min(w - 1)];

            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            for (ky, &yy) in y_idx.iter().enumerate() {
                let kernel_row_x = &kernel_x[ky];
                let kernel_row_y = &kernel_y[ky];
                for (xx, (&kx_weight, &ky_weight)) in x_idx
                    .iter()
                    .zip(kernel_row_x.iter().zip(kernel_row_y.iter()))
                {
                    let sample = l.get(*xx, yy);
                    sum_x += sample * kx_weight;
                    sum_y += sample * ky_weight;
                }
            }

            gx.set(x, y, sum_x);
            gy.set(x, y, sum_y);

            let magnitude = (sum_x * sum_x + sum_y * sum_y).sqrt();
            mag.set(x, y, magnitude);
            let idx = y * w + x;
            ori_q8[idx] = quantize_orientation(sum_y.atan2(sum_x));
        }
    }

    Grad {
        gx,
        gy,
        mag,
        ori_q8,
    }
}

pub fn sobel_gradients(l: &ImageF32) -> Grad {
    gradients_with_kernels(l, &SOBEL_KERNEL_X, &SOBEL_KERNEL_Y)
}

pub fn scharr_gradients(l: &ImageF32) -> Grad {
    gradients_with_kernels(l, &SCHARR_KERNEL_X, &SCHARR_KERNEL_Y)
}
