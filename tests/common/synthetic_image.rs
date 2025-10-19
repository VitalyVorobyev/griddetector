/// Generates a simple high-contrast checkerboard image.
pub fn checkerboard_u8(width: usize, height: usize, cell: usize) -> Vec<u8> {
    assert!(width > 0 && height > 0, "image dimensions must be positive");
    assert!(cell > 0, "cell size must be positive");

    let mut img = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let cx = (x / cell) as i32;
            let cy = (y / cell) as i32;
            let sum = cx + cy;
            let val = if sum & 1 == 0 { 32u8 } else { 220u8 };
            img[y * width + x] = val;
        }
    }
    img
}
