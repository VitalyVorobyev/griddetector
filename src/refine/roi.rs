
/// Axis-aligned region of interest around a segment in full-resolution coordinates.
#[derive(Clone, Copy, Debug, Default)]
pub struct SegmentRoi {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl SegmentRoi {
    #[inline]
    pub fn contains(&self, p: &[f32; 2]) -> bool {
        p[0] >= self.x0 && p[0] <= self.x1 && p[1] >= self.y0 && p[1] <= self.y1
    }

    #[inline]
    pub fn clamp_inside(&self, p: [f32; 2]) -> [f32; 2] {
        [p[0].clamp(self.x0, self.x1), p[1].clamp(self.y0, self.y1)]
    }
}

/// Integer bounds (inclusive) around a segment ROI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntBounds {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl IntBounds {
    #[inline]
    pub fn width(&self) -> usize {
        self.x1.saturating_sub(self.x0) + 1
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.y1.saturating_sub(self.y0) + 1
    }
}

/// Convert a floating-point ROI into integer bounds with a bilinear guard.
pub fn roi_to_int_bounds(roi: &SegmentRoi, width: usize, height: usize) -> Option<IntBounds> {
    convert_bounds(roi.x0, roi.x1, roi.y0, roi.y1, width, height)
}

fn convert_bounds(
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
    width: usize,
    height: usize,
) -> Option<IntBounds> {
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return None;
    }
    let mut x0 = min_x.floor() as isize - 1;
    let mut y0 = min_y.floor() as isize - 1;
    let mut x1 = max_x.ceil() as isize + 1;
    let mut y1 = max_y.ceil() as isize + 1;
    let max_x = width.saturating_sub(1) as isize;
    let max_y = height.saturating_sub(1) as isize;
    x0 = x0.clamp(0, max_x);
    y0 = y0.clamp(0, max_y);
    x1 = x1.clamp(0, max_x);
    y1 = y1.clamp(0, max_y);
    if x1 - x0 < 1 || y1 - y0 < 1 {
        return None;
    }
    Some(IntBounds {
        x0: x0 as usize,
        y0: y0 as usize,
        x1: x1 as usize,
        y1: y1 as usize,
    })
}

pub fn segment_roi_from_points(
    p0: [f32; 2],
    p1: [f32; 2],
    pad: f32,
    width: usize,
    height: usize,
) -> Option<SegmentRoi> {
    let (mut min_x, mut max_x) = (p0[0].min(p1[0]), p0[0].max(p1[0]));
    let (mut min_y, mut max_y) = (p0[1].min(p1[1]), p0[1].max(p1[1]));
    min_x -= pad;
    max_x += pad;
    min_y -= pad;
    max_y += pad;
    let w = width as f32;
    let h = height as f32;
    min_x = min_x.clamp(0.0, (w - 1.0).max(0.0));
    max_x = max_x.clamp(0.0, (w - 1.0).max(0.0));
    min_y = min_y.clamp(0.0, (h - 1.0).max(0.0));
    max_y = max_y.clamp(0.0, (h - 1.0).max(0.0));
    if min_x >= max_x || min_y >= max_y {
        None
    } else {
        Some(SegmentRoi {
            x0: min_x,
            y0: min_y,
            x1: max_x,
            y1: max_y,
        })
    }
}
