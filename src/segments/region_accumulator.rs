pub(crate) struct RegionAccumulator {
    pub indices: Vec<usize>,
    pub sum_x: f32,
    pub sum_y: f32,
    pub sum_xx: f32,
    pub sum_yy: f32,
    pub sum_xy: f32,
    pub aligned: usize,
    pub sum_mag: f32,
}

impl RegionAccumulator {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            indices: Vec::with_capacity(capacity),
            sum_x: 0.0,
            sum_y: 0.0,
            sum_xx: 0.0,
            sum_yy: 0.0,
            sum_xy: 0.0,
            aligned: 0,
            sum_mag: 0.0,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.indices.clear();
        self.sum_x = 0.0;
        self.sum_y = 0.0;
        self.sum_xx = 0.0;
        self.sum_yy = 0.0;
        self.sum_xy = 0.0;
        self.aligned = 0;
        self.sum_mag = 0.0;
    }

    pub(crate) fn push(&mut self, idx: usize, x: usize, y: usize, mag: f32, aligned: bool) {
        self.indices.push(idx);
        let xf = x as f32;
        let yf = y as f32;
        self.sum_x += xf;
        self.sum_y += yf;
        self.sum_xx += xf * xf;
        self.sum_yy += yf * yf;
        self.sum_xy += xf * yf;
        if aligned {
            self.aligned += 1;
        }
        self.sum_mag += mag;
    }

    pub(crate) fn len(&self) -> usize {
        self.indices.len()
    }

    pub(crate) fn aligned_fraction(&self) -> f32 {
        if self.indices.is_empty() {
            0.0
        } else {
            self.aligned as f32 / self.indices.len() as f32
        }
    }

    pub(crate) fn avg_mag(&self) -> f32 {
        if self.indices.is_empty() {
            0.0
        } else {
            self.sum_mag / self.indices.len() as f32
        }
    }

    /// Mark all pixels of the current region with the provided state.
    pub(crate) fn mark_as(&self, used: &mut [u8], state: u8) {
        for &idx in &self.indices {
            used[idx] = state;
        }
    }

    /// OR a bitmask into all pixels of the current region.
    pub(crate) fn mark_mask(&self, mask: &mut [u8], bit: u8) {
        for &idx in &self.indices {
            mask[idx] |= bit;
        }
    }
}
