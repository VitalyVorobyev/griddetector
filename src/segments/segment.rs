use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::cell::OnceCell;

/// Identifier referencing a segment recorded in the pipeline trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SegmentId(pub u32);

/// Line segment produced by the LSD-like extractor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub id: SegmentId,
    pub p0: [f32; 2],
    pub p1: [f32; 2],
    pub avg_mag: f32,
    pub strength: f32,
    #[serde(skip)]
    line: OnceCell<Vector3<f32>>,
    #[serde(skip)]
    normal: OnceCell<[f32; 2]>,
    #[serde(skip)]
    direction: OnceCell<[f32; 2]>,
    #[serde(skip)]
    length: OnceCell<f32>,
    #[serde(skip)]
    length_sq: OnceCell<f32>,
    #[serde(skip)]
    theta: OnceCell<f32>,
}

impl Segment {
    pub fn new(id: SegmentId, p0: [f32; 2], p1: [f32; 2], avg_mag: f32, strength: f32) -> Self {
        Self {
            id,
            p0,
            p1,
            avg_mag,
            strength,
            line: OnceCell::new(),
            normal: OnceCell::new(),
            direction: OnceCell::new(),
            length: OnceCell::new(),
            length_sq: OnceCell::new(),
            theta: OnceCell::new(),
        }
    }

    pub fn midpoint(&self) -> [f32; 2] {
        [
            (self.p0[0] + self.p1[0]) * 0.5,
            (self.p0[1] + self.p1[1]) * 0.5,
        ]
    }

    fn compute_line(&self) -> Vector3<f32> {
        let a = self.p1[1] - self.p0[1];
        let b = self.p0[0] - self.p1[0];
        let c = self.p1[0] * self.p0[1] - self.p0[0] * self.p1[1];
        let norm = (a * a + b * b).sqrt();
        Vector3::new(a / norm, b / norm, c / norm)
    }

    /// Line representation: ax + by + c = 0, with sqrt(a^2+b^2)=1
    pub fn line(&self) -> Vector3<f32> {
        *self.line.get_or_init(|| self.compute_line())
    }

    fn compute_normal(&self) -> [f32; 2] {
        let dir = self.direction();
        [-dir[1], dir[0]]
    }

    pub fn normal(&self) -> [f32; 2] {
        *self.normal.get_or_init(|| self.compute_normal())
    }

    fn compute_length(&self) -> f32 {
        let dx = self.p1[0] - self.p0[0];
        let dy = self.p1[1] - self.p0[1];
        (dx * dx + dy * dy).sqrt()
    }

    pub fn length(&self) -> f32 {
        *self.length.get_or_init(|| self.compute_length())
    }

    fn compute_length_sq(&self) -> f32 {
        let dx = self.p1[0] - self.p0[0];
        let dy = self.p1[1] - self.p0[1];
        dx * dx + dy * dy
    }

    pub fn length_sq(&self) -> f32 {
        *self.length_sq.get_or_init(|| self.compute_length_sq())
    }

    fn compute_theta(&self) -> f32 {
        let dir = self.direction();
        dir[1].atan2(dir[0])
    }

    pub fn theta(&self) -> f32 {
        *self.theta.get_or_init(|| self.compute_theta())
    }

    fn compute_direction(&self) -> [f32; 2] {
        let len = self.length();
        if len > 0.0 {
            [
                (self.p1[0] - self.p0[0]) / len,
                (self.p1[1] - self.p0[1]) / len,
            ]
        } else {
            [0.0, 0.0]
        }
    }

    pub fn direction(&self) -> [f32; 2] {
        *self.direction.get_or_init(|| self.compute_direction())
    }
}
