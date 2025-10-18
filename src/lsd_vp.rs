use nalgebra::{Matrix3, Vector3};
use crate::types::ImageF32;
use crate::segments::{lsd_extract_segments, Segment};

/// Coarse hypothesis returned by the LSD→VP engine
#[derive(Clone, Debug)]
pub struct Hypothesis {
    pub hmtx0: Matrix3<f32>,
    pub confidence: f32
}

/// Lightweight engine that finds two dominant line families from LSD segments,
/// estimates their vanishing points, and returns a coarse projective basis H0.
#[derive(Clone, Debug)]
pub struct Engine {
    /// Gradient magnitude threshold at the pyramid level (0..1)
    pub mag_thresh: f32,
    /// Angular tolerance (degrees) for clustering around histogram peaks
    pub angle_tol_deg: f32,
    /// Minimum accepted segment length in pixels (at that level)
    pub min_len: f32,
}

impl Default for Engine {
    fn default() -> Self {
        Self { mag_thresh: 0.05, angle_tol_deg: 22.5, min_len: 4.0 }
    }
}

impl Engine {
    /// Run the engine on a single pyramid level image. Returns a coarse H0 if successful.
    pub fn infer(&mut self, l: &ImageF32) -> Option<Hypothesis> {
        // 1) LSD-like segment extraction
        let segs = lsd_extract_segments(
            l,
            self.mag_thresh,
            self.angle_tol_deg.to_radians(),
            self.min_len,
        );
        if segs.len() < 12 {
            // TODO: log debug "too few segments"
            return None;
        }

        // 2) Orientation histogram (0..pi) weighted by segment length
        let bins = 36usize;
        let bin_w = std::f32::consts::PI / (bins as f32);
        let mut hist = vec![0.0f32; bins];
        let mut angles = Vec::with_capacity(segs.len());
        for s in &segs {
            // tangent direction
            let th = s.dir[1].atan2(s.dir[0]);
            let mut a = th;
            if a < 0.0 { a += std::f32::consts::PI; }
            if a >= std::f32::consts::PI { a -= std::f32::consts::PI; }
            angles.push(a);
            let mut b = (a / bin_w) as usize;
            if b >= bins { b = bins - 1; }
            hist[b] += s.len.max(1.0);
        }

        // 3) Take two dominant peaks separated by at least ~ angle_tol*2
        let mut first = 0usize; let mut second = 0usize;
        for i in 0..bins { if hist[i] > hist[first] { first = i; } }
        let sep_bins = ((self.angle_tol_deg * 2.0) * (bins as f32) / 180.0).ceil() as isize;
        let mut hist2 = hist.clone();
        for di in -sep_bins..=sep_bins {
            let j = ((first as isize + di).rem_euclid(bins as isize)) as usize; hist2[j] = 0.0;
        }
        for i in 0..bins { if hist2[i] > hist2[second] { second = i; } }
        if first == second || hist[first] <= 0.0 || hist[second] <= 0.0 { return None; }
        let a1 = (first as f32 + 0.5) * bin_w; let a2 = (second as f32 + 0.5) * bin_w;

        // 4) Soft-assign segments to the two families
        let tol = self.angle_tol_deg.to_radians();
        let mut fam1: Vec<&Segment> = Vec::new();
        let mut fam2: Vec<&Segment> = Vec::new();
        for (i,s) in segs.iter().enumerate() {
            let a = angles[i];
            let d1 = angle_sep(a, a1); let d2 = angle_sep(a, a2);
            if d1 < d2 && d1 <= tol { fam1.push(s); } else if d2 < d1 && d2 <= tol { fam2.push(s); }
        }
        if fam1.len() < 6 || fam2.len() < 6 { return None; }

        // 5) Estimate VPs from sets of normal-form lines (ax + by + c = 0)
        let vpu = estimate_vp(&fam1)?; // u-direction VP
        let vpv = estimate_vp(&fam2)?; // v-direction VP

        // 6) Compose a coarse projective basis H0. This isn't the final homography
        // yet; refinement will enforce spacing/metric later. We use the image center
        // as the third column to anchor translation.
        let cx = (l.w as f32) * 0.5; let cy = (l.h as f32) * 0.5;
        let x0 = Vector3::new(cx, cy, 1.0);
        let hmtx0 = Matrix3::from_columns(&[vpu, vpv, x0]);

        // 7) Confidence heuristic from support and angular separation
        let sep = angle_sep(a1, a2);
        let conf = (
            (fam1.len().min(50) as f32 / 50.0) *
            (fam2.len().min(50) as f32 / 50.0) *
            (sep / (0.5*std::f32::consts::PI)).min(1.0)
        ).clamp(0.0, 1.0);
        Some(Hypothesis { hmtx0, confidence: conf })
    }
}

#[inline]
fn angle_sep(a: f32, b: f32) -> f32 {
    let mut d = (a - b).abs();
    if d > std::f32::consts::PI * 0.5 {
        d = std::f32::consts::PI - d;
    }
    d
}

fn estimate_vp(segs: &Vec<&Segment>) -> Option<Vector3<f32>> {
    // Solve for v=(x,y) minimizing Σ w (a x + b y + c)^2 where each line is ax+by+c=0
    let mut a11=0.0f32; let mut a12=0.0f32; let mut a22=0.0f32; let mut bx=0.0f32; let mut by=0.0f32;
    for s in segs.iter() {
        let [a,b,c] = s.line; let w = s.len.max(1.0);
        a11 += w * a*a; a12 += w * a*b; a22 += w * b*b; bx  += -w * c * a; by  += -w * c * b;
    }
    let det = a11*a22 - a12*a12; if det.abs() < 1e-6 { return None; }
    let inv11 =  a22 / det; let inv12 = -a12 / det; let inv22 =  a11 / det;
    let x = inv11*bx + inv12*by; let y = inv12*bx + inv22*by;
    Some(Vector3::new(x, y, 1.0))
}
