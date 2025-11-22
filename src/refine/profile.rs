use super::roi::SegmentRoi;
use serde::Serialize;
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Debug, Default, Serialize)]
pub struct LevelProfile {
    pub level_index: usize,
    pub roi_count: u64,
    pub roi_area_px: f64,
    pub bilinear_samples: u64,
}

#[derive(Default)]
struct ProfileState {
    levels: Vec<LevelProfile>,
}

impl ProfileState {
    fn level_mut(&mut self, idx: usize) -> &mut LevelProfile {
        if self.levels.len() <= idx {
            self.levels.resize_with(idx + 1, LevelProfile::default);
        }
        let entry = &mut self.levels[idx];
        entry.level_index = idx;
        entry
    }
}

static STATE: OnceLock<Mutex<ProfileState>> = OnceLock::new();

fn state() -> &'static Mutex<ProfileState> {
    STATE.get_or_init(|| Mutex::new(ProfileState::default()))
}

pub fn record_roi(level_index: usize, roi: &SegmentRoi) {
    let width = (roi.x1 - roi.x0).max(0.0);
    let height = (roi.y1 - roi.y0).max(0.0);
    let area = (width * height) as f64;
    let mut guard = state().lock().expect("profile mutex poisoned");
    let entry = guard.level_mut(level_index);
    entry.roi_count += 1;
    entry.roi_area_px += area;
}

pub fn record_sample(level_index: usize) {
    let mut guard = state().lock().expect("profile mutex poisoned");
    guard.level_mut(level_index).bilinear_samples += 1;
}

pub fn take_profile() -> Vec<LevelProfile> {
    let mut guard = state().lock().expect("profile mutex poisoned");
    let snapshot = guard.levels.clone();
    guard.levels.clear();
    snapshot
}
