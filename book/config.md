# Configuration

Detector is driven by a JSON config. Example:

```json
{
  "grid": {
    "pyramid_levels": 4,
    "pyramid_blur_levels": null,
    "confidence_thresh": 0.35,
    "enable_refine": true,
    "refinement_schedule": { "passes": 1, "improvement_thresh": 0.0005 },
    "lsd_vp": { "mag_thresh": 0.05, "angle_tol_deg": 22.5, "min_len": 4.0 },
    "outlier_filter": { "angle_margin_deg": 8.0 },
    "bundling": {
      "orientation_tol_deg": 22.5,
      "merge_dist_px": 1.5,
      "min_weight": 3.0,
      "scale_mode": "full_res"
    }
  }
}
```

Tips:

* Tune lsd_vp.mag_thresh and min_len for your scale.
* Use bundling.scale_mode="full_res" for level-invariant thresholds.
* Increase refinement_schedule.passes to 2 on weak hypotheses.

