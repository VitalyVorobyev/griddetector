use grid_detector::config::vp_outlier_demo as cfg;
use grid_detector::config::vp_outlier_demo::VpOutlierDemoConfig;
use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::pyramid::{Pyramid, PyramidOptions};
use grid_detector::{GridDetector, GridParams};
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config_path = env::args().nth(1).ok_or_else(usage)?;
    let config = cfg::load_config(Path::new(&config_path))?;

    fs::create_dir_all(&config.output.dir)
        .map_err(|e| format!("Failed to create {}: {e}", config.output.dir.display()))?;

    let gray = load_grayscale_image(&config.input)?;
    let levels = config.pyramid.levels.max(1);
    let pyramid_opts = build_pyramid_options(&config, levels);
    let pyramid = Pyramid::build_u8(gray.as_view(), pyramid_opts);
    save_coarsest_level(&config, &pyramid)?;

    let params = grid_params_from_config(&config, levels);
    let mut detector = GridDetector::new(params);

    let report = detector.process_coarsest_with_diagnostics(gray.as_view());
    report.print_text_summary();

    write_json_file(&config.output.result_path(), &report)?;

    println!(
        "Wrote diagnostics JSON to {}",
        config.output.result_path().display()
    );

    Ok(())
}

fn usage() -> String {
    "Usage: vp_outlier_demo <config.json>".to_string()
}

fn build_pyramid_options(config: &VpOutlierDemoConfig, levels: usize) -> PyramidOptions<'static> {
    if let Some(blur_levels_cfg) = config.pyramid.blur_levels {
        let blur_levels = blur_levels_cfg.min(levels.saturating_sub(1));
        PyramidOptions::new(levels).with_blur_levels(Some(blur_levels))
    } else {
        PyramidOptions::new(levels)
    }
}

fn save_coarsest_level(config: &VpOutlierDemoConfig, pyramid: &Pyramid) -> Result<(), String> {
    let idx = pyramid
        .levels
        .len()
        .checked_sub(1)
        .ok_or_else(|| "Pyramid must contain at least one level".to_string())?;
    let coarsest = &pyramid.levels[idx];
    save_grayscale_f32(coarsest, &config.output.coarsest_path())
        .map_err(|e| format!("Failed to save coarsest level: {e}"))?;
    println!(
        "Saved coarsest level image to {} (level {})",
        config.output.coarsest_path().display(),
        idx
    );
    Ok(())
}

fn grid_params_from_config(config: &VpOutlierDemoConfig, levels: usize) -> GridParams {
    let mut params = GridParams::default();
    params.pyramid_levels = levels;
    params.pyramid_blur_levels = config.pyramid.blur_levels;
    params.lsd_vp_params = config.resolve_lsd_vp_params();
    params.outlier_filter = config.outlier.resolve();
    params.bundling_params = config.bundling.resolve();
    params.refine_params = config.refine.resolve();
    params
}
