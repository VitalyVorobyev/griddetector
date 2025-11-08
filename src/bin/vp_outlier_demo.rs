use grid_detector::config::vp_outlier_demo as cfg;
use grid_detector::config::vp_outlier_demo::VpOutlierDemoConfig;
use grid_detector::image::io::{
    load_grayscale_image, save_grayscale_f32, write_json_file, GrayImageU8,
};
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

fn build_pyramid_save_coarsest_image(
    img: &GrayImageU8,
    config: &VpOutlierDemoConfig,
    levels: usize,
) -> Result<(), String> {
    let pyramid_opts = build_pyramid_options(config, levels);
    let pyramid = Pyramid::build_u8(img.as_view(), pyramid_opts);
    save_coarsest_level(config, &pyramid)
}

fn run() -> Result<(), String> {
    let config_path = env::args().nth(1).ok_or_else(usage)?;
    let config = cfg::load_config(Path::new(&config_path))?;

    fs::create_dir_all(&config.output.dir)
        .map_err(|e| format!("Failed to create {}: {e}", config.output.dir.display()))?;

    let gray = load_grayscale_image(&config.input)?;
    let levels = config.pyramid.levels;
    build_pyramid_save_coarsest_image(&gray, &config, levels)?;

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
    PyramidOptions::new(levels).with_blur_levels(config.pyramid.blur_levels)
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
    GridParams {
        pyramid_levels: levels,
        pyramid_blur_levels: config.pyramid.blur_levels,
        lsd_params: config.lsd,
        outlier_filter: config.outlier.clone(),
        bundling_params: config.bundling.clone(),
        refine_params: config.refine.clone(),
        ..GridParams::default()
    }
}
