use grid_detector::config::grid::{self};
use grid_detector::diagnostics::DetailedResult;
use grid_detector::image::io::{
    load_grayscale_image, save_grayscale_f32, write_json_file, GrayImageU8,
};
use grid_detector::pyramid::{Pyramid, PyramidOptions};
use grid_detector::GridDetector;
use std::env;
use std::path::Path;

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let program = env::args()
        .next()
        .unwrap_or_else(|| "grid_demo".to_string());
    let config = grid::parse_cli(&program)?;

    let gray = load_grayscale_image(&config.input_path)?;
    let image = gray.as_view();

    let mut detector = GridDetector::new(config.grid_params.clone());
    let detailed = detector.process_with_diagnostics(image);
    detailed.print_text_summary();

    if let Some(path) = &config.output.json_out {
        write_json_file(path, &detailed)?;
        println!("JSON report written to {}", path.display());
    } else {
        eprintln!("No JSON output path specified, skipping JSON report.");
    }

    if let Some(dir) = &config.output.debug_dir {
        save_debug_artifacts(dir, &gray, &detailed, &config.grid_params)?;
        println!("Debug artifacts written to {}", dir.display());
    }

    Ok(())
}

fn save_debug_artifacts(
    dir: &Path,
    gray: &GrayImageU8,
    detailed: &DetailedResult,
    grid_params: &grid_detector::GridParams,
) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .map_err(|e| format!("Failed to create debug dir {}: {e}", dir.display()))?;

    write_json_file(&dir.join("detailed_result.json"), detailed)?;

    if let Some(lsd) = &detailed.diagnostics.lsd {
        write_json_file(&dir.join("lsd_diagnostics.json"), lsd)?;
    }
    if let Some(refine) = &detailed.diagnostics.refinement {
        write_json_file(&dir.join("refinement_diagnostics.json"), refine)?;
    }
    if let Some(bundles) = &detailed.diagnostics.bundling {
        write_json_file(&dir.join("bundles.json"), bundles)?;
    }

    // Rebuild the pyramid for debugging using the same blur schedule as the detector
    let pyramid_opts = match grid_params.pyramid_blur_levels {
        Some(blur_levels) => {
            PyramidOptions::new(grid_params.pyramid_levels).with_blur_levels(Some(blur_levels))
        }
        None => PyramidOptions::new(grid_params.pyramid_levels),
    };
    let pyramid = Pyramid::build_u8(gray.as_view(), pyramid_opts);
    for (level_idx, level) in pyramid.levels.iter().enumerate() {
        let path = dir.join(format!("pyramid_L{}.png", level_idx));
        save_grayscale_f32(level, &path)?;
    }

    Ok(())
}
