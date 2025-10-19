use grid_detector::config::edge;
use grid_detector::edges::{detect_edges_sobel_nms, EdgeElement};
use grid_detector::image::io::{load_grayscale_image, save_grayscale_f32, write_json_file};
use grid_detector::pyramid::Pyramid;
use serde::Serialize;
use std::env;
use std::path::Path;

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config_path = env::args().nth(1).ok_or_else(usage)?;
    let config = edge::load_config(Path::new(&config_path))?;

    let gray = load_grayscale_image(&config.input)?;
    let levels = config.pyramid.levels.max(1);
    let pyramid = if let Some(blur_levels_cfg) = config.pyramid.blur_levels {
        let blur_levels = blur_levels_cfg.min(levels.saturating_sub(1));
        Pyramid::build_u8_with_blur_levels(gray.as_view(), levels, blur_levels)
    } else {
        Pyramid::build_u8(gray.as_view(), levels)
    };
    let coarsest_index = pyramid
        .levels
        .len()
        .checked_sub(1)
        .ok_or("Pyramid has no levels")?;
    let coarsest = &pyramid.levels[coarsest_index];

    let edges = detect_edges_sobel_nms(coarsest, config.edge.magnitude_threshold);
    let summary = EdgeDetectionSummary {
        width: coarsest.w,
        height: coarsest.h,
        pyramid_level_index: coarsest_index,
        magnitude_threshold: config.edge.magnitude_threshold,
        edge_count: edges.len(),
        edges,
    };

    save_grayscale_f32(coarsest, &config.output.coarsest_image)?;
    write_json_file(&config.output.edges_json, &summary)?;

    println!(
        "Saved coarsest level image to {} (level {})",
        config.output.coarsest_image.display(),
        coarsest_index
    );
    println!(
        "Saved {} edge elements to {}",
        summary.edge_count,
        config.output.edges_json.display()
    );

    Ok(())
}

fn usage() -> String {
    "Usage: coarse_edges <config.json>".to_string()
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EdgeDetectionSummary {
    width: usize,
    height: usize,
    pyramid_level_index: usize,
    magnitude_threshold: f32,
    edge_count: usize,
    edges: Vec<EdgeElement>,
}
