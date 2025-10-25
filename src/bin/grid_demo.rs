use grid_detector::config::grid::{self, OutputFormat};
use grid_detector::diagnostics::DetailedResult;
use grid_detector::image::io::{
    load_grayscale_image, save_grayscale_f32, write_json_file, GrayImageU8,
};
use grid_detector::pyramid::Pyramid;
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

    if config.output.format.includes_text() {
        print_text_summary(&detailed);
    }

    if config.output.format.includes_json() {
        if let Some(path) = &config.output.json_out {
            write_json_file(path, &detailed)?;
            if !config.output.format.includes_text() {
                println!("JSON report written to {}", path.display());
            } else {
                println!("\nJSON report written to {}", path.display());
            }
        } else {
            let json = serde_json::to_string_pretty(&detailed)
                .map_err(|e| format!("Failed to serialize JSON: {e}"))?;
            if config.output.format == OutputFormat::Both {
                println!("\nJSON report:\n{json}");
            } else {
                println!("{json}");
            }
        }
    }

    if let Some(dir) = &config.output.debug_dir {
        save_debug_artifacts(dir, &gray, &detailed, &config.grid_params)?;
        if config.output.format.includes_text() {
            println!("Debug artifacts written to {}", dir.display());
        } else {
            eprintln!("Debug artifacts written to {}", dir.display());
        }
    }

    Ok(())
}

fn print_text_summary(detailed: &DetailedResult) {
    let res = &detailed.result;
    println!("Detection summary");
    println!("  found: {}", res.found);
    println!("  confidence: {:.3}", res.confidence);
    println!("  latency_ms: {:.3}", res.latency_ms);
    let h = &res.hmtx;
    println!(
        "  homography:\n    [{:.4} {:.4} {:.4}]\n    [{:.4} {:.4} {:.4}]\n    [{:.4} {:.4} {:.4}]",
        h[(0, 0)],
        h[(0, 1)],
        h[(0, 2)],
        h[(1, 0)],
        h[(1, 1)],
        h[(1, 2)],
        h[(2, 0)],
        h[(2, 1)],
        h[(2, 2)]
    );
    if let Some(pose) = &res.pose {
        println!(
            "  pose.t: [{:.3}, {:.3}, {:.3}]",
            pose.t[0], pose.t[1], pose.t[2]
        );
    } else {
        println!("  pose: unavailable (insufficient confidence)");
    }

    let diag = &detailed.diagnostics;
    println!("\nPyramid (built in {:.3} ms)", diag.pyramid_build_ms);
    for lvl in &diag.pyramid_levels {
        println!(
            "  L{}: {}x{} mean={:.4}",
            lvl.level, lvl.width, lvl.height, lvl.mean_intensity
        );
    }

    println!(
        "\nTimings (ms): pyramid={:.3} lsd={:.3} filter={:.3} bundling={:.3} seg_refine={:.3} refine={:.3} total={:.3}",
        diag.pyramid_build_ms,
        diag.lsd_ms,
        diag.outlier_filter_ms,
        diag.bundling_ms,
        diag.segment_refine_ms,
        diag.refine_ms,
        diag.total_latency_ms
    );

    if let Some(filter) = &diag.segment_filter {
        println!(
            "Segment filter: kept={}/{} (fam_u={} fam_v={} skipped_degenerate={}) angle_thresh={:.1} residual_thresh={:.2}px elapsed_ms={:.3}",
            filter.kept,
            filter.total,
            filter.kept_u,
            filter.kept_v,
            filter.skipped_degenerate,
            filter.angle_threshold_deg,
            filter.residual_threshold_px,
            filter.elapsed_ms,
        );
    }

    match &diag.lsd {
        Some(lsd) => {
            println!(
                "\nLSD stage: segments={} fam_u={} fam_v={} conf={:.3} elapsed_ms={:.3}",
                lsd.segments_total,
                lsd.family_u_count,
                lsd.family_v_count,
                lsd.confidence,
                lsd.elapsed_ms
            );
            println!(
                "  dominant_angles_deg=[{:.1}, {:.1}] sample_count={}",
                lsd.dominant_angles_deg[0],
                lsd.dominant_angles_deg[1],
                lsd.segments_sample.len()
            );
        }
        None => println!("\nLSD stage: no viable hypothesis"),
    }

    match &diag.refinement {
        Some(refine) => {
            println!(
                "\nRefinement: passes={} levels={} aggregated_confidence={:.3} inlier_ratio={:.3}",
                diag.refinement_passes,
                refine.levels_used,
                refine.aggregated_confidence,
                refine.final_inlier_ratio
            );
            for lvl in &refine.levels {
                println!(
                    "  L{}: {}x{} segs={} bundles={} fam_u={} fam_v={} improvement={} conf={} inliers={}",
                    lvl.level_index,
                    lvl.width,
                    lvl.height,
                    lvl.segments,
                    lvl.bundles,
                    lvl.family_u_count,
                    lvl.family_v_count,
                    format_opt(lvl.improvement),
                    format_opt(lvl.confidence),
                    format_opt(lvl.inlier_ratio),
                );
            }
        }
        None => println!("\nRefinement: disabled or no refinement result"),
    }
}

fn format_opt(val: Option<f32>) -> String {
    val.map(|v| format!("{:.3}", v))
        .unwrap_or_else(|| "-".to_string())
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

    let pyramid = Pyramid::build_u8(gray.as_view(), grid_params.pyramid_levels);
    for (level_idx, level) in pyramid.levels.iter().enumerate() {
        let path = dir.join(format!("pyramid_L{}.png", level_idx));
        save_grayscale_f32(level, &path)?;
    }

    Ok(())
}
