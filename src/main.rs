use grid_detector::diagnostics::DetailedResult;
use grid_detector::types::ImageU8;
use grid_detector::{GridDetector, GridParams};
use nalgebra::Matrix3;
use std::env;
use std::fs;
use std::path::PathBuf;

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
    let config = parse_args(&program)?;

    let img = image::open(&config.input_path)
        .map_err(|e| format!("Failed to open {}: {e}", config.input_path.display()))?
        .to_luma8();
    let width = img.width() as usize;
    let height = img.height() as usize;
    let stride = width;
    let gray_buf = img.into_raw();
    let image = ImageU8 {
        w: width,
        h: height,
        stride,
        data: &gray_buf,
    };

    let mut params = GridParams::default();
    if let Some(spacing) = config.spacing_mm {
        params.spacing_mm = spacing;
    }
    if let Some(k) = config.intrinsics {
        params.kmtx = k;
    }
    let mut detector = GridDetector::new(params);
    let detailed = detector.process_with_diagnostics(image);

    if config.format.includes_text() {
        print_text_summary(&detailed);
    }

    if config.format.includes_json() {
        let json = serde_json::to_string_pretty(&detailed)
            .map_err(|e| format!("Failed to serialize JSON: {e}"))?;
        if let Some(path) = config.json_out {
            fs::write(&path, json)
                .map_err(|e| format!("Failed to write JSON report to {}: {e}", path.display()))?;
            if !config.format.includes_text() {
                println!("JSON report written to {}", path.display());
            } else {
                println!("\nJSON report written to {}", path.display());
            }
        } else {
            if config.format == OutputFormat::Both {
                println!("\nJSON report:\n{json}");
            } else {
                println!("{json}");
            }
        }
    }

    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Text,
    Json,
    Both,
}

impl OutputFormat {
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            "both" => Ok(Self::Both),
            other => Err(format!("Unknown format '{other}'. Use text|json|both.")),
        }
    }

    fn includes_text(&self) -> bool {
        matches!(self, Self::Text | Self::Both)
    }

    fn includes_json(&self) -> bool {
        matches!(self, Self::Json | Self::Both)
    }
}

struct CliConfig {
    input_path: PathBuf,
    format: OutputFormat,
    json_out: Option<PathBuf>,
    spacing_mm: Option<f32>,
    intrinsics: Option<Matrix3<f32>>,
}

fn parse_args(program: &str) -> Result<CliConfig, String> {
    let mut args = env::args().skip(1).peekable();
    let mut input_path: Option<PathBuf> = None;
    let mut format = OutputFormat::Json;
    let mut json_out: Option<PathBuf> = None;
    let mut spacing_mm: Option<f32> = None;
    let mut intrinsics: Option<Matrix3<f32>> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{}", usage(program));
                std::process::exit(0);
            }
            "--format" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("--format expects a value\n{}", usage(program)))?;
                format = OutputFormat::from_str(&value)?;
            }
            "--json-out" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("--json-out expects a path\n{}", usage(program)))?;
                json_out = Some(PathBuf::from(value));
            }
            "--spacing-mm" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("--spacing-mm expects a value\n{}", usage(program)))?;
                let parsed: f32 = value
                    .parse()
                    .map_err(|_| format!("Invalid spacing '{value}'"))?;
                spacing_mm = Some(parsed);
            }
            "--intrinsics" => {
                let value = args.next().ok_or_else(|| {
                    format!("--intrinsics expects fx,fy,cx,cy\n{}", usage(program))
                })?;
                intrinsics = Some(parse_intrinsics(&value)?);
            }
            _ if arg.starts_with('-') => {
                return Err(format!("Unknown option '{arg}'\n{}", usage(program)));
            }
            _ => {
                if input_path.is_some() {
                    return Err(format!(
                        "Unexpected positional argument '{arg}'\n{}",
                        usage(program)
                    ));
                }
                input_path = Some(PathBuf::from(arg));
            }
        }
    }

    let input_path = input_path.ok_or_else(|| usage(program))?;
    Ok(CliConfig {
        input_path,
        format,
        json_out,
        spacing_mm,
        intrinsics,
    })
}

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} <image.png> [--format text|json|both] [--json-out report.json] \\\n         [--spacing-mm mm] [--intrinsics fx,fy,cx,cy]\n\n\
Runs the grid detector on a grayscale PNG image and emits diagnostics.\n\
Examples:\n  {program} data/sample.png --format both --json-out sample_report.json\n  {program} board.png --format text --spacing-mm 5.0\n"
    )
}

fn parse_intrinsics(value: &str) -> Result<Matrix3<f32>, String> {
    let parts: Vec<&str> = value.split(',').collect();
    if parts.len() != 4 {
        return Err(format!("Expected fx,fy,cx,cy but got '{value}'"));
    }
    let fx: f32 = parts[0]
        .trim()
        .parse()
        .map_err(|_| format!("Invalid fx '{}'", parts[0]))?;
    let fy: f32 = parts[1]
        .trim()
        .parse()
        .map_err(|_| format!("Invalid fy '{}'", parts[1]))?;
    let cx: f32 = parts[2]
        .trim()
        .parse()
        .map_err(|_| format!("Invalid cx '{}'", parts[2]))?;
    let cy: f32 = parts[3]
        .trim()
        .parse()
        .map_err(|_| format!("Invalid cy '{}'", parts[3]))?;
    Ok(Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0))
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
                "\nRefinement: levels={} aggregated_confidence={:.3} inlier_ratio={:.3}",
                refine.levels_used, refine.aggregated_confidence, refine.final_inlier_ratio
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
