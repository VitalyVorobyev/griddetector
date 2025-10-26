#!/usr/bin/env python3
"""
Visualize grid detector diagnostics produced by `cargo run -- <image> --format both --json-out report.json`.

The figure overlays LSD segments on the input image and plots per-level refinement metrics.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize grid detector diagnostics.")
    parser.add_argument("report", type=Path, help="Path to JSON report from grid_demo")
    parser.add_argument("image", type=Path, help="Path to the grayscale PNG used as input")
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path to save the figure instead of displaying it",
    )
    args = parser.parse_args()

    data = json.loads(args.report.read_text())
    img = plt.imread(args.image)
    if img.ndim == 3:
        # Average RGB to grayscale if needed.
        img = img[..., :3].mean(axis=2)

    detailed = data
    diagnostics = detailed.get("diagnostics") or {}
    lsd = diagnostics.get("lsd") or {}
    refinement = diagnostics.get("refinement") or {}
    result = detailed.get("result") or {}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_img = axes[0, 0]
    ax_metrics = axes[0, 1]
    ax_timings = axes[1, 0]
    ax_info = axes[1, 1]

    ax_img.imshow(img, cmap="gray", origin="upper")
    ax_img.set_title("LSD segments overlay")
    ax_img.axis("off")

    segments = lsd.get("segments_sample", [])
    if segments:
        strengths = [seg.get("strength", 0.0) for seg in segments]
        max_strength = max(strengths) if strengths else 1.0
        max_strength = max(max_strength, 1e-6)
        for seg in segments:
            p0 = seg.get("p0", [0.0, 0.0])
            p1 = seg.get("p1", [0.0, 0.0])
            family = seg.get("family")
            color = {"u": "tab:cyan", "v": "tab:orange"}.get(family, "tab:gray")
            strength = seg.get("strength", 0.0)
            alpha = 0.2 + 0.8 * np.clip(strength / max_strength, 0.0, 1.0)
            ax_img.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, alpha=float(alpha), linewidth=1.2)
        ax_img.set_xlim(0, img.shape[1])
        ax_img.set_ylim(img.shape[0], 0)
    else:
        ax_img.text(
            0.5,
            0.5,
            "No segment sample available",
            ha="center",
            va="center",
            color="red",
            fontsize=12,
            transform=ax_img.transAxes,
        )

    levels = refinement.get("levels", [])
    if levels:
        idx = [level.get("level_index", i) for i, level in enumerate(levels)]
        improvements = [
            level.get("improvement", 0.0) if level.get("improvement") is not None else 0.0
            for level in levels
        ]
        confidences = [
            level.get("confidence") if level.get("confidence") is not None else np.nan
            for level in levels
        ]
        inliers = [
            level.get("inlier_ratio") if level.get("inlier_ratio") is not None else np.nan
            for level in levels
        ]

        ax_metrics.plot(idx, improvements, marker="o", label="relative update")
        ax_metrics.plot(idx, confidences, marker="s", label="confidence")
        ax_metrics.plot(idx, inliers, marker="^", label="inlier ratio")
        ax_metrics.set_xlabel("Pyramid level (coarse → fine index)")
        ax_metrics.set_ylabel("Metric value")
        ax_metrics.set_title("Refinement metrics per level")
        ax_metrics.grid(True, linestyle="--", alpha=0.3)
        ax_metrics.legend(loc="best")
    else:
        ax_metrics.text(
            0.5,
            0.5,
            "No refinement metrics",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax_metrics.axis("off")

    # Timings bar chart.
    timing_entries = [
        ("pyramid", diagnostics.get("pyramid_build_ms")),
        ("lsd", diagnostics.get("lsd_ms")),
        ("filter", diagnostics.get("outlier_filter_ms")),
        ("bundling", diagnostics.get("bundling_ms")),
        ("seg_refine", diagnostics.get("segment_refine_ms")),
        ("refine", diagnostics.get("refine_ms")),
    ]
    timing_entries = [(name, value) for name, value in timing_entries if value is not None]
    if timing_entries:
        labels, values = zip(*timing_entries)
        ax_timings.bar(labels, values, color="tab:blue", alpha=0.7)
        ax_timings.set_ylabel("milliseconds")
        total_latency = diagnostics.get("total_latency_ms", 0.0)
        ax_timings.set_title(f"Stage timings (total {total_latency:.1f} ms)")
        ax_timings.tick_params(axis="x", rotation=30)
        for idx_bar, value in enumerate(values):
            ax_timings.text(
                idx_bar,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        ax_timings.text(
            0.5,
            0.5,
            "No timing diagnostics",
            ha="center",
            va="center",
            fontsize=12,
        )
    ax_timings.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Textual summary and counts.
    ax_info.axis("off")
    refine_conf = refinement.get("aggregated_confidence")
    summary_lines = [
        f"found: {result.get('found')}",
        f"confidence: {result.get('confidence', 0.0):.3f}",
        (f"refine_conf: {refine_conf:.3f}" if refine_conf is not None else "refine_conf: n/a"),
        f"latency_ms: {result.get('latency_ms', 0.0):.3f}",
        f"refinement_levels_used: {refinement.get('levels_used', 0)}",
        f"refinement_passes: {diagnostics.get('refinement_passes', 0)}",
        f"segments_total: {lsd.get('segments_total', 0)}",
        "lsd families u/v: {}/{}".format(
            lsd.get("family_u_count", 0), lsd.get("family_v_count", 0)
        ),
    ]

    filter_diag = diagnostics.get("segment_filter") or {}
    if filter_diag:
        summary_lines.append(
            "filter kept/rejected: {}/{}".format(
                filter_diag.get("kept", 0), filter_diag.get("total", 0)
            )
        )
        summary_lines.append(
            "filter families u/v: {}/{}".format(
                filter_diag.get("kept_u", 0), filter_diag.get("kept_v", 0)
            )
        )
        summary_lines.append(
            "filter thresholds: angle ≤ {:.1f}°, residual ≤ {:.2f}px".format(
                filter_diag.get("angle_threshold_deg", 0.0),
                filter_diag.get("residual_threshold_px", 0.0),
            )
        )

    angles = lsd.get("dominant_angles_deg")
    if angles:
        summary_lines.append(
            "dominant_angles_deg: [{:.1f}, {:.1f}]".format(angles[0], angles[1])
        )

    bundling_diag = diagnostics.get("bundling") or []
    if bundling_diag:
        counts = [
            "L{}:{}".format(entry.get("level_index", i), len(entry.get("bundles", [])))
            for i, entry in enumerate(bundling_diag)
        ]
        summary_lines.append("bundles per level: " + ", ".join(counts))

    pyramid_levels = diagnostics.get("pyramid_levels") or []
    if pyramid_levels:
        dims = [
            "L{}={}x{}".format(
                level.get("level", i), level.get("width", 0), level.get("height", 0)
            )
            for i, level in enumerate(pyramid_levels)
        ]
        summary_lines.append("pyramid levels: " + ", ".join(dims))

    ax_info.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )

    fig.suptitle("Grid Detector Diagnostics", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved visualization to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
