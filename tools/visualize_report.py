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
    diagnostics = detailed.get("diagnostics", {})
    lsd = diagnostics.get("lsd", {})
    refinement = diagnostics.get("refinement", {})
    result = detailed.get("result", {})

    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(14, 6))

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

        ax_plot.plot(idx, improvements, marker="o", label="relative update")
        ax_plot.plot(idx, confidences, marker="s", label="confidence")
        ax_plot.plot(idx, inliers, marker="^", label="inlier ratio")
        ax_plot.set_xlabel("Pyramid level (coarse → fine index)")
        ax_plot.set_ylabel("Metric value")
        ax_plot.set_title("Refinement metrics per level")
        ax_plot.grid(True, linestyle="--", alpha=0.3)
        ax_plot.legend(loc="best")
    else:
        ax_plot.text(
            0.5,
            0.5,
            "No refinement metrics",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax_plot.axis("off")

    summary_lines = [
        f"found: {result.get('found')}",
        f"confidence: {result.get('confidence', 0.0):.3f}",
        f"latency_ms: {result.get('latency_ms', 0.0):.3f}",
        f"segments_total: {lsd.get('segments_total', 0)}",
        f"refinement_levels: {refinement.get('levels_used', 0)}",
    ]
    diag_conf = refinement.get("aggregated_confidence")
    if diag_conf is not None:
        summary_lines.append(f"refine_conf: {diag_conf:.3f}")
    ax_plot.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=ax_plot.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.suptitle("Grid Detector Diagnostics", fontsize=14)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved visualization to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
