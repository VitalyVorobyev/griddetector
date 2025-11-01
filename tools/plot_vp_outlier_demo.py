#!/usr/bin/env python3
"""Visualize VP outlier demo output with inlier/outlier classification.

Refactored to use helpers in tools/plot_utils.py to keep the script concise
and consistent with other tools.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from plot_utils import (  # type: ignore
    rescale_homography_image_space,
    apply_homography_points,
    homogeneous_to_point,
    compute_vp_ray,
    extract_model_columns,
    figure_size,
)
from vp_outlier_result import VpOutlierReport  # type: ignore
from utils import gather_segments_from_model, gather_bundle_lines_from_model

INLIER_COLOR = "#2ca02c"
OUTLIER_COLOR = "#d62728"

MODEL_COLORS: Dict[str, Dict[str, str]] = {
    "coarse": {"u": "#1f77b4", "v": "#ff7f0e", "anchor": "#ffffff"},
    "refined": {"u": "#9467bd", "v": "#8c564b", "anchor": "#e377c2"},
}
MODEL_STYLES: Dict[str, str] = {"coarse": "--", "refined": "-"}

def _draw_model(ax, hmtx: np.ndarray, width: int, height: int, colors: Dict[str, str], linestyle: str) -> None:
    if hmtx is None or not isinstance(hmtx, np.ndarray) or hmtx.shape != (3, 3):
        return
    cols = extract_model_columns(hmtx)
    # Use the anchor from H instead of forcing the exact center
    anchor_xy, _ = homogeneous_to_point(cols["anchor"])
    for key in ("u", "v"):
        seg = compute_vp_ray(anchor_xy, cols[key], width, height)
        if seg is not None:
            ax.plot(seg[:, 0], seg[:, 1], color=colors.get(key, "#888"), linestyle=linestyle, linewidth=2.0)


# Standalone figures
def make_segments_figure(
    image: Image.Image,
    inlier_lines: list[np.ndarray],
    outlier_lines: list[np.ndarray],
    m_coarse: np.ndarray | None,
    m_refined: np.ndarray | None,
    src_w: int,
    src_h: int,
) -> tuple[plt.Figure, plt.Axes]:
    width, height = image.size
    fig, ax = plt.subplots(figsize=figure_size(width, height), dpi=120)
    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    if inlier_lines:
        ax.add_collection(LineCollection(inlier_lines, colors=INLIER_COLOR, linewidths=1.8, alpha=0.85))
    if outlier_lines:
        ax.add_collection(LineCollection(outlier_lines, colors=OUTLIER_COLOR, linewidths=1.4, alpha=0.85))

    if m_coarse is not None:
        m_draw = rescale_homography_image_space(m_coarse, src_w, src_h, width, height)
        _draw_model(ax, m_draw, width, height, MODEL_COLORS["coarse"], MODEL_STYLES["coarse"])
    if m_refined is not None:
        m_draw = rescale_homography_image_space(m_refined, src_w, src_h, width, height)
        _draw_model(ax, m_draw, width, height, MODEL_COLORS["refined"], MODEL_STYLES["refined"])

    handles: List[Line2D] = [
        Line2D([0], [0], color=INLIER_COLOR, linewidth=2.0, label="Inliers"),
        Line2D([0], [0], color=OUTLIER_COLOR, linewidth=2.0, label="Outliers"),
        Line2D([0], [0], color=MODEL_COLORS["coarse"]["u"], linestyle=MODEL_STYLES["coarse"], linewidth=2.0, label="Coarse VPx"),
        Line2D([0], [0], color=MODEL_COLORS["coarse"]["v"], linestyle=MODEL_STYLES["coarse"], linewidth=2.0, label="Coarse VPy"),
        Line2D([0], [0], color=MODEL_COLORS["refined"]["u"], linestyle=MODEL_STYLES["refined"], linewidth=2.2, label="Refined VPx"),
        Line2D([0], [0], color=MODEL_COLORS["refined"]["v"], linestyle=MODEL_STYLES["refined"], linewidth=2.2, label="Refined VPy"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, facecolor="#ffffffdd")
    fig.tight_layout()
    return fig, ax

def make_bundles_figure(
    image: Image.Image,
    bundle_lines: list[np.ndarray],
    bundle_weights: list[float],
    bundle_centers: list[np.ndarray],
    bundling_stage: dict | None,
) -> tuple[plt.Figure, plt.Axes] | tuple[None, None]:
    if not bundle_lines:
        return None, None
    width, height = image.size
    fig, ax = plt.subplots(figsize=figure_size(width, height), dpi=120)
    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()

    weights = np.asarray(bundle_weights, dtype=float)
    if weights.size:
        w_min = float(np.min(weights))
        w_max = float(np.max(weights))
        if w_max > w_min:
            norm = (weights - w_min) / (w_max - w_min)
        else:
            norm = np.zeros_like(weights)
    else:
        norm = np.array([])
    colors = plt.cm.plasma(0.3 + 0.7 * norm) if norm.size else plt.cm.plasma(0.6)
    widths = 0.9 + 2.2 * norm if norm.size else 1.5

    lc_bundle = LineCollection(bundle_lines, colors=colors, linewidths=widths, alpha=0.9)
    ax.add_collection(lc_bundle)

    centers_arr = np.asarray(bundle_centers)
    if centers_arr.size:
        sizes = 20.0 + 40.0 * norm if norm.size else 30.0
        ax.scatter(
            centers_arr[:, 0],
            centers_arr[:, 1],
            s=sizes,
            c=colors if norm.size else [colors],
            edgecolors="white",
            linewidths=0.3,
            alpha=0.9,
        )

    fig.tight_layout()
    return fig, ax

def make_rectified_figure(
    inlier_lines: list[np.ndarray],
    outlier_lines: list[np.ndarray],
    H_full: np.ndarray | None,
    src_w: int,
    src_h: int,
    img_w: int,
    img_h: int,
    alpha: float,
) -> tuple[plt.Figure, plt.Axes] | tuple[None, None]:
    if H_full is None:
        return None, None
    sx = float(img_w) / float(src_w) if src_w else 1.0
    sy = float(img_h) / float(src_h) if src_h else 1.0
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    try:
        H_inv_full = np.linalg.inv(H_full)
        S_inv = np.linalg.inv(S)
        H_rect = H_inv_full @ S_inv
    except np.linalg.LinAlgError:
        return None, None

    rect_inliers: List[np.ndarray] = []
    rect_outliers: List[np.ndarray] = []
    all_pts: List[np.ndarray] = []
    for seg in inlier_lines:
        r = apply_homography_points(H_rect, seg)
        if r is not None:
            rect_inliers.append(r)
            all_pts.append(r)
    for seg in outlier_lines:
        r = apply_homography_points(H_rect, seg)
        if r is not None:
            rect_outliers.append(r)
            all_pts.append(r)
    if not all_pts:
        return None, None
    pts = np.concatenate(all_pts, axis=0)
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    pad_x = 0.05 * dx if dx > 0 else 1.0
    pad_y = 0.05 * dy if dy > 0 else 1.0

    fig, ax = plt.subplots(figsize=figure_size(img_w, img_h, scale=1.2))
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_on()
    ax.set_xlim(float(xmin - pad_x), float(xmax + pad_x))
    ax.set_ylim(float(ymin - pad_y), float(ymax + pad_y))
    if rect_inliers:
        ax.add_collection(LineCollection(rect_inliers, colors=INLIER_COLOR, linewidths=1.8, alpha=alpha))
    if rect_outliers:
        ax.add_collection(LineCollection(rect_outliers, colors=OUTLIER_COLOR, linewidths=1.4, alpha=alpha))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    return fig, ax

def print_summary(model:VpOutlierReport):
    # Print summary (instead of titles)
    total = model.trace.outlier_filter.total
    kept = model.trace.outlier_filter.kept
    rejected = model.trace.outlier_filter.rejected
    total_ms = model.trace.timings.total_ms
    print("VP Outlier Demo summary:")
    print(f"  segments={total} (kept={kept}, rejected={rejected})")
    print(f"  refined conf={model.grid.confidence:.3f}")
    print(f"  total={total_ms:.1f} ms")

def plot_vp_outlier_demo(
    image_path: Path,
    model: VpOutlierReport,
    alpha: float,
    limit: int | None,
    save_path: Path | None,
) -> None:
    image = Image.open(image_path).convert("L")
    width, height = image.size
    inlier_lines, outlier_lines = gather_segments_from_model(
        model.trace.segments, model.trace.outlier_filter.classifications, limit
    )
    bundle_lines, bundle_weights, bundle_centers = gather_bundle_lines_from_model(
        model.trace.bundling, width, height, limit
    )

    # Build separate figures
    src_w = int(model.trace.input.width or width)
    src_h = int(model.trace.input.height or height)
    m_coarse = model.trace.coarse_homography
    m_refined = model.grid.hmtx
    fig_overlay, _ = make_segments_figure(image, inlier_lines, outlier_lines, m_coarse, m_refined, src_w, src_h)
    fig_bundles, _ = make_bundles_figure(image, bundle_lines, bundle_weights, bundle_centers, None)
    rect_fig, _ = make_rectified_figure(inlier_lines, outlier_lines, m_refined, src_w, src_h, width, height, alpha)

    # Save / show figures
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig_overlay.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.0)
        print(f"Saved overlay to {save_path}")
        if fig_bundles is not None:
            bpath = save_path.with_name(save_path.stem + "_bundles" + save_path.suffix)
            fig_bundles.savefig(bpath, dpi=150, bbox_inches="tight", pad_inches=0.0)
            print(f"Saved bundles to {bpath}")
        if rect_fig is not None:
            rpath = save_path.with_name(save_path.stem + "_rectified" + save_path.suffix)
            rect_fig.savefig(rpath, dpi=150, bbox_inches="tight", pad_inches=0.0)
            print(f"Saved rectified visualization to {rpath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize VP outlier demo output with inlier/outlier classification.",
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to coarsest PNG image")
    parser.add_argument("--result", required=True, type=Path, help="Path to demo JSON output")
    parser.add_argument("--alpha", type=float, default=0.85, help="Opacity for plotted segments")
    parser.add_argument("--limit", type=int, default=None, help="Plot at most N segments (default: all)")
    parser.add_argument("--save", type=Path, help="Save figure instead of showing it interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = VpOutlierReport.from_path(args.result)
    plot_vp_outlier_demo(args.image, model, args.alpha, args.limit, args.save)
    print_summary(model)
    plt.show()

if __name__ == "__main__":
    main()
