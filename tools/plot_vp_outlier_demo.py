#!/usr/bin/env python3
"""Visualize VP outlier demo output with inlier/outlier classification.

Refactored to use helpers in tools/plot_utils.py to keep the script concise
and consistent with other tools.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

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
from vp_utils import estimate_grid_vps, Segment, rectify_segments

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
    H_full: np.ndarray,
    src_w: int,
    src_h: int,
    img_w: int,
    img_h: int,
    alpha: float,
) -> tuple[plt.Figure, plt.Axes] | tuple[None, None]:
    """Visualize rectified segments with bounds driven by the rectified inliers.

    The goal is to show the grid defined by the kept segments, so the viewport is
    anchored to rectified inliers when available and only falls back to outliers
    when no inliers remain.  A small percentile-based clamp combined with a
    maximum extent keeps a single errant segment from blowing up the scale.  The
    chosen bounds are printed to stdout so demo users can quickly spot
    pathological rectifications without inspecting the matplotlib UI.
    """
    H_rect = rescale_homography_image_space(H_full, src_w, src_h, img_w, img_h)

    rect_inliers: List[np.ndarray] = []
    rect_outliers: List[np.ndarray] = []
    for seg in inlier_lines:
        r = apply_homography_points(H_rect, seg)
        if r is not None:
            rect_inliers.append(r)
    for seg in outlier_lines:
        r = apply_homography_points(H_rect, seg)
        if r is not None:
            rect_outliers.append(r)
    if not rect_inliers and not rect_outliers:
        return None, None

    def _stack_points(segments: List[np.ndarray]) -> np.ndarray:
        if not segments:
            return np.empty((0, 2), dtype=float)
        return np.concatenate(segments, axis=0)

    inlier_pts = _stack_points(rect_inliers)
    outlier_pts = _stack_points(rect_outliers)
    if inlier_pts.size:
        primary_pts = inlier_pts
        primary_label = "inliers"
        primary_count = len(rect_inliers)
    else:
        primary_pts = outlier_pts
        primary_label = "outliers"
        primary_count = len(rect_outliers)
    if primary_pts.size == 0:
        return None, None

    percentile_clip = 2.5
    max_extent = 2500.0
    raw_min = np.min(primary_pts, axis=0)
    raw_max = np.max(primary_pts, axis=0)
    if 0.0 < percentile_clip < 50.0:
        lower = np.percentile(primary_pts, percentile_clip, axis=0)
        upper = np.percentile(primary_pts, 100.0 - percentile_clip, axis=0)
        min_xy = lower
        max_xy = upper
    else:
        min_xy = raw_min
        max_xy = raw_max

    span = max_xy - min_xy
    span = np.where(span <= 0.0, 1.0, span)
    if max_extent is not None:
        center = 0.5 * (max_xy + min_xy)
        span = np.minimum(span, max_extent)
        min_xy = center - 0.5 * span
        max_xy = center + 0.5 * span

    pad = 0.05 * span
    xmin, ymin = (min_xy - pad)
    xmax, ymax = (max_xy + pad)

    print(
        f"[rectified bounds] using {primary_label} (n={primary_count}) "
        f"x=({xmin:.2f}, {xmax:.2f}) y=({ymin:.2f}, {ymax:.2f})"
    )

    fig, ax = plt.subplots(figsize=figure_size(img_w, img_h, scale=1.2))
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_on()
    ax.set_xlim(float(xmin), float(xmax))
    ax.set_ylim(float(ymin), float(ymax))
    if rect_inliers:
        ax.add_collection(LineCollection(rect_inliers, colors=INLIER_COLOR, linewidths=1.8, alpha=alpha))
    if False and rect_outliers:
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
    m_coarse = model.trace.coarse_homography.T if model.trace.coarse_homography is not None else None
    m_refined = model.grid.hmtx.T if model.grid.hmtx is not None else None
    fig_overlay, _ = make_segments_figure(image, inlier_lines, outlier_lines, m_coarse, m_refined, src_w, src_h)
    if False:
        fig_bundles, _ = make_bundles_figure(image, bundle_lines, bundle_weights, bundle_centers, None)
    rect_fig, _ = make_rectified_figure(inlier_lines, outlier_lines, m_refined, src_w, src_h, width, height, alpha)

def plot_segments(segments: list[Segment], family: list[str], color_u: str, color_v: str, img:Image.Image|None) -> None:
    fig = plt.figure()
    if img is not None:
        plt.imshow(img, cmap="gray", origin="upper")
        plt.xlim(0, img.size[0])
        plt.ylim(img.size[1], 0)
    else:
        plt.axis('equal')
        plt.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)

    for seg, fam in zip(segments, family):
        color = color_u if fam == 'u' else color_v
        plt.plot(
            [seg.p0[0], seg.p1[0]],
            [seg.p0[1], seg.p1[1]],
            color=color,
            linewidth=1.5,
            alpha=0.8,
        )
    fig.tight_layout()

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

    segs, fam = model.trace.segments, model.trace.lsd.segment_families

    hmtx = estimate_grid_vps(segs, fam)
    print(f"Estimated grid homography from segments:\n{hmtx}")

    hmtx_coarse = model.trace.coarse_homography
    if hmtx_coarse is None:
        return
    
    hmtx_coarse /= 8
    hmtx_coarse[-1] = 1.0
    print(f"Original coarse homography:\n{hmtx_coarse}")
    image = Image.open(args.image).convert("L")

    rsegs = rectify_segments(model.trace.segments, hmtx)
    plot_segments(rsegs, fam, color_u="#1f77b4", color_v="#ff7f0e", img=None)
    plot_segments(segs, fam, color_u="#1f77b4", color_v="#ff7f0e", img=image)

    plt.show()

if __name__ == "__main__":
    main()
