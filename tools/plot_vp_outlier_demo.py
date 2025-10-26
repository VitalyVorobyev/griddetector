#!/usr/bin/env python3
"""Visualize VP outlier demo output with inlier/outlier classification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

INLIER_COLOR = "#2ca02c"
OUTLIER_COLOR = "#d62728"

MODEL_COLORS: Dict[str, Dict[str, str]] = {
    "initial": {"u": "#1f77b4", "v": "#ff7f0e", "anchor": "#ffffff"},
    "inlier": {"u": "#17becf", "v": "#bcbd22", "anchor": "#ffdf5d"},
    "refined": {"u": "#9467bd", "v": "#8c564b", "anchor": "#e377c2"},
}

MODEL_STYLES: Dict[str, str] = {"initial": "-", "inlier": "--", "refined": ":"}


def figure_size(width: int, height: int, scale: float = 1.5) -> Tuple[float, float]:
    base = 110.0
    return (width / (base / scale), height / (base / scale))


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def gather_segments(
    segments: Iterable[dict],
    limit: int | None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    inliers: List[np.ndarray] = []
    outliers: List[np.ndarray] = []
    count = 0
    for seg in segments:
        if limit is not None and count >= limit:
            break
        p0 = seg.get("p0")
        p1 = seg.get("p1")
        if not is_two_vector(p0) or not is_two_vector(p1):
            continue
        line = np.array([[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]], dtype=float)
        if bool(seg.get("inlier", False)):
            inliers.append(line)
        else:
            outliers.append(line)
        count += 1
    return inliers, outliers


def is_two_vector(vec) -> bool:
    return isinstance(vec, (list, tuple)) and len(vec) == 2


def homogeneous_to_point(vec: np.ndarray) -> Tuple[np.ndarray, bool]:
    if abs(vec[2]) > 1e-6:
        return vec[:2] / vec[2], False
    return vec[:2], True


def clip_ray_to_bounds(
    origin: np.ndarray,
    direction: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    dx, dy = direction
    if np.linalg.norm(direction) < 1e-6:
        return None
    candidates: List[Tuple[float, float, float]] = []

    for x in (0.0, float(width)):
        if abs(dx) < 1e-6:
            continue
        t = (x - origin[0]) / dx
        if t <= 0:
            continue
        y = origin[1] + t * dy
        if 0.0 <= y <= float(height):
            candidates.append((t, x, y))

    for y in (0.0, float(height)):
        if abs(dy) < 1e-6:
            continue
        t = (y - origin[1]) / dy
        if t <= 0:
            continue
        x = origin[0] + t * dx
        if 0.0 <= x <= float(width):
            candidates.append((t, x, y))

    if not candidates:
        return None
    _, x, y = min(candidates, key=lambda item: item[0])
    return np.array([origin, [x, y]], dtype=float)


def compute_vp_ray(
    anchor: np.ndarray,
    vp: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    point, at_infinity = homogeneous_to_point(vp)
    if at_infinity:
        direction = point / (np.linalg.norm(point) + 1e-9)
        return clip_ray_to_bounds(anchor, direction, width, height)

    direction = point - anchor
    if np.linalg.norm(direction) < 1e-6:
        return None
    if 0.0 <= point[0] <= float(width) and 0.0 <= point[1] <= float(height):
        return np.array([anchor, point], dtype=float)
    return clip_ray_to_bounds(anchor, direction, width, height)


def draw_model(
    ax,
    model: dict | None,
    width: int,
    height: int,
    colors: Dict[str, str],
    linestyle: str,
) -> None:
    if not isinstance(model, dict):
        return
    matrix = np.array(model.get("coarseH", []), dtype=float)
    if matrix.shape != (3, 3):
        return

    anchor_h = matrix[:, 2]
    anchor_point, anchor_infinite = homogeneous_to_point(anchor_h)
    if not anchor_infinite:
        ax.scatter(
            anchor_point[0],
            anchor_point[1],
            c=colors.get("anchor", "#ffffff"),
            s=40,
            edgecolors="#000000",
            linewidths=0.75,
            zorder=5,
        )

    for key, column in (("u", 0), ("v", 1)):
        vp = matrix[:, column]
        ray = compute_vp_ray(anchor_point, vp, width, height)
        if ray is None:
            continue
        ax.plot(
            ray[:, 0],
            ray[:, 1],
            color=colors.get(key, "#888888"),
            linestyle=linestyle,
            linewidth=2.0,
        )


def plot_vp_outlier_demo(
    image_path: Path,
    result_path: Path,
    alpha: float,
    limit: int | None,
    save_path: Path | None,
) -> None:
    image = Image.open(image_path).convert("L")
    width, height = image.size
    data = load_result(result_path)
    segments = data.get("segments", [])
    stats = data.get("classification", {})
    perf = data.get("performanceMs", {})

    inlier_lines, outlier_lines = gather_segments(segments, limit)

    fig, ax = plt.subplots(figsize=figure_size(width, height), dpi=120)
    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()

    if inlier_lines:
        lc_in = LineCollection(inlier_lines, colors=INLIER_COLOR, linewidths=1.8, alpha=alpha)
        ax.add_collection(lc_in)
    if outlier_lines:
        lc_out = LineCollection(outlier_lines, colors=OUTLIER_COLOR, linewidths=1.4, alpha=alpha)
        ax.add_collection(lc_out)

    if True:
        draw_model(ax, data.get("initialModel"), width, height, MODEL_COLORS["initial"], MODEL_STYLES["initial"])
        draw_model(ax, data.get("inlierModel"), width, height, MODEL_COLORS["inlier"], MODEL_STYLES["inlier"])
    draw_model(ax, data.get("refinedModel"), width, height, MODEL_COLORS["refined"], MODEL_STYLES["refined"])

    handles: List[Line2D] = [
        Line2D([0], [0], color=INLIER_COLOR, linewidth=2.0, label="Inliers"),
        Line2D([0], [0], color=OUTLIER_COLOR, linewidth=2.0, label="Outliers"),
    ]
    if data.get("initialModel"):
        handles.extend([
            Line2D([0], [0], color=MODEL_COLORS["initial"]["u"], linestyle=MODEL_STYLES["initial"], linewidth=2.0, label="Initial VP\u2093"),
            Line2D([0], [0], color=MODEL_COLORS["initial"]["v"], linestyle=MODEL_STYLES["initial"], linewidth=2.0, label="Initial VP\u2094"),
        ])
    if data.get("inlierModel"):
        handles.extend([
            Line2D([0], [0], color=MODEL_COLORS["inlier"]["u"], linestyle=MODEL_STYLES["inlier"], linewidth=2.0, label="Inlier VP\u2093"),
            Line2D([0], [0], color=MODEL_COLORS["inlier"]["v"], linestyle=MODEL_STYLES["inlier"], linewidth=2.0, label="Inlier VP\u2094"),
        ])
    if data.get("refinedModel"):
        handles.extend([
            Line2D([0], [0], color=MODEL_COLORS["refined"]["u"], linestyle=MODEL_STYLES["refined"], linewidth=2.2, label="Refined VP\u2093"),
            Line2D([0], [0], color=MODEL_COLORS["refined"]["v"], linestyle=MODEL_STYLES["refined"], linewidth=2.2, label="Refined VP\u2094"),
        ])

    ax.legend(handles=handles, loc="upper right", frameon=True, facecolor="#ffffffdd")

    total = stats.get("total")
    inliers = stats.get("inliers")
    outliers = stats.get("outliers")
    total_ms = perf.get("totalMs")
    title_parts = ["VP Outlier Demo"]
    if total is not None and inliers is not None and outliers is not None:
        title_parts.append(f"segments={total} (inliers={inliers}, outliers={outliers})")
    if (initial := data.get("initialModel")) and isinstance(initial, dict):
        title_parts.append(f"H0 conf={initial.get('confidence', 0.0):.3f}")
    if (refined := data.get("refinedModel")) and isinstance(refined, dict):
        title_parts.append(f"refined conf={refined.get('confidence', 0.0):.3f}")
    if isinstance(total_ms, (int, float)):
        title_parts.append(f"total={total_ms:.1f} ms")
    ax.set_title(" | ".join(title_parts))

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.0)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


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
    plot_vp_outlier_demo(args.image, args.result, args.alpha, args.limit, args.save)


if __name__ == "__main__":
    main()
