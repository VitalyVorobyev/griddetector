#!/usr/bin/env python3
"""Visualize LSD→VP demo output with segment families and coarse basis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image

FAMILY_COLORS: Dict[str, str] = {
    "u": "#1f77b4",  # blue
    "v": "#ff7f0e",  # orange
    "none": "#8a8a8a",  # gray
}


def figure_size(width: int, height: int, scale: float = 1.5) -> Tuple[float, float]:
    base = 110.0
    return (width / (base / scale), height / (base / scale))


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def gather_lines(segments: Iterable[dict], limit: int | None) -> Dict[str, List[np.ndarray]]:
    grouped = {"u": [], "v": [], "none": []}
    count = 0
    for seg in segments:
        if limit is not None and count >= limit:
            break
        p0 = seg.get("p0")
        p1 = seg.get("p1")
        if not (is_two_vector(p0) and is_two_vector(p1)):
            continue
        line = np.array([[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]], dtype=float)
        family = seg.get("family")
        key = family if family in ("u", "v") else "none"
        grouped[key].append(line)
        count += 1
    return grouped


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
    # If the VP lies inside the image, draw directly; otherwise clip to border.
    if 0.0 <= point[0] <= float(width) and 0.0 <= point[1] <= float(height):
        return np.array([anchor, point], dtype=float)
    return clip_ray_to_bounds(anchor, direction, width, height)


def plot_lsd_vp(
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

    grouped = gather_lines(segments, limit)

    fig, ax = plt.subplots(figsize=figure_size(width, height), dpi=120)
    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()

    for key, lines in grouped.items():
        if not lines:
            continue
        lc = LineCollection(lines, colors=FAMILY_COLORS[key], linewidths=1.5, alpha=alpha)
        ax.add_collection(lc)

    hypothesis = data.get("hypothesis")
    if isinstance(hypothesis, dict):
        hmtx = np.array(hypothesis.get("hmtx0", []), dtype=float)
        if hmtx.shape == (3, 3):
            anchor_h = hmtx[:, 2]
            vpu = hmtx[:, 0]
            vpv = hmtx[:, 1]
            anchor, _ = homogeneous_to_point(anchor_h)
            ax.scatter(anchor[0], anchor[1], c="#ffffff", s=36, edgecolors="#000000", zorder=5)
            arrow_u = compute_vp_ray(anchor, vpu, width, height)
            arrow_v = compute_vp_ray(anchor, vpv, width, height)
            if arrow_u is not None:
                ax.plot(arrow_u[:, 0], arrow_u[:, 1], color=FAMILY_COLORS["u"], linewidth=2.0)
            if arrow_v is not None:
                ax.plot(arrow_v[:, 0], arrow_v[:, 1], color=FAMILY_COLORS["v"], linewidth=2.0)

    title = [f"LSD→VP Demo", f"segments={len(segments)}"]
    if data.get("dominantAnglesDeg"):
        angles = data["dominantAnglesDeg"]
        title.append(
            f"θ_u={angles[0]:.1f}° θ_v={angles[1]:.1f}°"
        )
    ax.set_title(" | ".join(title))

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.0)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize LSD→VP demo output with segment clusters and vanishing rays.")
    parser.add_argument("--image", required=True, type=Path, help="Path to coarsest PNG image")
    parser.add_argument("--result", required=True, type=Path, help="Path to LSD-VP JSON output")
    parser.add_argument("--alpha", type=float, default=0.85, help="Opacity for plotted segments")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Visualize at most N segments (default: all)",
    )
    parser.add_argument("--save", type=Path, help="Save figure instead of showing interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_lsd_vp(args.image, args.result, args.alpha, args.limit, args.save)


if __name__ == "__main__":
    main()
