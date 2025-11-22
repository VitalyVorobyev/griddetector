"""
Visualize the grid hypothesis produced by grid_hyp_demo.

Usage:
    python tools/plot_grid_hypothesis.py -r out/grid_hyp.json --image path/to/coarsest.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_report(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def plot_segments(ax: plt.Axes, segments: List[Dict[str, Any]], color: str, lw: float = 0.5):
    for seg in segments:
        p0 = seg["p0"]
        p1 = seg["p1"]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw, alpha=0.7)


def plot_bundles(ax: plt.Axes, bundles: List[Dict[str, Any]], color: str, lw: float = 1.5):
    h, w = ax.images[0].get_array().shape[:2]
    xs = np.array([0, w])
    for b in bundles:
        a, c, d = b["line"]
        if abs(c) < 1e-6 and abs(a) < 1e-6:
            continue
        if abs(c) > abs(a):
            ys = -(a * xs + d) / c
            ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.9)
        else:
            ys = np.array([0, h])
            xs_line = -(c * ys + d) / a
            ax.plot(xs_line, ys, color=color, linewidth=lw, alpha=0.9)
        ax.scatter(b["center"][0], b["center"][1], color=color, s=4)


def plot_vp(ax: plt.Axes, vp: Dict[str, Any], color: str):
    pos = vp["pos"]
    if abs(pos[2]) <= 1e-6:
        # VP at infinity: draw a small arrow.
        dir_ = np.array(vp["dir"])
        center = np.array(ax.images[0].get_array().shape[:2][::-1]) * 0.5
        end = center + dir_ * 50.0
        ax.annotate(
            "",
            xy=end,
            xytext=center,
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
        )
    else:
        ax.scatter(pos[0], pos[1], color=color, marker="x", s=40, linewidths=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot grid hypothesis.")
    ap.add_argument("-r", "--report", required=True, type=Path, help="JSON output from grid_hyp_demo")
    ap.add_argument("--image", type=Path, help="Image to overlay (coarsest level).")
    args = ap.parse_args()

    data = load_report(args.report)
    img = None
    if args.image:
        img = load_image(args.image)
    else:
        # Try to infer image path next to report.
        candidate = args.report.parent / "pyramid_L0.png"
        if candidate.exists():
            img = load_image(candidate)
    if img is None:
        raise FileNotFoundError("Image not provided and default pyramid_L0.png not found.")

    fig, ax = plt.subplots()
    ax.imshow(img)
    plot_segments(ax, data["refined_segments"], color="cyan", lw=0.5)
    plot_bundles(ax, data["hypothesis"]["bundles"], color="orange", lw=1.0)
    vp = data["hypothesis"].get("vp")
    if vp:
        plot_vp(ax, vp["u"], color="red")
        plot_vp(ax, vp["v"], color="blue")

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
