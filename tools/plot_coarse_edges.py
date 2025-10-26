#!/usr/bin/env python3
"""Visualize coarsest-level image with detected edge elements and gradient directions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_edges(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    edges = data.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("edges field must be a list")
    return data, edges


def figure_size_for_image(width: int, height: int, scale: float = 1.5) -> tuple[float, float]:
    base = 100.0
    return (width / (base / scale), height / (base / scale))


def plot_edges(
    image_path: Path,
    edges_path: Path,
    scale: float,
    alpha: float,
    limit: int | None,
    save: Path | None,
) -> None:
    image = Image.open(image_path).convert("L")
    width, height = image.size
    print(f"Loaded image {image_path} with size {width}x{height}")
    data, edges = load_edges(edges_path)

    if limit is not None:
        edges = edges[:limit]

    if not edges:
        print("No edges to visualize.")

    xs = np.array([edge["x"] for edge in edges], dtype=float)
    ys = np.array([edge["y"] for edge in edges], dtype=float)
    mags = np.array([edge.get("magnitude", 0.0) for edge in edges], dtype=float)
    dirs = np.array([edge.get("direction", 0.0) for edge in edges], dtype=float)

    dx = np.cos(dirs) * mags * scale
    dy = np.sin(dirs) * mags * scale

    cmap = matplotlib.colormaps["plasma"]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    settings = data.get("settings", {}) if isinstance(data.get("settings"), dict) else {}
    threshold = settings.get("magnitudeThreshold")
    if threshold is None:
        threshold = data.get("magnitudeThreshold")
    thresh_text = f"{threshold}" if isinstance(threshold, (int, float)) else "n/a"
    ax.set_title(
        f"Coarsest Level Edges\n{len(edges)} elements, threshold={thresh_text}"
    )
    if len(edges) > 0:
        # ax.quiver(
        #     xs,
        #     ys,
        #     dx,
        #     dy,
        #     mags,
        #     cmap=cmap,
        #     angles="xy",
        #     scale_units="xy",
        #     scale=10,
        #     width=0.003,
        #     alpha=alpha,
        #     minlength=0,
        # )
        ax.scatter(xs, ys, s=6, c=mags, cmap=cmap, alpha=alpha, linewidths=0)

    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight", pad_inches=0)
        print(f"Saved visualization to {save}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot coarsest-level grayscale image with edge detections."
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to coarsest PNG image")
    parser.add_argument("--edges", required=True, type=Path, help="Path to edges JSON output")
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="Scale factor for gradient arrows (default: 10.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Opacity for edge overlay (default: 0.8)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Plot at most N edges (useful for dense outputs)",
    )
    parser.add_argument("--save", type=Path, help="Save plot instead of showing interactively")
    args = parser.parse_args()

    plot_edges(args.image, args.edges, args.scale, args.alpha, args.limit, args.save)


if __name__ == "__main__":
    main()
