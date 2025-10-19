#!/usr/bin/env python3
"""Visualize coarsest-level image with detected line segments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image


def load_segments(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    segments = data.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("segments field must be a list")
    return data, segments


def figure_size_for_image(width: int, height: int, scale: float = 1.5) -> tuple[float, float]:
    base = 120.0
    return (width / (base / scale), height / (base / scale))


def metric_value(segment: dict, metric: str) -> float:
    if metric == "strength":
        return float(segment.get("strength", 0.0))
    if metric == "average-magnitude":
        return float(segment.get("averageMagnitude", 0.0))
    if metric == "length":
        return float(segment.get("length", 0.0))
    raise ValueError(f"Unknown metric '{metric}'")


def plot_segments(
    image_path: Path,
    segments_path: Path,
    scale: float,
    alpha: float,
    limit: int | None,
    metric: str,
    save: Path | None,
) -> None:
    image = Image.open(image_path).convert("L")
    width, height = image.size
    print(f"Loaded image {image_path} with size {width}x{height}")
    data, segments = load_segments(segments_path)

    if limit is not None:
        segments = segments[:limit]

    if not segments:
        print("No segments to visualize.")

    lines = []
    values: list[float] = []
    for seg in segments:
        p0 = seg.get("p0")
        p1 = seg.get("p1")
        if not (isinstance(p0, (list, tuple)) and isinstance(p1, (list, tuple))):
            continue
        if len(p0) != 2 or len(p1) != 2:
            continue
        lines.append([(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))])
        values.append(metric_value(seg, metric))

    if not lines:
        print("No valid segments after filtering; nothing to draw.")

    cmap = matplotlib.colormaps["plasma"]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(
        f"Coarsest Level Segments\n{len(lines)} segments, "
        f"metric={metric}, threshold={data.get('magnitudeThreshold', 'n/a')}"
    )

    if lines:
        segments_arr = np.array(lines)
        lc = LineCollection(
            segments_arr,
            linewidths=1.5,
            colors=None,
            cmap=cmap,
            norm=None,
            alpha=alpha,
        )
        if values:
            lc.set_array(np.array(values))
        ax.add_collection(lc)
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
        description="Plot coarsest-level grayscale image with LSD line segments."
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to coarsest PNG image")
    parser.add_argument(
        "--segments", required=True, type=Path, help="Path to segments JSON output"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help="Figure size scale relative to image resolution (default: 1.5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Opacity for segment overlay (default: 0.8)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Visualize at most N segments (useful for dense outputs)",
    )
    parser.add_argument(
        "--metric",
        choices=["strength", "average-magnitude", "length"],
        default="strength",
        help="Color segments by the chosen metric (default: strength)",
    )
    parser.add_argument("--save", type=Path, help="Save plot instead of showing interactively")
    args = parser.parse_args()

    plot_segments(
        args.image,
        args.segments,
        args.scale,
        args.alpha,
        args.limit,
        args.metric,
        args.save,
    )


if __name__ == "__main__":
    main()
