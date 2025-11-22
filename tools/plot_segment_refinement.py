"""
Plot refined segments from `segment_refine_demo` output onto a pyramid image.

Usage:
    python tools/plot_segment_refinement.py -r out/segment_demo/report.json
    # optionally pick a pyramid level or an explicit image:
    python tools/plot_segment_refinement.py -r out/segment_demo/report.json -l 2
    python tools/plot_segment_refinement.py -r out/segment_demo/report.json --image path/to/image.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


@dataclass
class Segment:
    p0: np.ndarray
    p1: np.ndarray


@dataclass
class LevelStats:
    elapsed_ms: float
    accepted: int


@dataclass
class SegmentRefinementReport:
    segments: List[Segment]
    levels: List[LevelStats]


def read_report(file_path: Path) -> SegmentRefinementReport:
    """Parse the JSON written by `segment_refine_demo`."""
    with file_path.open("r") as f:
        data = json.load(f)

    refine = data.get("refine", {})
    segments = [
        Segment(np.array(seg["p0"], dtype=float), np.array(seg["p1"], dtype=float))
        for seg in refine.get("segments", [])
    ]
    levels = [
        LevelStats(level["elapsed_ms"], level["accepted"])
        for level in refine.get("levels", [])
    ]
    return SegmentRefinementReport(segments=segments, levels=levels)


def load_image(img_path: Path) -> np.ndarray:
    with Image.open(img_path) as img:
        return np.array(img.convert("RGB"))


def default_image_for_level(report_path: Path, level: int) -> Path:
    """Choose pyramid image path relative to the report directory."""
    return report_path.parent / f"pyramid_L{level}.png"


def plot_segments(
    segments: Sequence[Segment],
    img: np.ndarray,
    color: str = "red",
    linewidth: float = 1.0,
) -> None:
    plt.figure()
    plt.imshow(img)
    for seg in segments:
        plt.plot(
            [seg.p0[0], seg.p1[0]],
            [seg.p0[1], seg.p1[1]],
            color=color,
            linewidth=linewidth,
        )
    plt.axis("off")
    plt.tight_layout()


def summarize(levels: Iterable[LevelStats]) -> None:
    for idx, lvl in enumerate(levels):
        print(f"Level {idx}: {lvl.accepted} segments, {lvl.elapsed_ms:5.2f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot refined segments from segment_refine_demo output."
    )
    parser.add_argument(
        "-r",
        "--report",
        required=True,
        type=Path,
        help="Path to report.json produced by segment_refine_demo.",
    )
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=0,
        help="Pyramid level image to overlay (default: 0 / finest).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional explicit image path; overrides --level.",
    )
    parser.add_argument(
        "--color",
        default="red",
        help="Matplotlib color for segments.",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=1.0,
        help="Line width for segment overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = read_report(args.report)
    summarize(report.levels)

    img_path = args.image or default_image_for_level(args.report, args.level)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = load_image(img_path)

    print(
        f"Loaded {len(report.segments)} refined segments; "
        f"overlaying on {img_path.name}"
    )
    plot_segments(report.segments, img, color=args.color, linewidth=args.linewidth)
    plt.show()


if __name__ == "__main__":
    main()
