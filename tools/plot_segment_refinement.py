from dataclasses import dataclass
import json

import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from vp_outlier_result import LsdVp

@dataclass
class Segment:
    p0: np.ndarray
    p1: np.ndarray

@dataclass
class RefinedLevel:
    segments: list[Segment]
    acceptance: float

@dataclass
class SegmentRefinementReport:
    lsd: LsdVp
    initial_segments: list[Segment]
    levels: list[RefinedLevel]

def read_report(file_path: str) -> SegmentRefinementReport:
    with open(file_path, "r") as f:
        data = json.load(f)

    lsd = LsdVp.from_json(data["lsd"])
    initial_segments = [Segment(np.array(seg['p0']), np.array(seg['p1'])) for seg in data['lsdSegments']]
    levels = []
    for level in data['levels']:
        segments = [Segment(np.array(seg['segment']['p0']),
                            np.array(seg['segment']['p1'])) for seg in level['results'] if seg['ok']]
        acceptance = level.get('acceptanceRatio', 0.0)
        levels.append(RefinedLevel(segments, acceptance))
    return SegmentRefinementReport(lsd, initial_segments, levels[::-1])

def load_image(level: int) -> np.ndarray:
    image_path = f"out/segment_demo/pyramid_L{level}.png"
    img = Image.open(image_path).convert("L")
    return np.array(img)

def plot_segments(segments: list[Segment], img:np.ndarray, color: str):
    plt.figure()
    plt.imshow(img, cmap='gray')
    for seg in segments:
        plt.plot([seg.p0[0], seg.p1[0]], [seg.p0[1], seg.p1[1]], color=color)
    plt.axis('off')
    plt.tight_layout()

def main():
    parser = argparse.ArgumentParser(description="Plot segment refinement process.")
    parser.add_argument('-r', '--report', type=str, help="Path to the segment refinement report JSON file.")
    parser.add_argument('-l', '--level', type=int, default=0, help="Refinement level to plot.")
    args = parser.parse_args()
    r = read_report(args.report)
    print(f"Number of levels: {len(r.levels)}")

    if args.level == len(r.levels):
        segments = r.initial_segments
    else:
        segments = r.levels[args.level].segments

    for i, level in enumerate(r.levels):
        print(f"Level {i}: Acceptance = {level.acceptance:.2f}, Segments = {len(level.segments)}")

    img = load_image(args.level)
    plot_segments(segments, img, color='red')
    plt.show()

if __name__ == "__main__":
    main()
