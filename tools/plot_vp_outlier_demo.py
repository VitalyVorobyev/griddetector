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
    "coarse": {"u": "#1f77b4", "v": "#ff7f0e", "anchor": "#ffffff"},
    "refined": {"u": "#9467bd", "v": "#8c564b", "anchor": "#e377c2"},
}

MODEL_STYLES: Dict[str, str] = {"coarse": "--", "refined": "-"}


def figure_size(width: int, height: int, scale: float = 1.5) -> Tuple[float, float]:
    base = 110.0
    return (width / (base / scale), height / (base / scale))


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_detection_report(data: dict) -> tuple[dict, dict]:
    if isinstance(data.get("report"), dict):
        data = data["report"]
    trace = data.get("trace", {}) if isinstance(data.get("trace"), dict) else {}
    return data, trace

def to_matrix3x3(matrix) -> np.ndarray | None:
    try:
        arr = np.array(matrix, dtype=float)
    except Exception:
        return None
    if arr.shape == (3, 3):
        return arr
    if arr.ndim == 1 and arr.size == 9:
        # nalgebra Matrix3 serde flattens in column-major order
        return arr.reshape((3, 3), order="F")
    return None

def rescale_homography(mat: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    if not (src_w and src_h and dst_w and dst_h):
        return mat
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return S @ mat


def gather_segments(
    descriptors: Dict[int, dict],
    classifications: Iterable[dict],
    limit: int | None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    inliers: List[np.ndarray] = []
    outliers: List[np.ndarray] = []
    count = 0
    for entry in classifications:
        if limit is not None and count >= limit:
            break
        seg_id = entry.get("segment")
        if seg_id is None:
            continue
        seg = descriptors.get(int(seg_id))
        if not seg:
            continue
        p0 = seg.get("p0")
        p1 = seg.get("p1")
        if not is_two_vector(p0) or not is_two_vector(p1):
            continue
        line = np.array([[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]], dtype=float)
        if entry.get("class") == "kept":
            inliers.append(line)
        else:
            outliers.append(line)
        count += 1
    # Fallback: if no classification data, treat every segment as an inlier for plotting.
    if not inliers and not outliers:
        for seg in descriptors.values():
            if limit is not None and count >= limit:
                break
            p0 = seg.get("p0")
            p1 = seg.get("p1")
            if not is_two_vector(p0) or not is_two_vector(p1):
                continue
            line = np.array([[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]], dtype=float)
            inliers.append(line)
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
    matrix: list | tuple | None,
    width: int,
    height: int,
    colors: Dict[str, str],
    linestyle: str,
    transpose: bool = False
) -> None:
    if matrix is None:
        return
    mat = np.array(matrix, dtype=float)
    if mat.shape != (3, 3):
        return
    if transpose:
        mat = mat.T

    # Always use image center as the anchor, ignore matrix-provided anchor.
    anchor_point = np.array([width / 2.0, height / 2.0], dtype=float)

    for key, column in (("u", 0), ("v", 1)):
        vp = mat[:, column]
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


def clip_line_to_bounds(
    coeffs: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    a, b, c = coeffs
    eps = 1e-6
    w = float(width)
    h = float(height)
    intersections: List[np.ndarray] = []

    if abs(b) > eps:
        for x in (0.0, w):
            y = -(a * x + c) / b
            if -eps <= y <= h + eps:
                intersections.append(np.array([x, np.clip(y, 0.0, h)], dtype=float))
    if abs(a) > eps:
        for y in (0.0, h):
            x = -(b * y + c) / a
            if -eps <= x <= w + eps:
                intersections.append(np.array([np.clip(x, 0.0, w), y], dtype=float))

    unique: List[np.ndarray] = []
    tol = 1e-4
    for pt in intersections:
        if not any(np.linalg.norm(pt - other) < tol for other in unique):
            unique.append(pt)

    if len(unique) < 2:
        return None

    if len(unique) > 2:
        max_dist = -1.0
        best = (unique[0], unique[1])
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                dist = float(np.linalg.norm(unique[i] - unique[j]))
                if dist > max_dist:
                    max_dist = dist
                    best = (unique[i], unique[j])
        return np.stack(best, axis=0)

    return np.stack(unique, axis=0)


def gather_bundle_lines(
    bundling_stage: dict,
    image_width: int,
    image_height: int,
    limit: int | None,
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
    levels = bundling_stage.get("levels")
    if not isinstance(levels, list) or not levels:
        return [], [], []

    stage_scale = bundling_stage.get("scaleApplied")

    def parse_scale(value) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 1.0
        return parsed if abs(parsed) > 1e-9 else 1.0

    has_stage_scale = isinstance(stage_scale, (list, tuple)) and len(stage_scale) == 2

    if has_stage_scale:
        stage_scale_x = parse_scale(stage_scale[0])
        stage_scale_y = parse_scale(stage_scale[1])
    else:
        stage_scale_x = 1.0
        stage_scale_y = 1.0

    bundle_lines: List[np.ndarray] = []
    bundle_weights: List[float] = []
    bundle_centers: List[np.ndarray] = []
    count = 0

    for level in levels:
        if limit is not None and count >= limit:
            break
        if not isinstance(level, dict):
            continue
        level_width = float(level.get("width", image_width))
        level_height = float(level.get("height", image_height))
        if not has_stage_scale:
            bundle_space_w = level_width
            bundle_space_h = level_height
        else:
            bundle_space_w = level_width * stage_scale_x
            bundle_space_h = level_height * stage_scale_y

        scale_x = float(image_width) / bundle_space_w if bundle_space_w > 1e-6 else 1.0
        scale_y = float(image_height) / bundle_space_h if bundle_space_h > 1e-6 else 1.0

        bundles = level.get("bundles", [])
        if not isinstance(bundles, list):
            continue
        for bundle in bundles:
            if limit is not None and count >= limit:
                break
            if not isinstance(bundle, dict):
                continue
            center = bundle.get("center")
            line = bundle.get("line")
            weight = bundle.get("weight")
            if not is_two_vector(center) or not isinstance(line, (list, tuple)) or len(line) != 3:
                continue
            sx = scale_x if abs(scale_x) > 1e-6 else 1.0
            sy = scale_y if abs(scale_y) > 1e-6 else 1.0
            center_scaled = np.array(
                [float(center[0]) * sx, float(center[1]) * sy],
                dtype=float,
            )
            a = float(line[0])
            b = float(line[1])
            c = float(line[2])
            a_scaled = a / sx
            b_scaled = b / sy
            norm = float(np.hypot(a_scaled, b_scaled))
            if norm < 1e-8:
                continue
            coeffs = np.array([a_scaled / norm, b_scaled / norm, c / norm], dtype=float)
            segment = clip_line_to_bounds(coeffs, image_width, image_height)
            if segment is None:
                tangent = np.array([-coeffs[1], coeffs[0]], dtype=float)
                tang_norm = float(np.linalg.norm(tangent))
                if tang_norm < 1e-8:
                    continue
                tangent /= tang_norm
                half = 0.15 * min(image_width, image_height)
                p0 = center_scaled - tangent * half
                p1 = center_scaled + tangent * half
                p0[0] = np.clip(p0[0], 0.0, float(image_width))
                p0[1] = np.clip(p0[1], 0.0, float(image_height))
                p1[0] = np.clip(p1[0], 0.0, float(image_width))
                p1[1] = np.clip(p1[1], 0.0, float(image_height))
                segment = np.stack([p0, p1], axis=0)

            bundle_lines.append(segment)
            bundle_weights.append(float(weight) if weight is not None else 0.0)
            bundle_centers.append(center_scaled)
            count += 1

    return bundle_lines, bundle_weights, bundle_centers


def plot_vp_outlier_demo(
    image_path: Path,
    result_path: Path,
    alpha: float,
    limit: int | None,
    save_path: Path | None,
) -> None:
    image = Image.open(image_path).convert("L")
    width, height = image.size
    raw = load_result(result_path)
    report, trace = extract_detection_report(raw)
    grid = report.get("grid", {}) if isinstance(report.get("grid"), dict) else {}
    segments = trace.get("segments", []) if isinstance(trace.get("segments"), list) else []
    descriptors = {
        int(seg["id"]): seg
        for seg in segments
        if isinstance(seg, dict) and "id" in seg
    }
    outlier_stage = trace.get("outlierFilter", {}) if isinstance(trace.get("outlierFilter"), dict) else {}
    classifications = outlier_stage.get("classifications", [])
    if not isinstance(classifications, list):
        classifications = []

    inlier_lines, outlier_lines = gather_segments(descriptors, classifications, limit)

    bundling_stage = trace.get("bundling", {}) if isinstance(trace.get("bundling"), dict) else {}
    bundle_lines, bundle_weights, bundle_centers = gather_bundle_lines(
        bundling_stage,
        width,
        height,
        limit,
    )
    has_bundles = bool(bundle_lines)

    if has_bundles:
        fig, axes = plt.subplots(1, 2, figsize=figure_size(width * 2, height), dpi=120)
        ax_main, ax_bundle = axes
    else:
        fig, ax_main = plt.subplots(figsize=figure_size(width, height), dpi=120)
        ax_bundle = None

    ax_main.imshow(image, cmap="gray", origin="upper")
    ax_main.set_xlim(0, width)
    ax_main.set_ylim(height, 0)
    ax_main.set_axis_off()

    if inlier_lines:
        lc_in = LineCollection(inlier_lines, colors=INLIER_COLOR, linewidths=1.8, alpha=alpha)
        ax_main.add_collection(lc_in)
    if outlier_lines:
        lc_out = LineCollection(outlier_lines, colors=OUTLIER_COLOR, linewidths=1.4, alpha=alpha)
        ax_main.add_collection(lc_out)

    src = trace.get("input", {}) if isinstance(trace.get("input"), dict) else {}
    src_w = int(src.get("width", width))
    src_h = int(src.get("height", height))

    m_coarse = to_matrix3x3(trace.get("coarseHomography"))
    if m_coarse is not None:
        m_coarse = rescale_homography(m_coarse, src_w, src_h, width, height)
        draw_model(
            ax_main,
            m_coarse,
            width,
            height,
            MODEL_COLORS["coarse"],
            MODEL_STYLES["coarse"],
        )

    m_refined = to_matrix3x3(grid.get("hmtx"))
    if m_refined is not None:
        m_refined = rescale_homography(m_refined, src_w, src_h, width, height)
        draw_model(
            ax_main,
            m_refined,
            width,
            height,
            MODEL_COLORS["refined"],
            MODEL_STYLES["refined"],
        )

    handles: List[Line2D] = [
        Line2D([0], [0], color=INLIER_COLOR, linewidth=2.0, label="Inliers"),
        Line2D([0], [0], color=OUTLIER_COLOR, linewidth=2.0, label="Outliers"),
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS["coarse"]["u"],
                linestyle=MODEL_STYLES["coarse"],
                linewidth=2.0,
                label="Coarse VP\u2093",
            ),
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS["coarse"]["v"],
                linestyle=MODEL_STYLES["coarse"],
                linewidth=2.0,
                label="Coarse VP\u2094",
            ),
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS["refined"]["u"],
                linestyle=MODEL_STYLES["refined"],
                linewidth=2.2,
                label="Refined VP\u2093",
            ),
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS["refined"]["v"],
                linestyle=MODEL_STYLES["refined"],
                linewidth=2.2,
                label="Refined VP\u2094",
            ),
        ]
    )

    ax_main.legend(handles=handles, loc="upper right", frameon=True, facecolor="#ffffffdd")

    total = outlier_stage.get("total")
    kept = outlier_stage.get("kept")
    rejected = outlier_stage.get("rejected")
    timings = trace.get("timings", {}) if isinstance(trace.get("timings"), dict) else {}
    total_ms = timings.get("totalMs")
    lsd_stage = trace.get("lsd", {}) if isinstance(trace.get("lsd"), dict) else {}
    title_parts = ["VP Outlier Demo"]
    if total is not None and kept is not None and rejected is not None:
        title_parts.append(f"segments={total} (kept={kept}, rejected={rejected})")
    if (conf := lsd_stage.get("confidence")) is not None:
        title_parts.append(f"LSD conf={conf:.3f}")
    if isinstance(grid.get("confidence"), (int, float)):
        title_parts.append(f"refined conf={grid['confidence']:.3f}")
    if isinstance(total_ms, (int, float)):
        title_parts.append(f"total={total_ms:.1f} ms")
    ax_main.set_title(" | ".join(title_parts))

    if has_bundles and ax_bundle is not None:
        ax_bundle.imshow(image, cmap="gray", origin="upper")
        ax_bundle.set_xlim(0, width)
        ax_bundle.set_ylim(height, 0)
        ax_bundle.set_axis_off()

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
        ax_bundle.add_collection(lc_bundle)

        centers_arr = np.asarray(bundle_centers)
        if centers_arr.size:
            sizes = 20.0 + 40.0 * norm if norm.size else 30.0
            ax_bundle.scatter(
                centers_arr[:, 0],
                centers_arr[:, 1],
                s=sizes,
                c=colors if norm.size else [colors],
                edgecolors="white",
                linewidths=0.3,
                alpha=0.9,
            )

        bundle_title_parts = [f"Bundles ({len(bundle_lines)})"]
        orientation_tol = bundling_stage.get("orientationTolDeg")
        merge_dist = bundling_stage.get("mergeDistancePx")
        if isinstance(orientation_tol, (int, float)):
            bundle_title_parts.append(f"tol={orientation_tol:.1f}°")
        if isinstance(merge_dist, (int, float)):
            bundle_title_parts.append(f"merge={merge_dist:.1f}px")
        ax_bundle.set_title(" | ".join(bundle_title_parts))

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
