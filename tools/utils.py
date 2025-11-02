from __future__ import annotations

import numpy as np
from typing import Dict, Iterable, Tuple, List

# Model types for typed helpers
try:
    from tools.vp_outlier_result import Segment as ModelSegment, OutlierClassification as ModelOutlierClass, BundlingStage as ModelBundling
except Exception:
    try:
        from vp_outlier_result import Segment as ModelSegment, OutlierClassification as ModelOutlierClass, BundlingStage as ModelBundling
    except Exception:
        ModelSegment = None  # type: ignore
        ModelOutlierClass = None  # type: ignore
        ModelBundling = None  # type: ignore


def is_two_vector(vec) -> bool:
    return isinstance(vec, (list, tuple)) and len(vec) == 2

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
        if p0 is None or p1 is None:
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
            if not is_two_vector(p0) or not is_two_vector(p1) or p0 is None or p1 is None:
                continue
            line = np.array([[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]], dtype=float)
            inliers.append(line)
            count += 1
    return inliers, outliers


def gather_segments_from_model(
    segments: List[ModelSegment],
    classifications: List[ModelOutlierClass],
    limit: int | None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build inlier/outlier 2-point segments from model objects.

    Falls back to treating all segments as inliers when no classifications.
    """
    inliers: List[np.ndarray] = []
    outliers: List[np.ndarray] = []
    if classifications:
        desc = {int(s.id): s for s in segments}
        count = 0
        for c in classifications:
            if limit is not None and count >= limit:
                break
            s = desc.get(int(c.segment))
            if s is None:
                continue
            line = np.array([[float(s.p0[0]), float(s.p0[1])], [float(s.p1[0]), float(s.p1[1])]], dtype=float)
            (inliers if c.inlier else outliers).append(line)
            count += 1
    else:
        count = 0
        for s in segments:
            if limit is not None and count >= limit:
                break
            line = np.array([[float(s.p0[0]), float(s.p0[1])], [float(s.p1[0]), float(s.p1[1])]], dtype=float)
            inliers.append(line)
            count += 1
    return inliers, outliers

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
    if stage_scale is None:
        return [], [], []

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
            if not is_two_vector(center) or not isinstance(line, (list, tuple)) or len(line) != 3 or center is None:
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


def gather_bundle_lines_from_model(
    bundling: ModelBundling | None,
    image_width: int,
    image_height: int,
    limit: int | None,
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
    if bundling is None:
        return [], [], []
    # Stage scale (pyramid to full-image)
    stage_scale_x, stage_scale_y = bundling.scale_applied if bundling.scale_applied else (1.0, 1.0)
    stage_scale_x = float(stage_scale_x or 1.0)
    stage_scale_y = float(stage_scale_y or 1.0)

    bundle_lines: List[np.ndarray] = []
    bundle_weights: List[float] = []
    bundle_centers: List[np.ndarray] = []
    count = 0
    for level in bundling.levels:
        if limit is not None and count >= limit:
            break
        level_width = float(level.width or image_width)
        level_height = float(level.height or image_height)
        bundle_space_w = level_width * stage_scale_x
        bundle_space_h = level_height * stage_scale_y
        scale_x = float(image_width) / bundle_space_w if bundle_space_w > 1e-6 else 1.0
        scale_y = float(image_height) / bundle_space_h if bundle_space_h > 1e-6 else 1.0
        for b in level.bundles:
            if limit is not None and count >= limit:
                break
            center = np.array([float(b.center[0]) * scale_x, float(b.center[1]) * scale_y], dtype=float)
            a, bcoef, c = float(b.line[0]), float(b.line[1]), float(b.line[2])
            a_scaled = a / scale_x
            b_scaled = bcoef / scale_y
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
                p0 = center - tangent * half
                p1 = center + tangent * half
                p0[0] = np.clip(p0[0], 0.0, float(image_width))
                p0[1] = np.clip(p0[1], 0.0, float(image_height))
                p1[0] = np.clip(p1[0], 0.0, float(image_width))
                p1[1] = np.clip(p1[1], 0.0, float(image_height))
                segment = np.stack([p0, p1], axis=0)
            bundle_lines.append(segment)
            bundle_weights.append(float(b.weight))
            bundle_centers.append(center)
            count += 1
    return bundle_lines, bundle_weights, bundle_centers
