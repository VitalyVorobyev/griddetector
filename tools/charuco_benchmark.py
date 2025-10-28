#!/usr/bin/env python3
"""Charuco board detection performance benchmark using OpenCV (aruco).

Loads detection configuration from a JSON file, runs detection on either
provided image(s) or synthetically generated Charuco images, and reports
timings for marker detection, Charuco interpolation, optional pose
estimation, and total time. Outputs a summary to stdout and writes a
machine-readable JSON (and optional CSV) report to the configured output.

Usage:
  python tools/charuco_benchmark.py --config config/charuco_benchmark.json

Requirements:
  - Python 3.8+
  - OpenCV built with aruco module (pip: opencv-contrib-python)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import platform

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    print("Failed to import OpenCV. Please install opencv-contrib-python.", file=sys.stderr)
    raise

# Capability flags (handle partial aruco builds gracefully)
_aruco = getattr(cv2, "aruco", None)
CHARUCO_BOARD_AVAILABLE = bool(_aruco) and (hasattr(_aruco, "CharucoBoard") or hasattr(_aruco, "CharucoBoard_create"))
CHARUCO_INTERP_AVAILABLE = bool(_aruco) and hasattr(_aruco, "interpolateCornersCharuco")
CHARUCO_POSE_AVAILABLE = bool(_aruco) and hasattr(_aruco, "estimatePoseCharucoBoard")
_WARNED = {"interp": False, "pose": False}


# --------------------------- Utilities & Types ---------------------------


def monotonic_ns() -> int:
    return time.perf_counter_ns()


def ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0


def ensure_gray(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_path_list(input_glob: Optional[str], inputs: Optional[Sequence[str]]) -> List[Path]:
    paths: List[Path] = []
    if inputs:
        for p in inputs:
            paths.append(Path(p))
    if input_glob:
        for p in sorted(Path().glob(input_glob)):
            paths.append(p)
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for p in paths:
        q = p.resolve()
        if q in seen:
            continue
        seen.add(q)
        unique.append(p)
    return unique


def resolve_dictionary(name: str):
    # Map string to cv2.aruco predefined dictionary
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco module is not available. Install opencv-contrib-python.")
    aruco = cv2.aruco
    # Common dictionaries
    mapping = {
        "DICT_4X4_50": getattr(aruco, "DICT_4X4_50", None),
        "DICT_4X4_100": getattr(aruco, "DICT_4X4_100", None),
        "DICT_4X4_250": getattr(aruco, "DICT_4X4_250", None),
        "DICT_4X4_1000": getattr(aruco, "DICT_4X4_1000", None),
        "DICT_5X5_50": getattr(aruco, "DICT_5X5_50", None),
        "DICT_5X5_100": getattr(aruco, "DICT_5X5_100", None),
        "DICT_5X5_250": getattr(aruco, "DICT_5X5_250", None),
        "DICT_5X5_1000": getattr(aruco, "DICT_5X5_1000", None),
        "DICT_6X6_50": getattr(aruco, "DICT_6X6_50", None),
        "DICT_6X6_100": getattr(aruco, "DICT_6X6_100", None),
        "DICT_6X6_250": getattr(aruco, "DICT_6X6_250", None),
        "DICT_6X6_1000": getattr(aruco, "DICT_6X6_1000", None),
        "DICT_7X7_50": getattr(aruco, "DICT_7X7_50", None),
        "DICT_7X7_100": getattr(aruco, "DICT_7X7_100", None),
        "DICT_7X7_250": getattr(aruco, "DICT_7X7_250", None),
        "DICT_7X7_1000": getattr(aruco, "DICT_7X7_1000", None),
        "DICT_ARUCO_ORIGINAL": getattr(aruco, "DICT_ARUCO_ORIGINAL", None),
    }
    if name not in mapping or mapping[name] is None:
        raise ValueError(f"Unknown/unsupported aruco dictionary: {name}")
    return aruco.getPredefinedDictionary(mapping[name])


def make_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
    dictionary,
):
    aruco = cv2.aruco
    # Handle OpenCV API differences
    if hasattr(aruco, "CharucoBoard"):
        return aruco.CharucoBoard((int(squares_x), int(squares_y)), float(square_length), float(marker_length), dictionary)
    if hasattr(aruco, "CharucoBoard_create"):
        return aruco.CharucoBoard_create(int(squares_x), int(squares_y), float(square_length), float(marker_length), dictionary)
    raise RuntimeError("Your OpenCV build lacks CharucoBoard support.")


def create_detector_params(cfg: Dict[str, Any]):
    aruco = cv2.aruco
    # Newer OpenCV: class-like constructor; older: *_create
    params = getattr(aruco, "DetectorParameters")( ) if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()
    # Assign known fields if present in config
    for key, value in cfg.items():
        if not hasattr(params, key):
            continue
        try:
            setattr(params, key, value)
        except Exception:
            # Ignore invalid assignments gracefully
            pass
    return params


def create_aruco_detector(dictionary, params):
    aruco = cv2.aruco
    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(dictionary, params)
    return None  # legacy path uses detectMarkers directly


def detect_markers(gray, dictionary, params, detector=None):
    aruco = cv2.aruco
    if detector is not None:
        return detector.detectMarkers(gray)
    return aruco.detectMarkers(gray, dictionary, parameters=params)


def refine_markers(
    gray,
    board,
    corners,
    ids,
    rejected,
    camera_matrix=None,
    dist_coeffs=None,
    params=None,
):
    aruco = cv2.aruco
    if board is None:
        return corners, ids, rejected
    if hasattr(aruco, "refineDetectedMarkers"):
        try:
            corners, ids, rejected, _ = aruco.refineDetectedMarkers(
                image=gray,
                board=board,
                detectedCorners=corners,
                detectedIds=ids,
                rejectedCorners=rejected,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
                parameters=params,
            )
        except TypeError:
            # Older versions with positional signature
            corners, ids, rejected, _ = aruco.refineDetectedMarkers(
                gray, board, corners, ids, rejected, camera_matrix, dist_coeffs, params
            )
    return corners, ids, rejected


@dataclass
class Timing:
    detect_ns: int = 0
    refine_ns: int = 0
    charuco_ns: int = 0
    pose_ns: int = 0

    @property
    def total_ns(self) -> int:
        return self.detect_ns + self.refine_ns + self.charuco_ns + self.pose_ns


def interpolate_charuco(gray, board, corners, ids, camera_matrix=None, dist_coeffs=None):
    aruco = cv2.aruco
    if not CHARUCO_INTERP_AVAILABLE:
        if not _WARNED["interp"]:
            print("Warning: cv2.aruco lacks interpolateCornersCharuco; skipping Charuco interpolation.", file=sys.stderr)
            _WARNED["interp"] = True
        return None
    if not corners or ids is None or len(corners) == 0:
        return None
    try:
        ret = aruco.interpolateCornersCharuco(corners, ids, gray, board, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)
    except TypeError:
        # Older API without kwargs
        ret = aruco.interpolateCornersCharuco(corners, ids, gray, board, camera_matrix, dist_coeffs)
    return ret


def estimate_pose(board, charuco_corners, charuco_ids, camera_matrix, dist_coeffs):
    aruco = cv2.aruco
    if camera_matrix is None or dist_coeffs is None:
        return None
    if not CHARUCO_POSE_AVAILABLE:
        if not _WARNED["pose"]:
            print("Warning: cv2.aruco lacks estimatePoseCharucoBoard; skipping pose estimation.", file=sys.stderr)
            _WARNED["pose"] = True
        return None
    try:
        ok, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs)
    except TypeError:
        ok, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None)
    return (ok, rvec, tvec)


def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


def draw_synthetic(board, width: int, height: int, dpi: int = 300) -> Any:
    # Charuco board renders in pixels proportional to square_length/marker_length ratio.
    # We'll draw at requested resolution and then place it on a white canvas.
    if board is None:
        raise RuntimeError("Cannot draw synthetic Charuco image: CharucoBoard is unavailable in this OpenCV build.")
    img = board.draw((int(width), int(height)))
    return img


def robust_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": float(np_percentile(values, 90.0)),
        "p95": float(np_percentile(values, 95.0)),
    }


def np_percentile(values: List[float], p: float) -> float:
    # Small dependency-free percentile implementation
    vals = sorted(values)
    if not vals:
        return float("nan")
    if p <= 0:
        return vals[0]
    if p >= 100:
        return vals[-1]
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return d0 + d1


# --------------------------- Benchmark Logic -----------------------------


def run_detection_once(
    gray,
    dictionary,
    board,
    params,
    detector,
    do_refine: bool,
    camera_matrix,
    dist_coeffs,
    min_charuco_markers: int,
    do_pose: bool,
) -> Timing:
    t0 = monotonic_ns()
    corners, ids, rejected = detect_markers(gray, dictionary, params, detector)
    t1 = monotonic_ns()
    refine_ns = 0
    if do_refine:
        cr0 = monotonic_ns()
        corners, ids, rejected = refine_markers(gray, board, corners, ids, rejected, camera_matrix, dist_coeffs, params)
        cr1 = monotonic_ns()
        refine_ns = cr1 - cr0

    if CHARUCO_INTERP_AVAILABLE:
        ci0 = monotonic_ns()
        res = interpolate_charuco(gray, board, corners, ids, camera_matrix, dist_coeffs)
        ci1 = monotonic_ns()
        charuco_ns = ci1 - ci0
    else:
        res = None
        charuco_ns = 0

    pose_ns = 0
    if res is not None:
        charuco_corners, charuco_ids = res
        if (
            charuco_corners is not None and charuco_ids is not None and
            len(charuco_ids) >= max(1, int(min_charuco_markers))
        ):
            if do_pose and camera_matrix is not None and dist_coeffs is not None and CHARUCO_POSE_AVAILABLE:
                ep0 = monotonic_ns()
                _ = estimate_pose(board, charuco_corners, charuco_ids, camera_matrix, dist_coeffs)
                ep1 = monotonic_ns()
                pose_ns = ep1 - ep0

    t2 = monotonic_ns()

    return Timing(
        detect_ns=(t1 - t0),
        refine_ns=refine_ns,
        charuco_ns=charuco_ns,
        pose_ns=pose_ns,
    )


def parse_camera_matrix(cfg: Dict[str, Any] | None):
    if not cfg:
        return None, None
    K = cfg.get("matrix")
    D = cfg.get("dist") or cfg.get("distCoeffs")
    K_mat = None
    D_vec = None
    try:
        import numpy as np  # type: ignore
        if K is not None:
            K_mat = np.array(K, dtype=float)
            if K_mat.shape == (3, 3) and K_mat.ndim == 2:
                pass
            elif K_mat.size == 9:
                K_mat = K_mat.reshape((3, 3))
            else:
                K_mat = None
        if D is not None:
            D_vec = np.array(D, dtype=float).reshape((-1, 1))
    except Exception:
        K_mat = None
        D_vec = None
    return K_mat, D_vec


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cols = [
        "image",
        "run",
        "detect_ms",
        "refine_ms",
        "charuco_ms",
        "pose_ms",
        "total_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def generate_synthetic_images(board, count: int, width: int, height: int) -> List[Any]:
    images: List[Any] = []
    for _ in range(int(count)):
        img = draw_synthetic(board, width, height)
        images.append(img)
    return images


def benchmark_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    aruco = cv2.aruco
    # ---------------------- Configuration ----------------------
    input_cfg = cfg.get("input", {}) if isinstance(cfg.get("input"), dict) else {}
    input_glob = input_cfg.get("glob")
    inputs = input_cfg.get("paths")
    synthetic = cfg.get("synthetic", {}) if False and isinstance(cfg.get("synthetic"), dict) else {}

    board_cfg = cfg.get("board", {})
    squares_x = int(board_cfg.get("squares_x", 7))
    squares_y = int(board_cfg.get("squares_y", 5))
    square_length = float(board_cfg.get("square_length", 0.04))
    marker_length = float(board_cfg.get("marker_length", 0.02))
    dict_name = str(board_cfg.get("dictionary", "DICT_4X4_50"))
    dictionary = resolve_dictionary(dict_name)
    if not CHARUCO_BOARD_AVAILABLE:
        print("Warning: CharucoBoard API not available in your OpenCV build.", file=sys.stderr)
        board = None
    else:
        board = make_charuco_board(squares_x, squares_y, square_length, marker_length, dictionary)

    detector_cfg = cfg.get("detector", {}) if isinstance(cfg.get("detector"), dict) else {}
    params_cfg = detector_cfg.get("parameters", {}) if isinstance(detector_cfg.get("parameters"), dict) else {}
    refine_enabled = bool(detector_cfg.get("refine", {}).get("enabled", True))
    params = create_detector_params(params_cfg)
    detector = create_aruco_detector(dictionary, params)

    charuco_cfg = cfg.get("charuco", {}) if isinstance(cfg.get("charuco"), dict) else {}
    min_charuco_markers = int(charuco_cfg.get("min_markers", 4))

    camera_cfg = cfg.get("camera", {}) if isinstance(cfg.get("camera"), dict) else {}
    camera_matrix, dist_coeffs = parse_camera_matrix(camera_cfg)
    do_pose = bool(camera_cfg.get("estimate_pose", False)) and camera_matrix is not None and CHARUCO_POSE_AVAILABLE
    if bool(camera_cfg.get("estimate_pose", False)) and not CHARUCO_POSE_AVAILABLE:
        if not _WARNED["pose"]:
            print("Warning: Pose timing requested but estimatePoseCharucoBoard is unavailable.", file=sys.stderr)
            _WARNED["pose"] = True

    timing_cfg = cfg.get("timing", {}) if isinstance(cfg.get("timing"), dict) else {}
    warmup_runs = int(timing_cfg.get("warmup_runs", 2))
    runs = int(timing_cfg.get("runs", 10))

    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), dict) else {}
    out_dir = Path(out_cfg.get("dir", "out/charuco_benchmark"))
    out_json = out_dir / str(out_cfg.get("result_json", "results.json"))
    out_csv = out_cfg.get("csv")
    out_csv_path = out_dir / str(out_csv) if out_csv else None

    # ------------------------ Prepare inputs ------------------------
    image_paths: List[Path] = to_path_list(input_glob, inputs)
    synthetic_images: List[Any] = []
    if not image_paths:
        # Fallback to synthetic images
        width = int(synthetic.get("width", 1280))
        height = int(synthetic.get("height", 720))
        count = int(synthetic.get("count", 5))
        if board is None:
            print("No input images provided and CharucoBoard is unavailable; cannot generate synthetic set.", file=sys.stderr)
        else:
            synthetic_images = generate_synthetic_images(board, count, width, height)

    # -------------------------- Benchmark --------------------------
    per_image_results: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    def run_on_gray(name: str, gray) -> Dict[str, Any]:
        # Warmup
        for _ in range(max(0, warmup_runs)):
            _ = run_detection_once(gray, dictionary, board, params, detector, refine_enabled, camera_matrix, dist_coeffs, min_charuco_markers, do_pose)

        timings: List[Timing] = []
        for r in range(max(1, runs)):
            t = run_detection_once(gray, dictionary, board, params, detector, refine_enabled, camera_matrix, dist_coeffs, min_charuco_markers, do_pose)
            timings.append(t)
            csv_rows.append(
                {
                    "image": name,
                    "run": r,
                    "detect_ms": f"{ns_to_ms(t.detect_ns):.3f}",
                    "refine_ms": f"{ns_to_ms(t.refine_ns):.3f}",
                    "charuco_ms": f"{ns_to_ms(t.charuco_ns):.3f}",
                    "pose_ms": f"{ns_to_ms(t.pose_ns):.3f}",
                    "total_ms": f"{ns_to_ms(t.total_ns):.3f}",
                }
            )

        detect_ms = [ns_to_ms(t.detect_ns) for t in timings]
        refine_ms = [ns_to_ms(t.refine_ns) for t in timings if t.refine_ns > 0]
        charuco_ms = [ns_to_ms(t.charuco_ns) for t in timings]
        pose_ms = [ns_to_ms(t.pose_ns) for t in timings if t.pose_ns > 0]
        total_ms = [ns_to_ms(t.total_ns) for t in timings]

        return {
            "image": name,
            "stats": {
                "detect_ms": summarize(detect_ms),
                "refine_ms": summarize(refine_ms),
                "charuco_ms": summarize(charuco_ms),
                "pose_ms": summarize(pose_ms),
                "total_ms": summarize(total_ms),
            },
        }

    if image_paths:
        for path in image_paths:
            gray = load_image(path)
            res = run_on_gray(str(path), gray)
            per_image_results.append(res)
    else:
        for i, gray in enumerate(synthetic_images):
            name = f"synthetic_{i:03d}"
            res = run_on_gray(name, gray)
            per_image_results.append(res)

    # Aggregate totals
    agg_detect: List[float] = []
    agg_refine: List[float] = []
    agg_charuco: List[float] = []
    agg_pose: List[float] = []
    agg_total: List[float] = []

    for item in per_image_results:
        s = item.get("stats", {})
        for _ in range(int(s.get("detect_ms", {}).get("count", 0))):
            pass
        # Each per-image already aggregates per-run. For overall aggregation, collect all raw rows.
    for row in csv_rows:
        agg_detect.append(float(row["detect_ms"]))
        if row.get("refine_ms"):
            agg_refine.append(float(row["refine_ms"]))
        agg_charuco.append(float(row["charuco_ms"]))
        if row.get("pose_ms"):
            agg_pose.append(float(row["pose_ms"]))
        agg_total.append(float(row["total_ms"]))

    overall = {
        "detect_ms": summarize(agg_detect),
        "refine_ms": summarize(agg_refine),
        "charuco_ms": summarize(agg_charuco),
        "pose_ms": summarize(agg_pose),
        "total_ms": summarize(agg_total),
    }

    env = {
        "python": sys.version.split(" (", 1)[0],
        "opencv": getattr(cv2, "__version__", "unknown"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "release": platform.release(),
    }

    report = {
        "config": cfg,
        "environment": env,
        "overall": overall,
        "images": per_image_results,
    }

    # Persist outputs
    save_json(out_json, report)
    if out_csv_path is not None:
        save_csv(out_csv_path, csv_rows)

    # Human-readable summary
    print("Charuco detection benchmark summary")
    print(f"- images: {len(per_image_results)}")
    print(f"- runs per image: {runs} (warmup {warmup_runs})")
    print(f"- cv2: {env['opencv']}")
    tm = overall.get("total_ms", {})
    parts = [
        f"mean={tm.get('mean', float('nan')):.3f} ms",
        f"median={tm.get('median', float('nan')):.3f} ms",
        f"p90={tm.get('p90', float('nan')):.3f} ms",
        f"p95={tm.get('p95', float('nan')):.3f} ms",
        f"min={tm.get('min', float('nan')):.3f} ms",
        f"max={tm.get('max', float('nan')):.3f} ms",
    ]
    print("- total: " + ", ".join(parts))
    print(f"Wrote report: {out_json}")
    if out_csv_path is not None:
        print(f"Wrote CSV: {out_csv_path}")

    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Charuco detection with OpenCV aruco")
    p.add_argument("--config", type=Path, required=True, help="Path to config JSON")
    return p.parse_args()


def main() -> None:
    if not hasattr(cv2, "aruco"):
        print("OpenCV build missing aruco module. Install opencv-contrib-python.", file=sys.stderr)
        sys.exit(2)
    args = parse_args()
    cfg = load_config(args.config)
    benchmark_from_config(cfg)


if __name__ == "__main__":
    main()
