"""Common plotting and homography helpers shared by tools.

This module centralizes small geometry, I/O, and drawing helpers to keep
individual plotting scripts concise and consistent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import ctypes
import json
import os
import sys
from ctypes import POINTER, c_float, c_int32, c_size_t, c_uint32

import numpy as np


def _default_library_candidates() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    target = root / "target"
    libnames = []
    if sys.platform.startswith("win"):
        libnames.extend(["grid_detector.dll", "libgrid_detector.dll"])
    elif sys.platform == "darwin":
        libnames.append("libgrid_detector.dylib")
    else:
        libnames.append("libgrid_detector.so")
    candidates: list[Path] = []
    for profile in ("release", "debug"):
        for name in libnames:
            candidates.append(target / profile / name)
    return candidates


def _load_grid_library() -> ctypes.CDLL | None:
    explicit = os.environ.get("GRID_DETECTOR_LIB")
    if explicit:
        try:
            return ctypes.CDLL(explicit)
        except OSError:
            return None
    for candidate in _default_library_candidates():
        if candidate.exists():
            try:
                return ctypes.CDLL(str(candidate))
            except OSError:
                continue
    return None


_GRID_LIB = _load_grid_library()
if _GRID_LIB is not None:
    _GRID_LIB.grid_rescale_homography.argtypes = [
        POINTER(c_float),
        c_uint32,
        c_uint32,
        c_uint32,
        c_uint32,
        POINTER(c_float),
    ]
    _GRID_LIB.grid_rescale_homography.restype = ctypes.c_bool
    _GRID_LIB.grid_apply_homography_points.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        c_size_t,
        POINTER(c_float),
    ]
    _GRID_LIB.grid_apply_homography_points.restype = c_int32


# -------------------- I/O --------------------

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_detection_report(data: dict) -> tuple[dict, dict]:
    """Return (report, trace) for both demo and detector JSON layouts.

    - Some demos wrap results under {"report": {...}}; others are flat.
    - We always return "trace" as a dict (possibly empty).
    """
    root = data.get("report") if isinstance(data.get("report"), dict) else data
    trace = root.get("trace", {}) if isinstance(root.get("trace"), dict) else {}
    return root, trace


# -------------------- Homography utils --------------------

def to_matrix3x3(matrix) -> np.ndarray | None:
    """Convert several JSON encodings to a 3x3 float ndarray.

    nalgebra's Matrix3 serde flattens in column-major order. We accept:
    - flat length-9 arrays (column-major)
    - explicit 3x3 arrays
    """
    try:
        arr = np.array(matrix, dtype=float)
    except Exception:
        return None
    if arr.shape == (3, 3):
        return arr
    if arr.ndim == 1 and arr.size == 9:
        return arr.reshape((3, 3), order="F")
    return None


def _rescale_homography_python(
    H: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int
) -> np.ndarray:
    if not (src_w and src_h and dst_w and dst_h):
        return H.astype(np.float32, copy=True)
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return S @ H


def rescale_homography_image_space(
    H: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int
) -> np.ndarray:
    """Rescale world->image homography from (src_w,src_h) to (dst_w,dst_h).

    Points transform as x_img' = S x_img, so H' = S H.
    """
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return S @ H


def apply_homography_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
    """Apply H to Nx2 points; return Nx2, or None if any maps to infinity.

    Accepts 2x2 (segment endpoints) as well.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    ph = np.concatenate([pts, ones], axis=1)
    q = (H @ ph.T).T
    return q[:, 0:2] / q[:, 2:3]

def apply_homography_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
    """Apply H to Nx2 points; return Nx2, or None if any maps to infinity."""
    if H is None or pts is None:
        return None
    H = np.asarray(H, dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None
    if _GRID_LIB is None:
        return _apply_homography_points_python(H, pts)

    out = np.empty_like(pts, dtype=np.float32)
    result = _GRID_LIB.grid_apply_homography_points(
        H.ctypes.data_as(POINTER(c_float)),
        pts.ctypes.data_as(POINTER(c_float)),
        c_size_t(pts.shape[0]),
        out.ctypes.data_as(POINTER(c_float)),
    )
    if result < 0:
        return None
    return out


def homogeneous_to_point(vec: np.ndarray) -> Tuple[np.ndarray, bool]:
    if abs(vec[2]) > 1e-6:
        return vec[:2] / vec[2], False
    return vec[:2], True


def compute_vp_ray(
    anchor_xy: np.ndarray,
    vp_h: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Return a 2x2 segment from anchor towards the VP, clipped to image bounds."""
    point, at_inf = homogeneous_to_point(vp_h)
    direction = (
        point / (np.linalg.norm(point) + 1e-9)
        if at_inf
        else (point - anchor_xy)
    )
    if np.linalg.norm(direction) < 1e-6:
        return None

    # Clip anchor + t*direction to the [0,w]x[0,h] rectangle for t>0
    dx, dy = float(direction[0]), float(direction[1])
    x0, y0 = float(anchor_xy[0]), float(anchor_xy[1])
    cand: list[tuple[float, float, float]] = []
    if abs(dx) > 1e-6:
        for x in (0.0, float(width)):
            t = (x - x0) / dx
            if t > 0:
                y = y0 + t * dy
                if 0.0 <= y <= float(height):
                    cand.append((t, x, y))
    if abs(dy) > 1e-6:
        for y in (0.0, float(height)):
            t = (y - y0) / dy
            if t > 0:
                x = x0 + t * dx
                if 0.0 <= x <= float(width):
                    cand.append((t, x, y))
    if not cand:
        return None
    _, x, y = min(cand, key=lambda it: it[0])
    return np.array([[x0, y0], [x, y]], dtype=float)


def extract_model_columns(H: np.ndarray) -> dict:
    """Return dict with homogeneous columns {"u","v","anchor"} from H.

    Expects columns to be (u, v, anchor) as produced by the Rust code.
    """
    return {"u": H[:, 0], "v": H[:, 1], "anchor": H[:, 2]}


def figure_size(width: int, height: int, scale: float = 1.5) -> Tuple[float, float]:
    base = 110.0
    return (width / (base / scale), height / (base / scale))

