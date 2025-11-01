from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import json
import numpy as np


def _to_vec2(v) -> Optional[np.ndarray]:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            return np.array([float(v[0]), float(v[1])], dtype=float)
        except Exception:
            return None
    return None


def _to_vec3(v) -> Optional[np.ndarray]:
    if isinstance(v, (list, tuple)) and len(v) == 3:
        try:
            return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float)
        except Exception:
            return None
    return None


def _to_mat3(m) -> Optional[np.ndarray]:
    try:
        arr = np.array(m, dtype=float)
    except Exception:
        return None
    if arr.shape == (3, 3):
        return arr
    if arr.ndim == 1 and arr.size == 9:
        # nalgebra Matrix3 serde flattens column-major
        return arr.reshape((3, 3), order="F")
    return None


@dataclass
class Pose:
    r: np.ndarray
    t: np.ndarray

    @staticmethod
    def from_json(d: dict) -> Optional["Pose"]:
        if not isinstance(d, dict):
            return None
        r = _to_mat3(d.get("r"))
        t = d.get("t")
        if isinstance(t, (list, tuple)) and len(t) == 3:
            try:
                t = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)
            except Exception:
                t = None
        else:
            t = None
        if r is None or t is None:
            return None
        return Pose(r=r, t=t)


@dataclass
class Grid:
    found: bool
    hmtx: np.ndarray
    pose: Optional[Pose]
    origin_uv: Tuple[int, int] = (0, 0)
    visible_range: Tuple[int, int, int, int] = (0, 0, 0, 0)
    coverage: float = 0.0
    reproj_rmse: float = 0.0
    confidence: float = 0.0
    latency_ms: float = 0.0

    @staticmethod
    def from_json(d: dict) -> "Grid":
        h = _to_mat3(d.get("hmtx"))
        if h is None:
            h = np.eye(3, dtype=float)
        pose = Pose.from_json(d.get("pose")) if isinstance(d.get("pose"), dict) else None
        origin = d.get("origin_uv") or d.get("originUv") or (0, 0)
        if isinstance(origin, (list, tuple)) and len(origin) == 2:
            origin_uv = (int(origin[0]), int(origin[1]))
        else:
            origin_uv = (0, 0)
        vr = d.get("visible_range") or d.get("visibleRange") or (0, 0, 0, 0)
        if isinstance(vr, (list, tuple)) and len(vr) == 4:
            visible_range = (int(vr[0]), int(vr[1]), int(vr[2]), int(vr[3]))
        else:
            visible_range = (0, 0, 0, 0)
        return Grid(
            found=bool(d.get("found", False)),
            hmtx=h,
            pose=pose,
            origin_uv=origin_uv,
            visible_range=visible_range,
            coverage=float(d.get("coverage", 0.0) or 0.0),
            reproj_rmse=float(d.get("reproj_rmse", d.get("reprojRmse", 0.0)) or 0.0),
            confidence=float(d.get("confidence", 0.0) or 0.0),
            latency_ms=float(d.get("latency_ms", d.get("latencyMs", 0.0)) or 0.0),
        )


@dataclass
class Segment:
    id: int
    p0: np.ndarray
    p1: np.ndarray
    direction: Optional[np.ndarray] = None
    length: Optional[float] = None
    line: Optional[np.ndarray] = None
    average_magnitude: Optional[float] = None
    strength: Optional[float] = None
    family: Optional[str] = None

    @staticmethod
    def from_json(d: dict) -> Optional["Segment"]:
        if not isinstance(d, dict):
            return None
        if "id" not in d:
            return None
        p0 = _to_vec2(d.get("p0"))
        p1 = _to_vec2(d.get("p1"))
        if p0 is None or p1 is None:
            return None
        return Segment(
            id=int(d.get("id")),
            p0=p0,
            p1=p1,
            direction=_to_vec2(d.get("direction")),
            length=float(d.get("length", 0.0)) if d.get("length") is not None else None,
            line=_to_vec3(d.get("line")),
            average_magnitude=float(d.get("averageMagnitude", 0.0)) if d.get("averageMagnitude") is not None else None,
            strength=float(d.get("strength", 0.0)) if d.get("strength") is not None else None,
            family=d.get("family"),
        )


@dataclass
class OutlierClassification:
    segment: int
    label: str  # "kept" or "rejected"

    @property
    def inlier(self) -> bool:
        return self.label == "kept"

    @staticmethod
    def from_json(d: dict) -> Optional["OutlierClassification"]:
        if not isinstance(d, dict):
            return None
        seg = d.get("segment")
        lab = d.get("class")
        if seg is None or lab is None:
            return None
        return OutlierClassification(segment=int(seg), label=str(lab))


@dataclass
class OutlierFilterStage:
    elapsed_ms: float = 0.0
    total: int = 0
    kept: int = 0
    rejected: int = 0
    kept_u: int = 0
    kept_v: int = 0
    degenerate_segments: int = 0
    classifications: List[OutlierClassification] = field(default_factory=list)

    @staticmethod
    def from_json(d: dict) -> "OutlierFilterStage":
        if not isinstance(d, dict):
            return OutlierFilterStage()
        cl_raw = d.get("classifications", [])
        cls: List[OutlierClassification] = []
        if isinstance(cl_raw, list):
            for item in cl_raw:
                c = OutlierClassification.from_json(item)
                if c is not None:
                    cls.append(c)
        return OutlierFilterStage(
            elapsed_ms=float(d.get("elapsedMs", 0.0) or 0.0),
            total=int(d.get("total", 0) or 0),
            kept=int(d.get("kept", 0) or 0),
            rejected=int(d.get("rejected", 0) or 0),
            kept_u=int(d.get("kept_u", d.get("keptU", 0)) or 0),
            kept_v=int(d.get("kept_v", d.get("keptV", 0)) or 0),
            degenerate_segments=int(d.get("degenerate_segments", d.get("degenerateSegments", 0)) or 0),
            classifications=cls,
        )


@dataclass
class Bundle:
    center: np.ndarray
    line: np.ndarray
    weight: float

    @staticmethod
    def from_json(d: dict) -> Optional["Bundle"]:
        if not isinstance(d, dict):
            return None
        c = _to_vec2(d.get("center"))
        l = _to_vec3(d.get("line"))
        w = d.get("weight")
        if c is None or l is None or w is None:
            return None
        try:
            w = float(w)
        except Exception:
            return None
        return Bundle(center=c, line=l, weight=w)


@dataclass
class BundlingLevel:
    level_index: int
    width: int
    height: int
    bundles: List[Bundle]

    @staticmethod
    def from_json(d: dict) -> Optional["BundlingLevel"]:
        if not isinstance(d, dict):
            return None
        bundles: List[Bundle] = []
        for b in d.get("bundles", []) or []:
            bb = Bundle.from_json(b)
            if bb is not None:
                bundles.append(bb)
        return BundlingLevel(
            level_index=int(d.get("levelIndex", d.get("level", 0)) or 0),
            width=int(d.get("width", 0) or 0),
            height=int(d.get("height", 0) or 0),
            bundles=bundles,
        )


@dataclass
class BundlingStage:
    elapsed_ms: float = 0.0
    orientation_tol_deg: float = 0.0
    merge_distance_px: float = 0.0
    min_weight: float = 0.0
    source_segments: int = 0
    scale_applied: Tuple[float, float] = (1.0, 1.0)
    levels: List[BundlingLevel] = field(default_factory=list)

    @staticmethod
    def from_json(d: dict) -> "BundlingStage":
        if not isinstance(d, dict):
            return BundlingStage()
        sc = d.get("scaleApplied")
        if isinstance(sc, (list, tuple)) and len(sc) == 2:
            try:
                scale_applied = (float(sc[0]), float(sc[1]))
            except Exception:
                scale_applied = (1.0, 1.0)
        else:
            scale_applied = (1.0, 1.0)
        levels: List[BundlingLevel] = []
        for lvl in d.get("levels", []) or []:
            ll = BundlingLevel.from_json(lvl)
            if ll is not None:
                levels.append(ll)
        return BundlingStage(
            elapsed_ms=float(d.get("elapsedMs", 0.0) or 0.0),
            orientation_tol_deg=float(d.get("orientationTolDeg", 0.0) or 0.0),
            merge_distance_px=float(d.get("mergeDistancePx", 0.0) or 0.0),
            min_weight=float(d.get("minWeight", 0.0) or 0.0),
            source_segments=int(d.get("sourceSegments", 0) or 0),
            scale_applied=scale_applied,
            levels=levels,
        )


@dataclass
class InputInfo:
    width: int
    height: int
    pyramid_levels: int

    @staticmethod
    def from_json(d: dict) -> "InputInfo":
        return InputInfo(
            width=int(d.get("width", 0) or 0),
            height=int(d.get("height", 0) or 0),
            pyramid_levels=int(d.get("pyramidLevels", d.get("pyramid_levels", 0)) or 0),
        )


@dataclass
class Timings:
    total_ms: float

    @staticmethod
    def from_json(d: dict) -> "Timings":
        return Timings(total_ms=float(d.get("totalMs", 0.0) or 0.0))


@dataclass
class Trace:
    input: InputInfo
    timings: Timings
    segments: List[Segment]
    outlier_filter: OutlierFilterStage
    bundling: BundlingStage
    coarse_homography: Optional[np.ndarray]

    @staticmethod
    def from_json(d: dict) -> "Trace":
        inp = InputInfo.from_json(d.get("input", {}) if isinstance(d.get("input"), dict) else {})
        tim = Timings.from_json(d.get("timings", {}) if isinstance(d.get("timings"), dict) else {})
        segs: List[Segment] = []
        for s in d.get("segments", []) or []:
            ss = Segment.from_json(s)
            if ss is not None:
                segs.append(ss)
        outlier = OutlierFilterStage.from_json(d.get("outlierFilter", {}) if isinstance(d.get("outlierFilter"), dict) else {})
        bundling = BundlingStage.from_json(d.get("bundling", {}) if isinstance(d.get("bundling"), dict) else {})
        Hc = _to_mat3(d.get("coarseHomography"))
        return Trace(input=inp, timings=tim, segments=segs, outlier_filter=outlier, bundling=bundling, coarse_homography=Hc)


@dataclass
class VpOutlierReport:
    grid: Grid
    trace: Trace

    @staticmethod
    def from_json(data: dict) -> "VpOutlierReport":
        # Allow either {"report": {...}} or flat JSON
        root = data.get("report") if isinstance(data.get("report"), dict) else data
        grid = Grid.from_json(root.get("grid", {}) if isinstance(root.get("grid"), dict) else {})
        trace = Trace.from_json(root.get("trace", {}) if isinstance(root.get("trace"), dict) else {})
        return VpOutlierReport(grid=grid, trace=trace)

    @staticmethod
    def from_path(path: Path) -> "VpOutlierReport":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return VpOutlierReport.from_json(data)
