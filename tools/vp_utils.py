import numpy as np

from vp_outlier_result import Segment

def estimate_family_vp(lines: np.ndarray) -> np.ndarray:
    """ Solve for v=(x,y) minimizing Σ w (a x + b y + c)^2 where each line is ax+by+c=0 """
    a11 = np.sum(lines[:, 0]**2)
    a12 = np.sum(lines[:, 0] * lines[:, 1])
    a22 = np.sum(lines[:, 1]**2)
    b1 = -np.sum(lines[:, 0] * lines[:, 2])
    b2 = -np.sum(lines[:, 1] * lines[:, 2])
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-8:
        return np.array([np.nan, np.nan])
    x = (a22 * b1 - a12 * b2) / det
    y = (a11 * b2 - a12 * b1) / det
    return np.array([x, y, 1])

def line_through_points(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """ Return line coefficients (a,b,c) for line through points p0 and p1 """
    a = p1[1] - p0[1]
    b = p0[0] - p1[0]
    c = p1[0] * p0[1] - p0[0] * p1[1]
    norm = np.sqrt(a * a + b * b)
    if norm > 1e-8:
        a /= norm
        b /= norm
        c /= norm
    return np.array([a, b, c])

def estimate_grid_vps(segments: list[Segment], segment_families: list[str]) -> np.ndarray:
    # Placeholder implementation
    print(segments[0])
    u_segments = np.vstack([line_through_points(s.p0, s.p1) for s, fam in zip(segments, segment_families) if fam == 'u'])
    v_segments = np.vstack([line_through_points(s.p0, s.p1) for s, fam in zip(segments, segment_families) if fam == 'v'])
    print(f"Estimating VP for {len(u_segments)} 'u' segments and {len(v_segments)} 'v' segments")
    print(f"U segments: {u_segments.shape}")
    print(f"V segments: {v_segments.shape}")
    assert u_segments.ndim == 2 and u_segments.shape[1] == 3
    assert v_segments.ndim == 2 and v_segments.shape[1] == 3

    vpu = estimate_family_vp(u_segments) if len(u_segments) > 0 else np.array([np.nan, np.nan])
    vpv = estimate_family_vp(v_segments) if len(v_segments) > 0 else np.array([np.nan, np.nan])
    hmtx = np.eye(3)
    hmtx[:, 0] = vpu
    hmtx[:, 1] = vpv
    return hmtx

def rectify_segments(segments: list[Segment], H0: np.ndarray) -> list[Segment]:
    """ Rectify segments using homography H0 """
    H0_inv = np.linalg.inv(H0)
    print(H0_inv @ H0[:, 0])
    print(H0_inv @ H0[:, 1])
    print(H0_inv @ H0[:, 2])
    rectified_segments = []
    for seg in segments:
        p0_h = np.array([seg.p0[0], seg.p0[1], 1.0])
        p1_h = np.array([seg.p1[0], seg.p1[1], 1.0])
        p0_rect_h = H0_inv @ p0_h
        p1_rect_h = H0_inv @ p1_h
        p0_rect = p0_rect_h[:2] / p0_rect_h[2]
        p1_rect = p1_rect_h[:2] / p1_rect_h[2]

        rectified_segments.append(Segment(id=seg.id, p0=p0_rect, p1=p1_rect))
    return rectified_segments

def segment_direction(seg: Segment) -> np.ndarray:
    """Unit direction vector of the segment (in image pixels)."""
    d = seg.p1 - seg.p0
    n = np.linalg.norm(d)
    if n < 1e-6:
        return np.array([1.0, 0.0])  # arbitrary, degenerate segment
    return d / n

def angle_residual_segment_vp_infinite(seg: Segment, vp_dir: np.ndarray) -> float:
    """
    Angular residual (in radians) between a segment and a vanishing point at infinity.

    Compare segment direction with vp direction (vx, vy).
    """
    t = segment_direction(seg)  # unit tangent

    if np.linalg.norm(t) < 1e-6:
        return 0.0  # ignore degenerate

    vdir = vp_dir
    n = np.linalg.norm(vdir)
    if n < 1e-6:
        return np.pi / 2  # undefined, treat as 90°
    vdir /= n
    # lines don't care about orientation sign, so take |cos|
    cosang = np.clip(np.abs(t @ vdir), -1.0, 1.0)
    return float(np.arccos(cosang))

def angle_residual_segment_vp(seg: Segment, vp: np.ndarray) -> float:
    """
    Angular residual (in radians) between a segment and a vanishing point.

    - If vp[2] != 0: vp is finite, compare segment direction with ray from midpoint to vp.
    - If vp[2] == 0: vp is at infinity, compare segment direction with vp direction (vx, vy).
    """
    t = segment_direction(seg)  # unit tangent

    if np.linalg.norm(t) < 1e-6:
        return 0.0  # ignore degenerate

    if abs(vp[2]) < 1e-6:
        return angle_residual_segment_vp_infinite(seg, vp[:2])

    # Finite VP: compare with ray from segment midpoint to VP
    vp_xy = vp[:2] / vp[2]
    mid = 0.5 * (seg.p0 + seg.p1)
    u = vp_xy - mid
    n = np.linalg.norm(u)
    if n < 1e-6:
        return 0.0  # vp almost at midpoint, arbitrary small residual
    u /= n
    cosang = np.clip(np.abs(t @ u), -1.0, 1.0)
    return float(np.arccos(cosang))

