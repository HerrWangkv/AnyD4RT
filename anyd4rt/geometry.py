"""Camera conventions (README §3.2-3.3).

Convention: OpenCV camera axes (+x right, +y down, +z forward).
T_cw is world->camera (x_cam = R x_world + t); T_wc = inv(T_cw).
Pixel coordinates are pixel-center indexed: pixel (0, 0) is centered at (0, 0), the grid spans 0..W-1.
"""

from __future__ import annotations

import numpy as np


def transform(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply 4x4 rigid transform(s) T [...,4,4] to points X [...,3] (broadcast over leading dims)."""
    return np.einsum("...ij,...j->...i", T[..., :3, :3], X) + T[..., :3, 3]


def project(K: np.ndarray, X_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points. Returns pixel coords [...,2] and depth z [...]."""
    z = X_cam[..., 2]
    p = np.einsum("...ij,...j->...i", K, X_cam)
    with np.errstate(divide="ignore", invalid="ignore"):
        px = p[..., :2] / p[..., 2:3]
    return px, z


def in_frame(px: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    w, h = wh
    return np.isfinite(px).all(-1) & (px[..., 0] >= 0) & (px[..., 0] <= w - 1) & (px[..., 1] >= 0) & (px[..., 1] <= h - 1)


def uv_normalize(px: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    """OpenD4RT query coordinates: u = x/(W-1), v = y/(H-1)."""
    w, h = wh
    return px / np.array([max(w - 1, 1), max(h - 1, 1)], dtype=px.dtype)


def scale_intrinsics(K: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int], convention: str) -> np.ndarray:
    """Rescale intrinsics for an image resize src_wh -> dst_wh.

    convention="multiply": K rows scaled by new/orig (AnyView `scale_intrinsics`, AV/anyview/cameras.py:109-126).
    convention="center":   pixel-center exact resize, x' = (x + 0.5) * s - 0.5.
    The two differ in the principal point by 0.5 * (s - 1) pixels (README §3.2 item 2).
    """
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    K = np.array(K, dtype=np.float64, copy=True)
    K[..., 0, 0] *= sx
    K[..., 1, 1] *= sy
    if convention == "multiply":
        K[..., 0, 2] *= sx
        K[..., 1, 2] *= sy
    elif convention == "center":
        K[..., 0, 2] = (K[..., 0, 2] + 0.5) * sx - 0.5
        K[..., 1, 2] = (K[..., 1, 2] + 0.5) * sy - 0.5
    else:
        raise ValueError(convention)
    return K
