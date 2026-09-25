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


VIS_UNKNOWN, VIS_VISIBLE, VIS_OCCLUDED = 0, 1, 2


def zbuffer(depth_src, K_src, T_cw_src, K_tgt, T_cw_tgt, tgt_wh, grid_wh=(256, 256), stride=1):
    """Forward-splat the source frame's depth into the target camera: min depth per grid cell (README §5).

    depth_src: [H,W] metric depth of the source frame (<= 0 or non-finite = invalid). Pixels are sampled every
    `stride` px. Target pixels (pixel-center, 0..W-1) map to cells by x * gw / W. Returns [gh, gw] (inf = empty).
    """
    H, W = depth_src.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    d = depth_src[ys, xs]
    ok = np.isfinite(d) & (d > 0)
    pix = np.stack([xs[ok], ys[ok], np.ones(ok.sum())], -1).astype(np.float64)
    X_cam = (np.linalg.solve(K_src, pix.T).T) * d[ok][:, None]
    X_w = transform(np.linalg.inv(T_cw_src), X_cam)
    px, z = project(K_tgt, transform(T_cw_tgt, X_w))
    gw, gh = grid_wh
    tw, th = tgt_wh
    keep = (z > 0) & in_frame(px, tgt_wh)
    cx = np.clip((px[keep, 0] * gw / tw).astype(int), 0, gw - 1)
    cy = np.clip((px[keep, 1] * gh / th).astype(int), 0, gh - 1)
    zb = np.full((gh, gw), np.inf)
    np.minimum.at(zb, (cy, cx), z[keep])
    return zb


def zbuffer_visibility(zb, px, z, tgt_wh, rel_tol=0.02, abs_tol=0.05):
    """Three-state visibility of points (target pixels px [...,2], depths z [...]) against a z-buffer (README §5).

    visible:  no known closer surface; occluded: a known surface closer than z - max(abs_tol, rel_tol * z);
    unknown:  the point's own cell and its 8 neighbours are all empty, or the point is behind the camera / off-frame.
    The point's own cell is used when it has a sample; the 3x3 neighbourhood minimum only fills splatting holes
    (taking the neighbourhood minimum everywhere marks background points next to depth edges as occluded).
    """
    gh, gw = zb.shape
    out = np.full(z.shape, VIS_UNKNOWN, dtype=np.int8)
    ok = np.isfinite(z) & (z > 0) & in_frame(px, tgt_wh)
    tw, th = tgt_wh
    pad = np.pad(zb, 1, constant_values=np.inf)
    nb = np.min(np.stack([pad[dy:dy + gh, dx:dx + gw] for dy in range(3) for dx in range(3)]), axis=0)
    cx = np.clip((px[..., 0] * gw / tw).astype(int), 0, gw - 1)
    cy = np.clip((px[..., 1] * gh / th).astype(int), 0, gh - 1)
    own = zb[cy, cx]
    zmin = np.where(ok, np.where(np.isfinite(own), own, nb[cy, cx]), np.inf)
    covered = ok & np.isfinite(zmin)
    tol = np.maximum(abs_tol, rel_tol * z)
    out[covered & (zmin < z - tol)] = VIS_OCCLUDED
    out[covered & ~(zmin < z - tol)] = VIS_VISIBLE
    return out


def orbit_camera(T_cw, pivot, axis_world, theta_deg):
    """Target trajectory of README §7.2: rotate the source trajectory by theta about `axis_world` through `pivot`.

    c_b(t) = p + R (c_a(t) - p),  R_wc_b(t) = R R_wc_a(t). T_cw: [T,4,4] world->camera. Returns [T,4,4].
    """
    ax = np.asarray(axis_world, dtype=np.float64)
    ax = ax / np.linalg.norm(ax)
    th = np.deg2rad(theta_deg)
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx
    T_wc = np.linalg.inv(T_cw)
    out = np.empty_like(T_wc)
    out[:] = np.eye(4)
    out[:, :3, :3] = R @ T_wc[:, :3, :3]
    out[:, :3, 3] = pivot + (R @ (T_wc[:, :3, 3] - pivot).T).T
    return np.linalg.inv(out)
