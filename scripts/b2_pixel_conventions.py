#!/usr/bin/env python3
"""B2 (README §7.3 item 4): quantify pixel-convention offsets, link by link. Measures, does not compensate.

Links:
  1. image resize: where does content at source pixel x land after resizing? Measured with a sub-pixel
     Gaussian blob, fitted as x' = a*x + b, for the resamplers actually used:
       PIL LANCZOS (AnyView, scripts/infer.py:124), PIL BILINEAR (OpenD4RT training loaders),
       cv2 INTER_AREA (OpenD4RT infer_track_3d.py and our S1 script).
  2. intrinsics: AnyView `scale_intrinsics` and the OpenD4RT loaders both scale K multiplicatively,
     so a projected point moves to s*x (no half-pixel term).
  3. pixel -> query: OpenD4RT samples the local patch at u*(W_in - 1) (grid_sample, align_corners=True).

Chains reported, in OpenD4RT 256-grid pixels, as the offset between where the query samples and where the
content actually is, across the image width:
  A. OpenD4RT training labels: u = s*x / (W_in - 1)                     (K scaled multiplicatively)
  B. our S1 queries:           u = x / (W_src - 1)
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from PIL import Image


def blob_image(w, h, cx, cy, sigma=3.0):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))


def centroid(img):
    img = np.clip(img, 0, None)
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    return float((xs * img).sum() / img.sum()), float((ys * img).sum() / img.sum())


def resize(img, wh, method):
    if method.startswith("PIL"):
        rs = {"PIL_LANCZOS": Image.LANCZOS, "PIL_BILINEAR": Image.BILINEAR}[method]
        return np.asarray(Image.fromarray(img.astype(np.float32), mode="F").resize(wh, resample=rs), dtype=np.float64)
    return cv2.resize(img.astype(np.float32), wh, interpolation={"cv2_AREA": cv2.INTER_AREA}[method]).astype(np.float64)


def measure_resize(src_wh, dst_wh, method, n=25, seed=0):
    """Fit x' = a*x + b (and y) from blobs at random sub-pixel positions away from the border."""
    rng = np.random.default_rng(seed)
    w, h = src_wh
    pts, got = [], []
    for _ in range(n):
        cx, cy = rng.uniform(40, w - 40), rng.uniform(40, h - 40)
        got.append(centroid(resize(blob_image(w, h, cx, cy, sigma=max(3.0, 2.0 * w / dst_wh[0])), dst_wh, method)))
        pts.append((cx, cy))
    pts, got = np.array(pts), np.array(got)
    fit = {}
    for k, name in ((0, "x"), (1, "y")):
        a, b = np.polyfit(pts[:, k], got[:, k], 1)
        s = dst_wh[k] / src_wh[k]
        fit[name] = {"a": a, "b": b, "s": s, "b_pixel_center": 0.5 * s - 0.5,
                     "resid_px_max": float(np.abs(a * pts[:, k] + b - got[:, k]).max())}
    return fit


def chain_offsets(src_w, in_w=256):
    """Offset (256-grid px) of query sample position minus true content position, at x = 0, mid, W-1."""
    s = in_w / src_w
    xs = np.array([0.0, (src_w - 1) / 2, src_w - 1.0])
    content = (xs + 0.5) * s - 0.5  # pixel-center resize (link 1, verified below)
    sample_A = (s * xs / (in_w - 1)) * (in_w - 1)  # OpenD4RT training labels
    sample_B = (xs / (src_w - 1)) * (in_w - 1)  # our S1 queries
    return {"x_src": xs.tolist(),
            "A_training_label_minus_content": (sample_A - content).tolist(),
            "B_s1_query_minus_content": (sample_B - content).tolist(),
            "B_minus_A": (sample_B - sample_A).tolist()}


def main():
    out = {"resize": {}, "chains": {}}
    for method in ("PIL_LANCZOS", "PIL_BILINEAR", "cv2_AREA"):
        for src, dst in (((960, 540), (576, 320)), ((960, 540), (256, 256)), ((576, 320), (256, 256))):
            out["resize"][f"{method} {src}->{dst}"] = measure_resize(src, dst, method)
    for w in (960, 576):
        out["chains"][f"W_src={w}"] = chain_offsets(w)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
