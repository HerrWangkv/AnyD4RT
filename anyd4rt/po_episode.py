"""PointOdyssey clip -> AnyView episode (README §7.2 minimal prototype, S4).

Coordinates: PointOdyssey `extrinsics` are OpenCV world->camera (T_cw), verified in S1 (0.14 px). The reward and
the z-buffer always use these original-world cameras; AnyView receives the same cameras and re-anchors them
internally (infer.reanchor_world2cam), which never feeds back into the reward (README §3.2).
Scale: PointOdyssey is not in AnyView's DATASET_SCALE_FACTORS; 1/8 (the table's default for non-driving,
non-Kubric data) is a provisional S2 choice.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from anyd4rt.geometry import in_frame, orbit_camera, project, transform

PO_SCALE_FACTOR = 1.0 / 8.0  # provisional (README §6 S2)


def load_clip(scene: Path, start: int, stride: int, T: int = 41, load_depth: bool = True) -> dict:
    anno = np.load(scene / "anno.npz", allow_pickle=True)
    idx = start + stride * np.arange(T)
    n = min(anno["trajs_3d"].shape[0], len(list((scene / "rgbs").iterdir())))
    assert idx[0] >= 0 and idx[-1] < n, (scene.name, start, stride, n)
    rgb = sorted((scene / "rgbs").iterdir())
    frames = np.stack([cv2.cvtColor(cv2.imread(str(rgb[i])), cv2.COLOR_BGR2RGB) for i in idx])
    K = anno["intrinsics"][idx].astype(np.float64)
    assert np.allclose(K, K[:1]), "PointOdyssey intrinsics are expected constant within a clip"
    clip = {"scene": scene.name, "start": int(start), "stride": int(stride), "idx": idx, "frames": frames,
            "K": K[0], "T_cw": anno["extrinsics"][idx].astype(np.float64),
            "X": anno["trajs_3d"][idx].astype(np.float64), "vis": anno["visibs"][idx], "val": anno["valids"][idx]}
    if load_depth:
        dp = sorted((scene / "depths").iterdir())
        clip["depth"] = np.stack([np.asarray(Image.open(dp[i]), dtype=np.uint16).astype(np.float64) / 65535.0 * 1000.0 for i in idx])
    return clip


def target_trajectory(clip: dict, theta_deg: float) -> np.ndarray:
    """README §7.2: pivot = median of GT points visible at t0, axis = C_a(0)'s y axis (in world)."""
    ok = clip["vis"][0] & clip["val"][0] & np.isfinite(clip["X"][0]).all(-1)
    pivot = np.median(clip["X"][0][ok], axis=0)
    axis = np.linalg.inv(clip["T_cw"][0])[:3, 1]
    return orbit_camera(clip["T_cw"], pivot, axis, theta_deg)


def av_item(clip: dict, T_cw_target: np.ndarray, target_gt=None) -> dict:
    """AnyView AVBDataset-style item: view 1 = input (source camera), view 0 = target camera."""
    K = torch.from_numpy(clip["K"]).float()
    return {"episode": f"{clip['scene']}@{clip['start']}", "input": torch.from_numpy(clip["frames"]),
            "target_gt": torch.from_numpy(target_gt if target_gt is not None else clip["frames"]),
            "intrinsics_input": K, "intrinsics_target": K,
            "world2cam_input": torch.from_numpy(clip["T_cw"]).float(), "world2cam_target": torch.from_numpy(T_cw_target).float(),
            "scale_factor": PO_SCALE_FACTOR}


def target_view_gt(clip: dict, T_cw_b: np.ndarray, zbuf_fn):
    """GT points in the target camera: pixels, depths, and z-buffer visibility state per (t, point).

    zbuf_fn(t) -> (zbuffer grid for target frame t). Returns px [T,N,2], z [T,N], state [T,N] (geometry.VIS_*).
    """
    from anyd4rt.geometry import zbuffer_visibility
    H, W = clip["frames"].shape[1:3]
    Xc = transform(T_cw_b[:, None], clip["X"])
    px, z = project(clip["K"][None, None], Xc)
    state = np.stack([zbuffer_visibility(zbuf_fn(t), px[t], z[t], (W, H)) for t in range(len(T_cw_b))])
    return px, z, state
