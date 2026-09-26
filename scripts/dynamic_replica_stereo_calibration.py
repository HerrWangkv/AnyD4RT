#!/usr/bin/env python3
"""Calibrate source-side D4RT reliability against a real synchronized stereo view.

This is deliberately a calibration experiment, not a training pipeline. It
uses DynamicReplica's left/right synchronized RGB and cameras, left-view world
trajectory GT, and right-view depth for target visibility. The right view has
no independent trajectory ``.pth`` files in the local release; projecting the
shared world trajectory into the right camera is therefore the minimal valid
target-side GT construction.

Thresholds are selected on ``--tune-split`` and frozen before the
``--report-split`` evaluation. Source masks never read target predictions or
target errors. The report contains raw coverage/error results and an
equal-anchor-coverage random baseline so a lower error cannot be attributed
only to deleting more points.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
OD_ROOT = ROOT / "third_party" / "Open-d4rt"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(OD_ROOT) not in sys.path:
    sys.path.insert(0, str(OD_ROOT))

from anyd4rt.d4rt_critic import encode, load_d4rt, prepare_video, run_queries  # noqa: E402
from anyd4rt.reward_d4rt_gt import RewardParams, fit_scale, reward_d4rt_gt  # noqa: E402
from src.data.dynamic_replica_raw_dataset import _viewpoint_to_camera  # noqa: E402

T_D4RT = 40
P = RewardParams()


def _annotation_path(root: Path, split: str) -> Path:
    for name in (f"frame_annotations_{split}.json", f"frame_annotations_{split}.jgz"):
        path = root / split / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no frame annotation JSON/JGZ under {root / split}")


def _load_annotations(root: Path, split: str) -> dict[str, dict[str, dict[int, dict]]]:
    path = _annotation_path(root, split)
    if path.suffix == ".jgz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            raw = json.load(handle)
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, dict[int, dict]]] = {}
    for item in raw:
        image_path = str(item.get("image", {}).get("path", ""))
        scene = str(item.get("sequence_name", ""))
        camera = str(item.get("camera_name", ""))
        frame = int(item.get("frame_number", -1))
        if not scene or not camera or frame < 0 or "/" not in image_path:
            continue
        out.setdefault(scene, {}).setdefault(camera, {})[frame] = item
    return out


def paired_scenes(root: Path, split: str) -> list[str]:
    ann = _load_annotations(root, split)
    return sorted(scene for scene, cams in ann.items() if {"left", "right"} <= set(cams))


def _path(root: Path, split: str, item: dict, key: str) -> Path | None:
    value = item.get(key, {})
    if not isinstance(value, dict):
        return None
    rel = value.get("path")
    return root / split / rel if rel else None


def _read_depth(path: Path, scale_adjustment: float) -> np.ndarray:
    raw = np.asarray(Image.open(path), dtype=np.uint16)
    # DynamicReplica v2's geometric PNGs are float16 bitcasts. Keep the decode
    # explicit here and validate it against left-view trajectory depth below.
    depth = np.ascontiguousarray(raw).view(np.float16).astype(np.float32)
    depth[~np.isfinite(depth) | (depth <= 0)] = 0.0
    return depth * float(scale_adjustment)


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _world_to_cam(T_cw: np.ndarray, X: np.ndarray) -> np.ndarray:
    if T_cw.shape[0] == 1 and X.shape[0] != 1:
        T_cw = np.repeat(T_cw, X.shape[0], axis=0)
    return np.einsum("tij,tnj->tni", T_cw[:, :3, :3], X) + T_cw[:, None, :3, 3]


def _project(K: np.ndarray, X_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.einsum("tij,tnj->tni", K, X_cam)
    with np.errstate(divide="ignore", invalid="ignore"):
        px = p[..., :2] / p[..., 2:3]
    return px, X_cam[..., 2]


def _in_frame(px: np.ndarray, width: int, height: int) -> np.ndarray:
    return (np.isfinite(px).all(-1) & (px[..., 0] >= 0) & (px[..., 0] <= width - 1)
            & (px[..., 1] >= 0) & (px[..., 1] <= height - 1))


def _depth_visibility(depth: np.ndarray, px: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Visibility against the target depth map, with a small hole fallback."""
    h, w = depth.shape
    ok = np.isfinite(z) & (z > 0) & _in_frame(px, w, h)
    x = np.clip(np.rint(px[..., 0]).astype(int), 0, w - 1)
    y = np.clip(np.rint(px[..., 1]).astype(int), 0, h - 1)
    sampled = depth[y, x]
    missing = ok & (sampled <= 0)
    if missing.any():
        pad = np.pad(depth, 1, constant_values=0)
        neighbours = np.stack([pad[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)], axis=0)
        valid = neighbours > 0
        fallback = np.where(valid, neighbours, np.inf).min(axis=0)
        sampled = np.where(missing, fallback[y, x], sampled)
    tol = np.maximum(0.05, 0.02 * z)
    return ok & np.isfinite(sampled) & (sampled > 0) & (np.abs(sampled - z) <= tol)


def _load_clip(root: Path, split: str, scene: str, start: int, stride: int,
               annotations: dict[str, dict[str, dict[int, dict]]], num_points: int,
               seed: int) -> dict:
    cams = annotations[scene]
    frame_ids = [int(start + stride * i) for i in range(T_D4RT)]
    if not all(fid in cams["left"] and fid in cams["right"] for fid in frame_ids):
        raise ValueError(f"clip does not have synchronized left/right frames: {scene} {start} {stride}")

    frames = {"left": [], "right": []}
    K = {"left": [], "right": []}
    T_cw = {"left": [], "right": []}
    depth_visibility = {"left": [], "right": []}
    X_list, vis_list, traj_2d_list = [], [], []
    source_depth_errors = []
    image_h = image_w = None

    for fid in frame_ids:
        trajectory_item = cams["left"][fid]
        traj_path = _path(root, split, trajectory_item, "trajectories")
        if traj_path is None or not traj_path.exists():
            raise FileNotFoundError(f"left trajectory missing for {scene} frame {fid}: {traj_path}")
        pack = torch.load(traj_path, map_location="cpu")
        X_list.append(pack["traj_3d_world"].numpy().astype(np.float32))
        vis_list.append(pack["verts_inds_vis"].numpy().astype(bool))
        traj_2d_list.append(pack["traj_2d"].numpy().astype(np.float32))

        for view in ("left", "right"):
            item = cams[view][fid]
            h, w = item["image"]["size"]
            image_h, image_w = int(h), int(w)
            k, t_cw, _ = _viewpoint_to_camera(item["viewpoint"], image_h, image_w, "dynamic_replica_v2")
            K[view].append(k.astype(np.float32))
            T_cw[view].append(t_cw.astype(np.float32))
            image_path = _path(root, split, item, "image")
            depth_path = _path(root, split, item, "depth")
            if image_path is None or depth_path is None:
                raise FileNotFoundError(f"RGB/depth missing for {scene} {view} frame {fid}")
            frames[view].append(_read_rgb(image_path))
            depth = _read_depth(depth_path, item.get("depth", {}).get("scale_adjustment", 1.0))
            depth_visibility[view].append(depth)

    X = np.stack(X_list, axis=0)
    source_traj_vis = np.stack(vis_list, axis=0)
    traj_2d = np.stack(traj_2d_list, axis=0)
    if len({x.shape for x in X_list}) != 1 or len({x.shape for x in vis_list}) != 1:
        raise ValueError(f"trajectory point shape changes inside {scene}")
    K = {view: np.stack(value, axis=0) for view, value in K.items()}
    T_cw = {view: np.stack(value, axis=0) for view, value in T_cw.items()}
    frames = {view: np.stack(value, axis=0) for view, value in frames.items()}

    X_cam = {view: _world_to_cam(T_cw[view], X) for view in ("left", "right")}
    px = {}
    z = {}
    visible = {}
    for view in ("left", "right"):
        px[view], z[view] = _project(K[view], X_cam[view])
        visible[view] = np.stack([
            _depth_visibility(depth_visibility[view][t], px[view][t], z[view][t])
            for t in range(T_D4RT)
        ], axis=0)
    visible["left"] &= source_traj_vis

    valid_source_projection = source_traj_vis & _in_frame(px["left"], image_w, image_h) & (z["left"] > 0)
    proj_err = np.linalg.norm(px["left"] - traj_2d[..., :2], axis=-1)
    pe = proj_err[valid_source_projection & np.isfinite(proj_err)]
    for t in range(T_D4RT):
        m = visible["left"][t]
        if m.any():
            sampled = depth_visibility["left"][t][
                np.clip(np.rint(px["left"][t, m, 1]).astype(int), 0, image_h - 1),
                np.clip(np.rint(px["left"][t, m, 0]).astype(int), 0, image_w - 1),
            ]
            source_depth_errors.append(np.abs(sampled - z["left"][t, m]) / np.maximum(z["left"][t, m], 1e-6))
    depth_err = np.concatenate(source_depth_errors) if source_depth_errors else np.empty(0)

    rng = np.random.default_rng(seed)
    common_t0 = visible["left"][0] & visible["right"][0]
    ids = np.flatnonzero(common_t0)
    if ids.size < num_points:
        chosen = ids
    else:
        # Balance dynamic/static points using the source GT motion. This label
        # is computed before any D4RT prediction and is not a learned signal.
        y_src_all = X_cam["left"]
        zbar = np.maximum(y_src_all[0, :, 2], P.z_min)
        motion = np.linalg.norm(y_src_all - y_src_all[:1], axis=-1).max(axis=0) / zbar
        dynamic = motion > 0.02
        dyn_ids, sta_ids = ids[dynamic[ids]], ids[~dynamic[ids]]
        n_dyn = min(len(dyn_ids), num_points // 2)
        n_sta = min(len(sta_ids), num_points - n_dyn)
        chosen = np.concatenate([
            rng.choice(dyn_ids, n_dyn, replace=False) if n_dyn else np.empty(0, dtype=int),
            rng.choice(sta_ids, n_sta, replace=False) if n_sta else np.empty(0, dtype=int),
        ])
        if chosen.size < num_points:
            rest = np.setdiff1d(ids, chosen, assume_unique=False)
            chosen = np.concatenate([chosen, rng.choice(rest, num_points - chosen.size, replace=False)])
        rng.shuffle(chosen)

    X = X[:, chosen]
    visible = {view: value[:, chosen] for view, value in visible.items()}
    px = {view: value[:, chosen] for view, value in px.items()}
    rays = {}
    centers = {view: np.linalg.inv(T_cw[view])[:, :3, 3] for view in ("left", "right")}
    for view in ("left", "right"):
        rays[view] = X - centers[view][:, None]
    cos = (rays["left"] * rays["right"]).sum(-1) / (
        np.linalg.norm(rays["left"], axis=-1) * np.linalg.norm(rays["right"], axis=-1) + 1e-12
    )
    view_angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).T

    y = {}
    y_loc = {}
    for view in ("left", "right"):
        y[view] = _world_to_cam(T_cw[view][:1], X).transpose(1, 0, 2)
        y_loc[view] = _world_to_cam(T_cw[view], X).transpose(1, 0, 2)
    zbar = np.maximum(y["left"][:, 0, 2], P.z_min)
    motion_rel = np.linalg.norm(y["left"] - y["left"][:, :1], axis=-1) / zbar[:, None]
    dynamic = np.nanmax(np.where(visible["left"].T, motion_rel, np.nan), axis=1) > 0.02

    return {
        "scene": scene, "split": split, "start": int(start), "stride": int(stride),
        "frames": frames, "K": K, "T_cw": T_cw, "X": X, "px": px, "visible": visible,
        "Y": y, "Y_loc": y_loc, "view_angle": view_angle, "dynamic": dynamic,
        "source_reprojection_median_px": float(np.median(pe)) if pe.size else None,
        "source_depth_rel_median": float(np.median(depth_err)) if depth_err.size else None,
        "source_depth_rel_p90": float(np.percentile(depth_err, 90)) if depth_err.size else None,
        "n_points": int(len(chosen)), "n_common_t0": int(common_t0[chosen].sum()),
    }


def _query_uv(px0: np.ndarray, width: int, height: int) -> np.ndarray:
    # OpenD4RT training convention after the native -> 256 resize.
    return px0 * np.array([256.0 / width, 256.0 / height], dtype=np.float32) / 255.0


def _predict_view(model, frames: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    video, aspect = prepare_video(frames, "cuda")
    memory = encode(model, video, aspect)
    n = len(uv)
    t_tgt = np.tile(np.arange(T_D4RT, dtype=np.int64), n)
    uv_rep = np.repeat(uv, T_D4RT, axis=0)
    t_src = np.zeros_like(t_tgt)
    global_pred = run_queries(model, video, aspect, uv_rep, t_src, t_tgt, t_src, memory=memory)["xyz_3d"]
    local_pred = run_queries(model, video, aspect, uv_rep, t_src, t_tgt, t_tgt, memory=memory)["xyz_3d"]
    return global_pred.reshape(n, T_D4RT, 3), local_pred.reshape(n, T_D4RT, 3)


def _per_anchor(err: np.ndarray, mask: np.ndarray) -> np.ndarray:
    count = mask.sum(1)
    return np.where(count > 0, (err * mask).sum(1) / np.maximum(count, 1), np.nan)


def _run_critic(model, clip: dict) -> dict:
    outputs = {}
    for view in ("left", "right"):
        h, w = clip["frames"][view].shape[1:3]
        uv = _query_uv(clip["px"][view][0], w, h)
        pred, pred_loc = _predict_view(model, clip["frames"][view], uv)
        V = clip["visible"][view].T
        fit = fit_scale(pred, clip["Y"][view], V, 0, P)
        out = reward_d4rt_gt(pred, clip["Y"][view], V, 0, P, pred_loc, clip["Y_loc"][view],
                             term_masks={"e": V, "d": V, "loc": V}, scale_mask=V,
                             scale_override=fit["s"] if fit["scale_valid"] else np.nan)
        if not out.get("scale_valid", False):
            raise RuntimeError(f"scale fit failed for {clip['split']} {clip['scene']} {view}: {fit}")
        outputs[view] = {
            "fit": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                    for k, v in fit.items() if k in ("s", "n_Q0", "n_Vplus")},
            "V": V, "Y": clip["Y"][view], "e": out["e"], "d": out["d"], "loc": out["e_loc"],
            "e_i": _per_anchor(out["e"], V),
            "d_i": _per_anchor(out["d"], V & (np.arange(T_D4RT) != 0)[None] & V[:, :1]),
            "loc_i": _per_anchor(out["e_loc"], V & (np.arange(T_D4RT) != 0)[None]),
        }
    return outputs


def _source_support(view: dict, dynamic: np.ndarray, pos_tau: float, loc_tau: float,
                    disp_abs_tau: float, disp_ratio: float, min_motion: float) -> dict:
    V = view["V"]
    src0 = V[:, 0]
    not_t0 = np.arange(V.shape[1]) != 0
    motion = _per_anchor(
        np.linalg.norm(view["Y"] - view["Y"][:, :1], axis=-1) /
        np.maximum(view["Y"][:, :1, 2], P.z_min),
        V & not_t0[None, :] & V[:, :1],
    )
    pos = src0 & np.isfinite(view["e_i"]) & (view["e_i"] <= pos_tau)
    loc = src0 & np.isfinite(view["loc_i"]) & (view["loc_i"] <= loc_tau)
    static_d = (~dynamic) & np.isfinite(view["d_i"]) & (view["d_i"] <= disp_abs_tau)
    dynamic_d = (dynamic & np.isfinite(motion) & (motion >= min_motion) & np.isfinite(view["d_i"])
                 & (view["d_i"] / np.maximum(motion, 1e-12) <= disp_ratio))
    disp = src0 & (static_d | dynamic_d)
    return {"e": pos, "d": disp, "loc": loc, "motion": motion,
            "low_motion_dynamic": dynamic & np.isfinite(motion) & (motion < min_motion)}


def _masks(clip: dict, target: dict, support: dict, config: str,
           angle_thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    V = target["V"]
    base = clip["visible"]["left"].T[:, :1] & V[:, :1]
    angle = clip["view_angle"]
    masks = {}
    not_t0 = (np.arange(V.shape[1]) != 0)[None, :]
    for key in ("e", "d", "loc"):
        source = np.ones(len(base), bool) if config == "N" else support[key]
        view = np.ones_like(V, bool) if config != "SV" else (angle <= float(angle_thresholds[key]))
        masks[key] = V & base & source[:, None] & view
        if key in ("d", "loc"):
            masks[key] &= not_t0
        if key == "d":
            masks[key] &= V[:, :1]
    return masks


def _term_stats(err: np.ndarray, mask: np.ndarray, base: np.ndarray, dynamic: np.ndarray) -> dict:
    values = err[mask]
    anchors = mask.any(1)
    per = _per_anchor(err, mask)
    return {
        "mean_entry": float(values.mean()) if values.size else None,
        "median_entry": float(np.median(values)) if values.size else None,
        "mean_anchor": float(np.nanmean(per)) if np.isfinite(per).any() else None,
        "median_anchor": float(np.nanmedian(per)) if np.isfinite(per).any() else None,
        "entries": int(mask.sum()), "base_entries": int(base.sum()),
        "entry_coverage": float(mask.sum() / max(base.sum(), 1)),
        "anchors": int(anchors.sum()), "base_anchors": int(base.any(1).sum()),
        "anchor_coverage": float(anchors.sum() / max(base.any(1).sum(), 1)),
        "dynamic_anchor_coverage": float(anchors[dynamic].sum() / max(base.any(1)[dynamic].sum(), 1)) if dynamic.any() else None,
        "static_anchor_coverage": float(anchors[~dynamic].sum() / max(base.any(1)[~dynamic].sum(), 1)) if (~dynamic).any() else None,
    }


def _random_equal_coverage(err: np.ndarray, selected: np.ndarray, base: np.ndarray,
                           dynamic: np.ndarray, rng: np.random.Generator, draws: int = 100) -> dict:
    eligible = np.flatnonzero(base.any(1))
    chosen = np.flatnonzero(selected.any(1))
    if chosen.size == 0 or chosen.size > eligible.size:
        return {"n_anchors": int(chosen.size), "mean_entry": None, "std_mean_entry": None}
    vals = []
    for _ in range(draws):
        pick = rng.choice(eligible, size=chosen.size, replace=False)
        keep = np.zeros(err.shape[0], bool); keep[pick] = True
        vals.append(float(err[base & keep[:, None]].mean()) if (base & keep[:, None]).any() else np.nan)
    return {"n_anchors": int(chosen.size), "mean_entry": float(np.nanmean(vals)),
            "std_mean_entry": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else None,
            "draws": draws}


def _selected_stats(target: dict, clip: dict, support: dict, config: str,
                    angle_thresholds: dict[str, float], rng: np.random.Generator) -> dict:
    masks = _masks(clip, target, support, config, angle_thresholds)
    V = target["V"]
    not_t0 = (np.arange(V.shape[1]) != 0)[None, :]
    base = {
        "e": V & (clip["visible"]["left"].T[:, :1] & V[:, :1]),
        "d": V & (clip["visible"]["left"].T[:, :1] & V[:, :1]) & not_t0 & V[:, :1],
        "loc": V & (clip["visible"]["left"].T[:, :1] & V[:, :1]) & not_t0,
    }
    out = {"coverage": {key: _term_stats(target[key], masks[key], base[key], clip["dynamic"]) for key in base},
           "equal_anchor_coverage": {}}
    s_masks = _masks(clip, target, support, "S", angle_thresholds)
    for key in base:
        out["equal_anchor_coverage"][key] = {
            "N_random": _random_equal_coverage(target[key], masks[key], base[key], clip["dynamic"], rng),
        }
        if config == "SV":
            out["equal_anchor_coverage"][key]["S_random"] = _random_equal_coverage(
                target[key], masks[key], s_masks[key], clip["dynamic"], rng)
    return out


def _tau_b(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return {"tau_b": None, "pvalue": None, "n": int(mask.sum())}
    result = kendalltau(x[mask], y[mask], variant="b", nan_policy="omit")
    return {"tau_b": float(result.statistic), "pvalue": float(result.pvalue), "n": int(mask.sum())}


def _source_target_association(clip: dict) -> dict:
    """Per-anchor source-vs-target and angle-vs-target associations.

    These are descriptive calibration statistics. The target errors are never
    used to form a source mask; they are only compared after the mask is fixed.
    """
    source, target = clip["source"], clip["target"]
    base = clip["visible"]["left"].T[:, 0] & clip["visible"]["right"].T[:, 0]
    V = target["V"]
    not_t0 = (np.arange(V.shape[1]) != 0)[None, :]
    term_masks = {
        "e": V & base[:, None],
        "d": V & base[:, None] & not_t0 & V[:, :1],
        "loc": V & base[:, None] & not_t0,
    }
    out = {}
    for key, suffix in (("e", "e_i"), ("d", "d_i"), ("loc", "loc_i")):
        mask = term_masks[key].any(1)
        angle_i = _per_anchor(clip["view_angle"], term_masks[key])
        out[key] = {
            "source_error_vs_target_error": _tau_b(source[suffix][mask], target[suffix][mask]),
            "mean_view_angle_deg_vs_target_error": _tau_b(angle_i[mask], target[suffix][mask]),
        }
    return out


def _aggregate_clips(clip_results: list[dict], pos_tau: float, loc_tau: float,
                     disp_abs_tau: float, disp_ratio: float, min_motion: float,
                     angle_thresholds: dict[str, float], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for clip in clip_results:
        support = _source_support(clip["source"], clip["dynamic"], pos_tau, loc_tau, disp_abs_tau, disp_ratio, min_motion)
        row = {"scene": clip["scene"], "split": clip["split"], "start": clip["start"],
               "n_points": clip["n_points"], "n_common_t0": clip["n_common_t0"],
               "view_angle_deg_percentiles": [float(x) for x in np.percentile(
                   clip["view_angle"][clip["visible"]["left"].T & clip["visible"]["right"].T],
                   [0, 25, 50, 75, 90, 95, 99, 100],
               )],
               "source": {"support": {k: {"n": int(support[k].sum()), "fraction": float(support[k].mean())}
                                         for k in ("e", "d", "loc")},
                          "low_motion_dynamic": int(support["low_motion_dynamic"].sum())},
               "association": _source_target_association(clip), "target": {}}
        for config in ("N", "S", "SV"):
            row["target"][config] = _selected_stats(clip["target"], clip, support, config, angle_thresholds, rng)
        rows.append(row)
    return {"n_clips": len(rows), "clips": rows}


def _tune_thresholds(clips: list[dict], args) -> dict:
    min_cov = float(args.min_tune_coverage)

    def grid_term(key, candidates, current):
        table = []
        for value in candidates:
            total, count, entries, coverage = 0.0, 0, 0, 0
            for clip in clips:
                support = _source_support(clip["source"], clip["dynamic"],
                                          value if key == "e" else current["pos_tau"],
                                          value if key == "loc" else current["loc_tau"],
                                          current["disp_abs_tau"], current["disp_ratio"], current["min_motion"])
                if key == "d":
                    support = _source_support(clip["source"], clip["dynamic"], current["pos_tau"], current["loc_tau"],
                                              current["disp_abs_tau"], value, current["min_motion"])
                target = clip["target"]
                base = target["V"] & (clip["visible"]["left"].T[:, :1] & target["V"][:, :1])
                not_t0 = (np.arange(base.shape[1]) != 0)[None, :]
                if key == "e":
                    mask = base & support["e"][:, None]
                    err = target["e"]
                elif key == "d":
                    mask = base & support["d"][:, None] & not_t0 & target["V"][:, :1]
                    err = target["d"]
                else:
                    mask = base & support["loc"][:, None] & not_t0
                    err = target["loc"]
                entries += int(mask.sum()); total += float(err[mask].sum()) if mask.any() else 0.0
                count += int(mask.sum())
                coverage += float(mask.sum() / max(base.sum(), 1))
            mean = total / max(count, 1)
            cov = coverage / max(len(clips), 1)
            table.append({"value": value, "mean_entry": mean if count else None, "entry_coverage": cov})
        valid = [x for x in table if x["mean_entry"] is not None and x["entry_coverage"] >= min_cov]
        best = min(valid, key=lambda x: x["mean_entry"]) if valid else table[0]
        return best["value"], table

    current = {"pos_tau": args.default_pos_tau, "loc_tau": args.default_loc_tau,
               "disp_abs_tau": args.default_disp_abs_tau, "disp_ratio": args.default_disp_ratio,
               "min_motion": args.default_min_motion}
    pos, pos_table = grid_term("e", [0.05, 0.10, 0.15, 0.20], current); current["pos_tau"] = pos
    loc, loc_table = grid_term("loc", [0.05, 0.10, 0.15, 0.20], current); current["loc_tau"] = loc
    ratio, ratio_table = grid_term("d", [0.25, 0.50, 0.75, 1.00], current); current["disp_ratio"] = ratio

    angle_tables = {}
    angle_thresholds = {}
    for key in ("e", "d", "loc"):
        table = []
        # DynamicReplica's left/right baseline is only 6.3 cm, so 5--30 deg
        # would keep every point and would not test the view-change factor.
        for angle in (0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
            total = count = coverage = 0.0
            for clip in clips:
                support = _source_support(clip["source"], clip["dynamic"], **current)
                target = clip["target"]
                base = target["V"] & (clip["visible"]["left"].T[:, :1] & target["V"][:, :1])
                not_t0 = (np.arange(base.shape[1]) != 0)[None, :]
                mask = base & support[key][:, None] & (clip["view_angle"] <= angle)
                if key in ("d", "loc"):
                    mask &= not_t0
                if key == "d":
                    mask &= target["V"][:, :1]
                err = target[key]
                total += float(err[mask].sum()) if mask.any() else 0.0
                count += int(mask.sum()); coverage += float(mask.sum() / max(base.sum(), 1))
            table.append({"value": angle, "mean_entry": total / max(count, 1),
                          "entry_coverage": coverage / max(len(clips), 1)})
        valid = [x for x in table if x["mean_entry"] is not None and x["entry_coverage"] >= min_cov]
        angle_thresholds[key] = min(valid, key=lambda x: x["mean_entry"])["value"] if valid else 10.0
        angle_tables[key] = table
    return {**current, "angle_deg_e": angle_thresholds["e"], "angle_deg_d": angle_thresholds["d"],
            "angle_deg_loc": angle_thresholds["loc"],
            "tune_rule": "minimize target GT entry error on tune split subject to average entry coverage >= min_tune_coverage",
            "min_tune_coverage": min_cov,
            "tables": {"pos_tau": pos_table, "loc_tau": loc_table, "disp_ratio": ratio_table, "angle": angle_tables}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/tmp/kwang-mnt/dynamicreplica/v2")
    ap.add_argument("--tune-split", default="valid")
    ap.add_argument("--report-split", default="test")
    ap.add_argument("--tune-scenes", default="06dcf6-3_obj,0cde48-3_obj")
    ap.add_argument("--report-scenes", default="01f258-3_obj,029beb-3_obj,26b966-3_obj,3592f1-3_obj")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--num-points", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-tune-coverage", type=float, default=0.25)
    ap.add_argument("--default-pos-tau", type=float, default=0.10)
    ap.add_argument("--default-loc-tau", type=float, default=0.10)
    ap.add_argument("--default-disp-abs-tau", type=float, default=0.10)
    ap.add_argument("--default-disp-ratio", type=float, default=0.50)
    ap.add_argument("--default-min-motion", type=float, default=0.03)
    ap.add_argument("--exp-dir", default=str(ROOT / "third_party/Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "dynamic_replica_stereo" / "report.json"))
    args = ap.parse_args()

    root = Path(args.data_root)
    tune_scenes = [x for x in args.tune_scenes.split(",") if x]
    report_scenes = [x for x in args.report_scenes.split(",") if x]
    ann_tune = _load_annotations(root, args.tune_split)
    ann_report = _load_annotations(root, args.report_split)
    missing_tune = sorted(set(tune_scenes) - set(paired_scenes(root, args.tune_split)))
    missing_report = sorted(set(report_scenes) - set(paired_scenes(root, args.report_split)))
    if missing_tune or missing_report:
        raise ValueError(f"unpaired scenes: tune={missing_tune}, report={missing_report}")

    model = load_d4rt(args.exp_dir)
    tune_clips = []
    for i, scene in enumerate(tune_scenes):
        clip = _load_clip(root, args.tune_split, scene, args.start, args.stride, ann_tune, args.num_points, args.seed + i)
        pred = _run_critic(model, clip)
        clip["source"], clip["target"] = pred["left"], pred["right"]
        tune_clips.append(clip)
        print(json.dumps({"split": args.tune_split, "scene": scene, "n_points": clip["n_points"],
                          "source_reprojection_median_px": clip["source_reprojection_median_px"],
                          "source_depth_rel_median": clip["source_depth_rel_median"]}), flush=True)
    thresholds = _tune_thresholds(tune_clips, args)
    angle_thresholds = {"e": thresholds["angle_deg_e"], "d": thresholds["angle_deg_d"], "loc": thresholds["angle_deg_loc"]}

    report_clips = []
    for i, scene in enumerate(report_scenes):
        clip = _load_clip(root, args.report_split, scene, args.start, args.stride, ann_report, args.num_points, args.seed + 100 + i)
        pred = _run_critic(model, clip)
        clip["source"], clip["target"] = pred["left"], pred["right"]
        report_clips.append(clip)
        print(json.dumps({"split": args.report_split, "scene": scene, "n_points": clip["n_points"],
                          "source_reprojection_median_px": clip["source_reprojection_median_px"],
                          "source_depth_rel_median": clip["source_depth_rel_median"]}), flush=True)

    result = {
        "experiment": "real_dynamic_replica_stereo_calibration",
        "data": {"root": str(root), "tune_split": args.tune_split, "report_split": args.report_split,
                 "tune_scenes": tune_scenes, "report_scenes": report_scenes,
                 "start": args.start, "stride": args.stride, "frames": T_D4RT, "num_points": args.num_points},
        "field_check": {
            "synchronized_left_right_rgb": True,
            "synchronized_left_right_cameras": True,
            "left_world_trajectory_gt": True,
            "right_independent_trajectory_files": False,
            "right_depth_gt_used_for_visibility": True,
            "target_gt_construction": "left traj_3d_world projected with right T_cw; right geometric depth checks visibility",
        },
        "thresholds": thresholds,
        "tune_summary": _aggregate_clips(tune_clips, thresholds["pos_tau"], thresholds["loc_tau"],
                                          thresholds["disp_abs_tau"], thresholds["disp_ratio"], thresholds["min_motion"],
                                          angle_thresholds, args.seed + 200),
        "report_summary": _aggregate_clips(report_clips, thresholds["pos_tau"], thresholds["loc_tau"],
                                            thresholds["disp_abs_tau"], thresholds["disp_ratio"], thresholds["min_motion"],
                                            angle_thresholds, args.seed + 300),
        "interpretation": {
            "source_filter_uses_target_prediction_or_error": False,
            "target_gt_used_for": "evaluation and threshold selection on tune split only",
            "same_coverage_control": "N random anchor subsets matched to each filtered term's retained anchor count",
            "no_formal_training": True,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out), "thresholds": thresholds, "report_clips": len(report_clips)}, indent=1), flush=True)


if __name__ == "__main__":
    main()
