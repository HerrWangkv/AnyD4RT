#!/usr/bin/env python3
"""S1 minimal checks on real PointOdyssey videos, no generation (README §6 S1).

Per clip (48 frames, stride s):
  1. projection: K, T_cw applied to trajs_3d vs trajs_2d (visible points).
  2. C_b = C_a reward (noise floor): OpenD4RT on the real clip, queries of §3.5
     (t_src = t_cam = t0 = 0, t_tgt = t), with the 40-frame interaction input.
  3. 40-frame vs 48-frame input: same queries, compare e / d on the common times t in [0, 39],
     plus per-frame depth (t_src = t_tgt = t_cam = t) AbsRel after a per-clip median scale.
  4. frozen-video control (degradation for diagnosis only, §5.1(a)): frame 0 repeated for 40 frames.
  5. combined reward (t0-frame e, d + local-frame e_loc, §3.5) on real and frozen video.

For C_b = C_a, target-view visibility is the GT `visibs` (no z-buffer needed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anyd4rt.d4rt_critic import load_d4rt, prepare_video, run_queries  # noqa: E402
from anyd4rt.geometry import in_frame, project, transform, uv_normalize  # noqa: E402
from anyd4rt.reward_d4rt_gt import RewardParams, reward_d4rt_gt  # noqa: E402

T_FULL, T_INTER = 48, 40


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/tmp/kwang-mnt/pointodyssey/v2/val")
    ap.add_argument("--exp-dir", default="third_party/Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG")
    ap.add_argument("--num-scenes", type=int, default=15)
    ap.add_argument("--clips-per-scene", type=int, default=2)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--num-anchors", type=int, default=1024)
    ap.add_argument("--dyn-thresh", type=float, default=0.02, help="dynamic if max 3D motion > thresh * anchor depth")
    ap.add_argument("--depth-points", type=int, default=256, help="per frame, for the depth check")
    ap.add_argument("--out", default="outputs/s1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resummarize", action="store_true", help="rebuild summary.json from an existing clips.json, no model")
    return ap.parse_args()


def load_frames(scene: Path, idxs: np.ndarray) -> np.ndarray:
    paths = sorted((scene / "rgbs").iterdir())
    return np.stack([cv2.cvtColor(cv2.imread(str(paths[i])), cv2.COLOR_BGR2RGB) for i in idxs])


def sample_anchors(rng, ok0, dynamic, n):
    dyn, sta = np.flatnonzero(ok0 & dynamic), np.flatnonzero(ok0 & ~dynamic)
    n_dyn = min(len(dyn), n // 2)
    n_sta = min(len(sta), n - n_dyn)
    return np.concatenate([rng.choice(dyn, n_dyn, replace=False), rng.choice(sta, n_sta, replace=False)])


def traj_queries(model, video, aspect, uv0, T):
    n = uv0.shape[0]
    t_tgt = np.tile(np.arange(T), n)
    pred = run_queries(model, video, aspect, np.repeat(uv0, T, 0), np.zeros_like(t_tgt), t_tgt, np.zeros_like(t_tgt))
    return pred["xyz_3d"].reshape(n, T, 3)


def traj_queries_local(model, video, aspect, uv0, T):
    """Local-frame queries of §3.5 (t_src = t0, t_tgt = t_cam = t): positions in each frame's own camera."""
    n = uv0.shape[0]
    t_tgt = np.tile(np.arange(T), n)
    pred = run_queries(model, video, aspect, np.repeat(uv0, T, 0), np.zeros_like(t_tgt), t_tgt, t_tgt)
    return pred["xyz_3d"].reshape(n, T, 3)


def depth_check(model, video, aspect, px_t, zc_t, ok_t, frames, rng, m):
    """Per-frame depth: query at the GT pixel of frame t with t_src = t_tgt = t_cam = t."""
    uv, tt, gt = [], [], []
    for t in frames:
        cand = np.flatnonzero(ok_t[t])
        pick = rng.choice(cand, min(m, len(cand)), replace=False)
        uv.append(px_t[t, pick]); tt.append(np.full(len(pick), t)); gt.append(zc_t[t, pick])
    uv, tt, gt = np.concatenate(uv), np.concatenate(tt), np.concatenate(gt)
    z = run_queries(model, video, aspect, uv, tt, tt, tt)["xyz_3d"][:, 2]
    return z, gt, tt


def absrel(z, gt):
    ok = np.isfinite(z) & (z > 0)
    s = np.median(gt[ok] / z[ok])
    return float(np.mean(np.abs(s * z[ok] - gt[ok]) / gt[ok])), float(ok.mean())


def summarize_reward(out, dyn_mask):
    res = {k: out.get(k) for k in ("r", "e_mean", "d_mean", "eloc_mean", "s", "scale_valid", "n_V", "n_Vd", "n_Vloc", "n_Q0", "n_Vplus", "n_capped")}
    if out.get("scale_valid"):
        V = out["_V"]
        Vd = V & (np.arange(V.shape[1]) != 0)[None] & V[:, :1]
        for name, m in (("dyn", dyn_mask), ("sta", ~dyn_mask)):
            res[f"e_{name}"] = float(out["e"][V & m[:, None]].mean()) if (V & m[:, None]).any() else None
            res[f"d_{name}"] = float(out["d"][Vd & m[:, None]].mean()) if (Vd & m[:, None]).any() else None
            if out.get("e_loc") is not None:
                Vl = V & (np.arange(V.shape[1]) != 0)[None]
                res[f"eloc_{name}"] = float(out["e_loc"][Vl & m[:, None]].mean()) if (Vl & m[:, None]).any() else None
    return res


def reward(pred, Y, V, p, dyn, pred_loc=None, Y_loc=None):
    out = reward_d4rt_gt(pred, Y, V, t0=0, p=p, pred_loc=pred_loc, Y_loc=Y_loc)
    out["_V"] = V
    return summarize_reward(out, dyn)


def write_summary(rows, out_dir, p, stride):
    def mean(key, sub=None):
        vals = [(r[key].get(sub) if sub else r[key]) for r in rows]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    summary = {"n_clips": len(rows), "params": p.__dict__, "stride": stride,
               "proj_err_px_median": mean("proj_err_px_median"),
               "depth_absrel_T48": mean("depth_absrel_T48"), "depth_absrel_T40": mean("depth_absrel_T40")}
    keys = ["reward_T40", "reward_T48", "reward_T48_common40", "reward_frozen_T40",
            "reward_T40_comb", "reward_frozen_T40_comb", "reward_T40_local", "reward_frozen_T40_local"]
    for k in keys:
        summary[k] = {s: mean(k, s) for s in ("r", "e_mean", "d_mean", "eloc_mean", "e_dyn", "e_sta", "d_dyn", "d_sta", "eloc_dyn", "eloc_sta")}
        summary[k]["scale_valid_frac"] = mean(k, "scale_valid")
    if rows:
        summary["frozen_below_real_frac"] = float(np.mean([r["reward_frozen_T40"]["r"] < r["reward_T40"]["r"] for r in rows]))
        summary["frozen_below_real_static_e_frac"] = float(np.mean([r["reward_frozen_T40"]["e_sta"] > r["reward_T40"]["e_sta"] for r in rows]))
        summary["frozen_below_real_frac_comb"] = float(np.mean([r["reward_frozen_T40_comb"]["r"] < r["reward_T40_comb"]["r"] for r in rows]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))



def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = RewardParams()
    if a.resummarize:
        write_summary(json.loads((out_dir / "clips.json").read_text()), out_dir, p, a.stride)
        return
    t_load = time.time()
    model = load_d4rt(a.exp_dir)
    print(f"loaded OpenD4RT in {time.time() - t_load:.0f}s", flush=True)
    dev = next(model.parameters()).device

    scenes = sorted(d for d in Path(a.data_root).iterdir() if d.is_dir() and (d / "anno.npz").exists())[: a.num_scenes]
    rows = []
    for scene in scenes:
        anno = np.load(scene / "anno.npz", allow_pickle=True)
        n_frames = anno["trajs_3d"].shape[0]
        span = a.stride * (T_FULL - 1) + 1
        starts = np.linspace(0, n_frames - span, a.clips_per_scene).astype(int)
        for start in starts:
            t_clip = time.time()
            idx = start + a.stride * np.arange(T_FULL)
            X = anno["trajs_3d"][idx].astype(np.float64)  # [T,N,3] world
            vis, val = anno["visibs"][idx], anno["valids"][idx]
            K, T_cw = anno["intrinsics"][idx].astype(np.float64), anno["extrinsics"][idx].astype(np.float64)
            frames = load_frames(scene, idx)
            H, W = frames.shape[1:3]

            Xc_t = transform(T_cw[:, None], X)  # each point in its own frame's camera
            px_t, zc_t = project(K[:, None], Xc_t)
            gt_ok = val & vis & in_frame(px_t, (W, H)) & (zc_t > 0)
            proj_err = np.linalg.norm(px_t - anno["trajs_2d"][idx], axis=-1)[val & vis]

            # Anchors at t0 = 0; §3.5 reference Y in cam(t0).
            motion = np.nanmax(np.linalg.norm(X - X[:1], axis=-1), axis=0)
            dynamic = motion > a.dyn_thresh * np.maximum(zc_t[0], 1e-3)
            ok0 = gt_ok[0] & np.isfinite(X).all(-1).all(0)
            if ok0.sum() < 64:
                print(f"skip {scene.name}@{start}: {ok0.sum()} anchors"); continue
            ids = sample_anchors(rng, ok0, dynamic, a.num_anchors)
            Y = transform(T_cw[0], X[:, ids]).transpose(1, 0, 2)  # [N,T,3]
            V = gt_ok[:, ids].T  # GT-only (C_b = C_a: target visibility = GT visibs)
            dyn = dynamic[ids]
            uv0 = uv_normalize(px_t[0, ids], (W, H))
            uv_t = uv_normalize(px_t, (W, H))

            row = {"scene": scene.name, "start": int(start), "stride": a.stride, "n_anchor": len(ids), "n_dyn": int(dyn.sum()),
                   "proj_err_px_median": float(np.median(proj_err)), "proj_err_px_p99": float(np.percentile(proj_err, 99))}
            depth_frames = np.arange(0, T_INTER, 4)
            for T in (T_FULL, T_INTER):
                video, aspect = prepare_video(frames[:T], dev)
                pred = traj_queries(model, video, aspect, uv0, T)
                row[f"reward_T{T}"] = reward(pred, Y[:, :T], V[:, :T], p, dyn)
                if T == T_FULL:  # 48-frame prediction restricted to the common times [0, 39]
                    row["reward_T48_common40"] = reward(pred[:, :T_INTER], Y[:, :T_INTER], V[:, :T_INTER], p, dyn)
                z, gt, _ = depth_check(model, video, aspect, uv_t, zc_t, gt_ok, depth_frames, np.random.default_rng(a.seed), a.depth_points)
                row[f"depth_absrel_T{T}"], row[f"depth_finite_T{T}"] = absrel(z, gt)
            # Combined reward (§3.5: t0-frame e, d + local e_loc) on the 40-frame interaction input, real and frozen.
            Y_loc = Xc_t[:T_INTER, ids].transpose(1, 0, 2)  # Y_loc(i,t) = T_cw(t) X_GT(i,t)
            Vi = V[:, :T_INTER]
            for tag, clip in (("", frames[:T_INTER]), ("frozen_", np.repeat(frames[:1], T_INTER, 0))):
                video, aspect = prepare_video(clip, dev)
                pr, pl = traj_queries(model, video, aspect, uv0, T_INTER), traj_queries_local(model, video, aspect, uv0, T_INTER)
                if tag:
                    row["reward_frozen_T40"] = reward(pr, Y[:, :T_INTER], Vi, p, dyn)
                row[f"reward_{tag}T40_comb"] = reward(pr, Y[:, :T_INTER], Vi, p, dyn, pl, Y_loc)
                row[f"reward_{tag}T40_local"] = reward(pl, Y_loc, Vi, p, dyn)  # everything in the local frame (reference)
            row["sec"] = round(time.time() - t_clip, 1)
            rows.append(row)
            r40, r48c, rf = row["reward_T40"], row["reward_T48_common40"], row["reward_frozen_T40"]
            rc, rcf = row["reward_T40_comb"], row["reward_frozen_T40_comb"]
            print(f"{scene.name}@{start}: proj {row['proj_err_px_median']:.2f}px | t0-only r40 {r40['r']:.3f} frozen {rf['r']:.3f} "
                  f"| comb r40 {rc['r']:.3f} frozen {rcf['r']:.3f} | r48c {r48c['r']:.3f} | depth48 {row['depth_absrel_T48']:.3f} "
                  f"depth40 {row['depth_absrel_T40']:.3f} | {row['sec']}s", flush=True)
            (out_dir / "clips.json").write_text(json.dumps(rows, indent=1))

    write_summary(rows, out_dir, p, a.stride)


if __name__ == "__main__":
    main()
