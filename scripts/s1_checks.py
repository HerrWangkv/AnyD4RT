#!/usr/bin/env python3
"""S1 minimal checks on real PointOdyssey videos, no generation (README §6 S1).

Per clip (48 frames, stride s):
  1. projection: K, T_cw applied to trajs_3d vs trajs_2d (visible points).
  2. C_b = C_a reward (noise floor): OpenD4RT on the real clip, queries of §3.5
     (t_src = t_cam = t0 = 0, t_tgt = t), with the 40-frame interaction input.
  3. 40-frame vs 48-frame input: same queries, compare e / d on the common times t in [0, 39],
     plus per-frame depth (t_src = t_tgt = t_cam = t) AbsRel after a per-clip median scale.
  4. frozen-video control (degradation for diagnosis only, §5.1(a)): frame 0 repeated for 40 frames.
  5. combined reward (t0-frame e, d + local-frame e_loc, §3.5) on real and frozen video, and for 48 vs 40 frames.

Each video is encoded once; trajectory, local and depth queries reuse the encoding.
Clips that do not fit in the video (n_frames < stride*(48-1)+1) are skipped and listed in the summary.

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
from anyd4rt.d4rt_critic import encode, load_d4rt, prepare_video, run_queries  # noqa: E402
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
    ap.add_argument("--query-convention", choices=("s1", "d4rt_train"), default="d4rt_train",
                    help="s1: u = x/(W-1); d4rt_train: u = (256/W)*x/255, the OpenD4RT loaders' label convention (B2)")
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


class Critic:
    """One encoded video; all query sets reuse the same encoding."""

    def __init__(self, model, frames, dev):
        self.model = model
        self.video, self.aspect = prepare_video(frames, dev)
        self.memory = encode(model, self.video, self.aspect)
        self.T = frames.shape[0]

    def _q(self, uv, t_src, t_tgt, t_cam):
        return run_queries(self.model, self.video, self.aspect, uv, t_src, t_tgt, t_cam, memory=self.memory)["xyz_3d"]

    def traj(self, uv0, local: bool):
        """§3.5 queries: t_src = t0 = 0, t_tgt = t; t_cam = 0 (t0 frame) or t (local frame)."""
        n, T = uv0.shape[0], self.T
        t_tgt = np.tile(np.arange(T), n)
        return self._q(np.repeat(uv0, T, 0), np.zeros_like(t_tgt), t_tgt, t_tgt if local else np.zeros_like(t_tgt)).reshape(n, T, 3)

    def depth(self, px_t, zc_t, ok_t, frames, rng, m):
        """Per-frame depth: query at the GT pixel of frame t with t_src = t_tgt = t_cam = t."""
        uv, tt, gt = [], [], []
        for t in frames:
            cand = np.flatnonzero(ok_t[t])
            pick = rng.choice(cand, min(m, len(cand)), replace=False)
            uv.append(px_t[t, pick]); tt.append(np.full(len(pick), t)); gt.append(zc_t[t, pick])
        uv, tt, gt = np.concatenate(uv), np.concatenate(tt), np.concatenate(gt)
        return self._q(uv, tt, tt, tt)[:, 2], gt


def absrel(z, gt):
    """Scale-aligned AbsRel over valid predictions. Returns (absrel or None if no valid prediction, valid fraction)."""
    ok = np.isfinite(z) & (z > 0)
    if not ok.any():
        return None, 0.0
    s = np.median(gt[ok] / z[ok])
    return float(np.mean(np.abs(s * z[ok] - gt[ok]) / gt[ok])), float(ok.mean())


DIAG = ("n_invalid_e", "n_invalid_d", "n_invalid_eloc", "frac_capped_e", "frac_capped_d", "frac_capped_eloc")


def summarize_reward(out, dyn_mask):
    res = {k: out.get(k) for k in ("r", "e_mean", "d_mean", "eloc_mean", "s", "scale_valid", "n_V", "n_Vd", "n_Vloc", "n_Q0", "n_Vplus") + DIAG}
    if out.get("scale_valid"):
        V = out["_V"]
        not_t0 = (np.arange(V.shape[1]) != 0)[None]
        Vd, Vl = V & not_t0 & V[:, :1], V & not_t0
        for name, m in (("dyn", dyn_mask), ("sta", ~dyn_mask)):
            mm = m[:, None]
            res[f"e_{name}"] = float(out["e"][V & mm].mean()) if (V & mm).any() else None
            res[f"d_{name}"] = float(out["d"][Vd & mm].mean()) if (Vd & mm).any() else None
            if out.get("e_loc") is not None:
                res[f"eloc_{name}"] = float(out["e_loc"][Vl & mm].mean()) if (Vl & mm).any() else None
    return res


def reward(pred, Y, V, p, dyn, pred_loc=None, Y_loc=None):
    out = reward_d4rt_gt(pred, Y, V, t0=0, p=p, pred_loc=pred_loc, Y_loc=Y_loc)
    out["_V"] = V
    return summarize_reward(out, dyn)


REWARD_KEYS = ["reward_T40", "reward_T48", "reward_T48_common40", "reward_frozen_T40",
               "reward_T40_comb", "reward_T48_common40_comb", "reward_frozen_T40_comb",
               "reward_T40_local", "reward_frozen_T40_local"]
STATS = ("r", "e_mean", "d_mean", "eloc_mean", "e_dyn", "e_sta", "d_dyn", "d_sta", "eloc_dyn", "eloc_sta") + DIAG


def write_summary(rows, out_dir, p, stride, skipped=None):
    def vals(key, sub=None):
        v = [(r.get(key) or {}).get(sub) if sub else r.get(key) for r in rows]
        return [x for x in v if x is not None]

    def mean(key, sub=None):
        v = vals(key, sub)
        return float(np.mean(v)) if v else None

    def paired(key_a, key_b, sub, cmp):
        """Compare two per-clip statistics only where both exist; report how many clips were compared."""
        pairs = [(r[key_a].get(sub), r[key_b].get(sub)) for r in rows if key_a in r and key_b in r]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        return {"frac": float(np.mean([cmp(x, y) for x, y in pairs])) if pairs else None, "n_compared": len(pairs)}

    def absdiff(key_a, key_b, sub):
        d = [abs(r[key_a][sub] - r[key_b][sub]) for r in rows
             if key_a in r and key_b in r and r[key_a].get(sub) is not None and r[key_b].get(sub) is not None]
        return {"mean": float(np.mean(d)), "max": float(np.max(d)), "n_compared": len(d)} if d else None

    summary = {"n_clips": len(rows), "params": p.__dict__, "stride": stride,
               "query_convention": rows[0].get("query_convention") if rows else None, "skipped": skipped or [],
               "proj_err_px_median": mean("proj_err_px_median")}
    for T in (48, 40):
        summary[f"depth_T{T}"] = {"absrel_mean_over_clips_with_valid": mean(f"depth_absrel_T{T}"),
                                  "valid_frac_mean": mean(f"depth_finite_T{T}"),
                                  "n_clips_no_valid_prediction": sum(r.get(f"depth_absrel_T{T}") is None for r in rows)}
    for k in REWARD_KEYS:
        summary[k] = {s: mean(k, s) for s in STATS}
        summary[k]["scale_valid_frac"] = mean(k, "scale_valid")
        summary[k]["n_clips"] = len(vals(k, "r"))
    lt = lambda x, y: x < y  # noqa: E731
    summary["frozen_below_real"] = paired("reward_frozen_T40", "reward_T40", "r", lt)
    summary["frozen_below_real_comb"] = paired("reward_frozen_T40_comb", "reward_T40_comb", "r", lt)
    summary["frozen_static_e_above_real"] = paired("reward_frozen_T40", "reward_T40", "e_sta", lambda x, y: x > y)
    summary["T40_vs_T48_common40"] = {
        "t0_terms": {s: absdiff("reward_T40", "reward_T48_common40", s) for s in ("r", "e_mean", "d_mean")},
        "combined": {s: absdiff("reward_T40_comb", "reward_T48_common40_comb", s) for s in ("r", "e_mean", "d_mean", "eloc_mean")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def clip_starts(n_frames, span, k):
    """Evenly spaced clip starts that fit in the video; [] if the video is shorter than one clip."""
    if n_frames < span:
        return []
    return sorted(set(np.linspace(0, n_frames - span, k).astype(int).tolist()))


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = RewardParams()
    if a.resummarize:
        rows = json.loads((out_dir / "clips.json").read_text())
        skipped_path = out_dir / "skipped.json"
        write_summary(rows, out_dir, p, a.stride, json.loads(skipped_path.read_text()) if skipped_path.exists() else None)
        return
    t_load = time.time()
    model = load_d4rt(a.exp_dir)
    print(f"loaded OpenD4RT in {time.time() - t_load:.0f}s", flush=True)
    dev = next(model.parameters()).device

    scenes = sorted(d for d in Path(a.data_root).iterdir() if d.is_dir() and (d / "anno.npz").exists())[: a.num_scenes]
    span = a.stride * (T_FULL - 1) + 1
    rows, skipped = [], []
    for scene in scenes:
        anno = np.load(scene / "anno.npz", allow_pickle=True)
        n_frames = min(anno["trajs_3d"].shape[0], len(list((scene / "rgbs").iterdir())))
        starts = clip_starts(n_frames, span, a.clips_per_scene)
        if not starts:
            skipped.append({"scene": scene.name, "reason": f"{n_frames} frames < clip span {span}"})
            print(f"skip {scene.name}: {n_frames} frames < span {span}", flush=True)
            continue
        for start in starts:
            t_clip = time.time()
            idx = start + a.stride * np.arange(T_FULL)
            assert idx[0] >= 0 and idx[-1] < n_frames, (scene.name, start)
            X = anno["trajs_3d"][idx].astype(np.float64)  # [T,N,3] world
            vis, val = anno["visibs"][idx], anno["valids"][idx]
            K, T_cw = anno["intrinsics"][idx].astype(np.float64), anno["extrinsics"][idx].astype(np.float64)
            frames = load_frames(scene, idx)
            H, W = frames.shape[1:3]

            Xc_t = transform(T_cw[:, None], X)  # each point in its own frame's camera
            px_t, zc_t = project(K[:, None], Xc_t)
            gt_ok = val & vis & in_frame(px_t, (W, H)) & (zc_t > 0)
            proj_err = np.linalg.norm(px_t - anno["trajs_2d"][idx], axis=-1)[val & vis]

            # Anchors at t0 = 0; §3.5 references: Y in cam(t0), Y_loc in cam(t).
            motion = np.nanmax(np.linalg.norm(X - X[:1], axis=-1), axis=0)
            dynamic = motion > a.dyn_thresh * np.maximum(zc_t[0], 1e-3)
            ok0 = gt_ok[0] & np.isfinite(X).all(-1).all(0)
            if ok0.sum() < 64:
                skipped.append({"scene": scene.name, "start": int(start), "reason": f"{int(ok0.sum())} anchors"})
                print(f"skip {scene.name}@{start}: {ok0.sum()} anchors", flush=True)
                continue
            ids = sample_anchors(rng, ok0, dynamic, a.num_anchors)
            Y = transform(T_cw[0], X[:, ids]).transpose(1, 0, 2)  # [N,T,3]
            Y_loc = Xc_t[:, ids].transpose(1, 0, 2)  # Y_loc(i,t) = T_cw(t) X_GT(i,t)
            V = gt_ok[:, ids].T  # GT-only (C_b = C_a: target visibility = GT visibs)
            dyn = dynamic[ids]
            if a.query_convention == "s1":
                uv0, uv_t = uv_normalize(px_t[0, ids], (W, H)), uv_normalize(px_t, (W, H))
            else:  # K scaled multiplicatively to the 256 grid, then normalized by (256 - 1)
                to_in = np.array([256.0 / W, 256.0 / H])
                uv0, uv_t = uv_normalize(px_t[0, ids] * to_in, (256, 256)), uv_normalize(px_t * to_in, (256, 256))
            c40 = slice(0, T_INTER)

            row = {"scene": scene.name, "start": int(start), "stride": a.stride, "query_convention": a.query_convention, "n_anchor": len(ids), "n_dyn": int(dyn.sum()),
                   "proj_err_px_median": float(np.median(proj_err)), "proj_err_px_p99": float(np.percentile(proj_err, 99))}
            depth_frames = np.arange(0, T_INTER, 4)
            for tag, clip in (("T48", frames), ("T40", frames[:T_INTER]), ("frozen_T40", np.repeat(frames[:1], T_INTER, 0))):
                cr = Critic(model, clip, dev)
                pr, pl = cr.traj(uv0, local=False), cr.traj(uv0, local=True)
                Tn = cr.T
                row[f"reward_{tag}"] = reward(pr, Y[:, :Tn], V[:, :Tn], p, dyn)
                row[f"reward_{tag}_local"] = reward(pl, Y_loc[:, :Tn], V[:, :Tn], p, dyn)  # everything in the local frame (reference)
                if tag == "T48":  # 48-frame input, evaluated on the common times [0, 39]
                    row["reward_T48_common40"] = reward(pr[:, c40], Y[:, c40], V[:, c40], p, dyn)
                    row["reward_T48_common40_comb"] = reward(pr[:, c40], Y[:, c40], V[:, c40], p, dyn, pl[:, c40], Y_loc[:, c40])
                else:
                    row[f"reward_{tag}_comb"] = reward(pr, Y[:, :Tn], V[:, :Tn], p, dyn, pl, Y_loc[:, :Tn])
                if tag != "frozen_T40":
                    z, gt = cr.depth(uv_t, zc_t, gt_ok, depth_frames, np.random.default_rng(a.seed), a.depth_points)
                    row[f"depth_absrel_{tag}"], row[f"depth_finite_{tag}"] = absrel(z, gt)
                del cr
            row["sec"] = round(time.time() - t_clip, 1)
            rows.append(row)
            rc, r48, rcf = row["reward_T40_comb"], row["reward_T48_common40_comb"], row["reward_frozen_T40_comb"]
            fmt = lambda x: "None" if x is None else f"{x:.3f}"  # noqa: E731
            print(f"{scene.name}@{start}: proj {row['proj_err_px_median']:.2f}px | comb r40 {fmt(rc['r'])} r48c {fmt(r48['r'])} "
                  f"frozen {fmt(rcf['r'])} | depth48 {fmt(row['depth_absrel_T48'])} depth40 {fmt(row['depth_absrel_T40'])} | {row['sec']}s", flush=True)
            (out_dir / "clips.json").write_text(json.dumps(rows, indent=1))
            (out_dir / "skipped.json").write_text(json.dumps(skipped, indent=1))

    (out_dir / "skipped.json").write_text(json.dumps(skipped, indent=1))
    write_summary(rows, out_dir, p, a.stride, skipped)


if __name__ == "__main__":
    main()
