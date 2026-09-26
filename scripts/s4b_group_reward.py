#!/usr/bin/env python3
"""S4 on PointOdyssey: is the within-group reward ranking trustworthy? (README §6 S4, first pass)

Per clip and target angle theta (0 = same camera, then a small nonzero baseline, README §7.2 orbit):
  fixed for the whole group: target cameras C_b, anchors, GT references, the valid set V (z-buffer visibility from
  the source depth, fixed version), the critic, the query chain (generated grid -> 256^2; native pixel -> grid by
  the pixel-center rule measured in s4a; OpenD4RT training query convention); only the seed changes.
  G candidates (35-step SDE, a = --a, seed g for both starting and per-step noise), combined reward (§3.5).
Reports: reward spread within the group; ranking stability = Kendall tau between the rankings computed on two
random halves of the anchors (mean over --splits splits); per-term breakdown (e, d, e_loc; static / dynamic) of the
best and worst candidate; frames of input / best / worst. PSNR vs the input is reported only at theta = 0 and only
as an auxiliary signal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
AV = ROOT / "third_party" / "AnyView-DVS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AV))

from anyd4rt import po_episode  # noqa: E402
from anyd4rt.anyview_rl import rollout  # noqa: E402
from anyd4rt.d4rt_critic import encode, load_d4rt, prepare_video, run_queries  # noqa: E402
from anyd4rt.geometry import VIS_VISIBLE, in_frame, transform, zbuffer  # noqa: E402
from anyd4rt.reward_d4rt_gt import RewardParams, reward_d4rt_gt  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("s4a", ROOT / "scripts" / "s4a_po_same_view.py")
s4a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s4a)
b4 = s4a.b4
T_D4RT = 40


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/tmp/kwang-mnt/pointodyssey/v2/val")
    ap.add_argument("--scene", default="cnb_dlab_0215_ego2")
    ap.add_argument("--thetas", default="0,10")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--a", type=float, default=0.7)
    ap.add_argument("--num-anchors", type=int, default=1024)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--exp-dir", default=str(ROOT / "third_party/Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "s4b"))
    return ap.parse_args()


def kendall_tau(x, y):
    """Standard Kendall tau-b, including correction for ties."""
    result = kendalltau(np.asarray(x), np.asarray(y), variant="b", nan_policy="omit")
    return float(result.statistic) if np.isfinite(result.statistic) else 0.0


def predict(model, frames_grid, uv0):
    gh, gw = frames_grid.shape[1:3]
    video, aspect = prepare_video(frames_grid[:T_D4RT], "cuda")
    mem = encode(model, video, aspect)
    n, T = len(uv0), T_D4RT
    t_tgt = np.tile(np.arange(T), n)
    uv = np.repeat(uv0, T, 0)
    pr = run_queries(model, video, aspect, uv, np.zeros_like(t_tgt), t_tgt, np.zeros_like(t_tgt), memory=mem)["xyz_3d"].reshape(n, T, 3)
    pl = run_queries(model, video, aspect, uv, np.zeros_like(t_tgt), t_tgt, t_tgt, memory=mem)["xyz_3d"].reshape(n, T, 3)
    return pr, pl


def terms(out, V, dyn):
    res = {k: out.get(k) for k in ("r", "e_mean", "d_mean", "eloc_mean", "s", "scale_valid", "frac_capped_e", "frac_capped_eloc")}
    if out.get("scale_valid"):
        nt = (np.arange(V.shape[1]) != 0)[None]
        for nm, m in (("dyn", dyn[:, None]), ("sta", ~dyn[:, None])):
            res[f"e_{nm}"] = float(out["e"][V & m].mean()) if (V & m).any() else None
            res[f"eloc_{nm}"] = float(out["e_loc"][V & nt & m].mean()) if (V & nt & m).any() else None
    return res


def main():
    a = parse_args()
    from anyview.config import AnyViewConfig
    from anyview.pipe import load_pipeline
    from anyview.vae import load_vae

    device = torch.device("cuda")
    config = AnyViewConfig(checkpoint_path=str(AV / "checkpoints/anyview_dvs_2b.pt"), tokenizer_path=str(AV / "checkpoints/tokenizer.pth"))
    vae = load_vae(config.tokenizer_path, device)
    pipe = load_pipeline(config, device)
    critic = load_d4rt(a.exp_dir)
    p = RewardParams()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clip = po_episode.load_clip(Path(a.data_root) / a.scene, 0, a.stride)
    H, W = clip["frames"].shape[1:3]
    hw = b4.infer_lib.snap_shape(H, W)
    gh, gw = hw
    real_grid = s4a.to_uint8(b4.infer_lib.resize_video(clip["frames"], tuple(hw)).permute(1, 0, 2, 3))
    rows = []
    for theta in [float(x) for x in a.thetas.split(",")]:
        t0 = time.time()
        T_b = po_episode.target_trajectory(clip, theta)
        zbs = [zbuffer(clip["depth"][t], clip["K"], clip["T_cw"][t], clip["K"], T_b[t], (W, H)) for t in range(len(T_b))]
        px, z, state = po_episode.target_view_gt(clip, T_b, lambda t: zbs[t])
        V_all = clip["val"] & in_frame(px, (W, H)) & (z > 0) & (state == VIS_VISIBLE)
        ok0 = V_all[0] & np.isfinite(clip["X"]).all(-1).all(0)
        motion = np.nanmax(np.linalg.norm(clip["X"] - clip["X"][:1], axis=-1), axis=0)
        dyn_all = motion > 0.02 * np.maximum(z[0], 1e-3)
        ids = s4a.b4_sample(np.random.default_rng(0), ok0, dyn_all, a.num_anchors)
        dyn = dyn_all[ids]
        Y = transform(T_b[0], clip["X"][:, ids]).transpose(1, 0, 2)[:, :T_D4RT]
        Y_loc = transform(T_b[:, None], clip["X"][:, ids]).transpose(1, 0, 2)[:, :T_D4RT]
        V = V_all[:, ids].T[:, :T_D4RT]
        xg = s4a.grid_px(px[0, ids], (W, H), (gw, gh), "pc")  # pixel-center rule (measured, s4a)
        uv0 = xg * np.array([256.0 / gw, 256.0 / gh]) / 255.0  # OpenD4RT training query convention
        vis_frac = {"anchor_candidates_visible_t0": int(ok0.sum()), "V_frac_of_anchor_times": float(V.mean())}

        item = po_episode.av_item(clip, T_b)
        with torch.no_grad():
            entries, _ = b4.make_entries(item, vae, config, device)
        cands, gens, preds = [], [], []
        for g in range(a.G):
            r = rollout.sample(pipe, entries, seed=g, num_steps=35, mode="sde", a=a.a, noise_seed=g)
            gen = s4a.to_uint8(b4.decode(vae, r.y0_pred_streams))
            pr, pl = predict(critic, gen, uv0)
            o = reward_d4rt_gt(pr, Y, V, 0, p, pl, Y_loc)
            c = {"seed": g, **terms(o, V, dyn)}
            if theta == 0:
                c["psnr_vs_input_aux"] = float(np.mean([cv2.PSNR(x, y) for x, y in zip(gen, real_grid)]))
            cands.append(c); gens.append(gen); preds.append((pr, pl))
        rs = np.array([c["r"] for c in cands])
        # ranking stability over random anchor halves
        rng = np.random.default_rng(1)
        taus = []
        for _ in range(a.splits):
            perm = rng.permutation(len(ids))
            h1, h2 = perm[: len(ids) // 2], perm[len(ids) // 2:]
            r1 = [reward_d4rt_gt(pr[h1], Y[h1], V[h1], 0, p, pl[h1], Y_loc[h1])["r"] for pr, pl in preds]
            r2 = [reward_d4rt_gt(pr[h2], Y[h2], V[h2], 0, p, pl[h2], Y_loc[h2])["r"] for pr, pl in preds]
            taus.append(kendall_tau(r1, r2))
        best, worst = int(np.argmax(rs)), int(np.argmin(rs))
        row = {"scene": a.scene, "theta": theta, "n_anchor": len(ids), "n_dyn": int(dyn.sum()), **vis_frac,
               "reward": {"mean": float(rs.mean()), "std": float(rs.std(ddof=1)), "min": float(rs.min()), "max": float(rs.max())},
               "kendall_tau_halves": {"mean": float(np.mean(taus)), "min": float(np.min(taus)), "max": float(np.max(taus))},
               "best": cands[best], "worst": cands[worst], "candidates": cands, "sec": round(time.time() - t0, 1)}
        (out / f"{a.scene}.json").write_text(json.dumps(rows + [row], indent=1))  # results before any plotting
        ts = (0, 20, 39)
        grid = [np.ascontiguousarray(np.concatenate([v[t] for t in ts], 1)).copy() for v in (real_grid, gens[best], gens[worst])]
        for im, lab in zip(grid, ("input (source camera)", f"best seed {best} r={rs[best]:.3f}", f"worst seed {worst} r={rs[worst]:.3f}")):
            cv2.putText(im, f"theta={theta:g}: {lab}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imwrite(str(out / f"{a.scene}_theta{theta:g}_best_worst.jpg"), cv2.cvtColor(np.concatenate(grid, 0), cv2.COLOR_RGB2BGR))
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("scene", "theta", "n_anchor", "anchor_candidates_visible_t0", "V_frac_of_anchor_times", "reward", "kendall_tau_halves", "sec")}), flush=True)
        (out / f"{a.scene}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
