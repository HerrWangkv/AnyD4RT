#!/usr/bin/env python3
"""Cache one S4 group (scene, theta, seeds) for offline reward analysis (source-side gating, aggregation variants).

Group setup is exactly s4b's: target cameras, anchors (rng 0), V (z-buffer), GT references, query chain. Saved:
  group:   ids, dyn (GT motion), V, Y, Y_loc, uv0 (target), px_t_grid / px_s_grid [T,N,2] (grid pixels in target /
           source view, for the critic-free alignment reference), src_ok [T,N] (GT-visible in the source view)
  source:  the REAL source video through the same grid -> 256 chain, OpenD4RT queried at each anchor's SOURCE pixel
           at t0 (source camera): pr_src, pl_src, with Y_src (cam_a(t0)), Y_loc_src (cam_a(t)), V_src (GT valid &
           GT-visible in the source & in frame), src_vis0 (anchor GT-visible in the source at t0)
  per seed: candidate video (grid uint8, 41 frames), pr, pl (OpenD4RT on the candidate, target queries)
Optionally person flags from the instance mask (--person-id; diagnostic only, never used for scoring).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AV = ROOT / "third_party" / "AnyView-DVS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AV))

from anyd4rt import po_episode  # noqa: E402
from anyd4rt.anyview_rl import rollout  # noqa: E402
from anyd4rt.d4rt_critic import load_d4rt  # noqa: E402
from anyd4rt.geometry import VIS_VISIBLE, in_frame, project, transform, zbuffer  # noqa: E402
from anyd4rt.reward_d4rt_gt import RewardParams, reward_d4rt_gt  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("s4b", ROOT / "scripts" / "s4b_group_reward.py")
s4b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s4b)
s4a, b4 = s4b.s4a, s4b.s4a.b4
T_D4RT = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--theta", type=float, required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--person-id", type=int, default=-1)
    ap.add_argument("--init-seed", type=int, default=-1,
                    help=">= 0: fix the starting latent to this seed and vary only the SDE noise (cand_XX = noise seed XX)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "s4cache"))
    a = ap.parse_args()
    from anyview.config import AnyViewConfig
    from anyview.pipe import load_pipeline
    from anyview.vae import load_vae

    device = torch.device("cuda")
    config = AnyViewConfig(checkpoint_path=str(AV / "checkpoints/anyview_dvs_2b.pt"), tokenizer_path=str(AV / "checkpoints/tokenizer.pth"))
    vae = load_vae(config.tokenizer_path, device)
    pipe = load_pipeline(config, device)
    critic = load_d4rt(str(ROOT / "third_party/Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG"))
    p = RewardParams()
    gdir = Path(a.out) / (f"{a.scene}@{a.start}_theta{a.theta:g}" + (f"_init{a.init_seed}" if a.init_seed >= 0 else ""))
    gdir.mkdir(parents=True, exist_ok=True)
    scene = Path("/tmp/kwang-mnt/pointodyssey/v2/val") / a.scene
    clip = po_episode.load_clip(scene, a.start, 2)
    H, W = clip["frames"].shape[1:3]
    gh, gw = b4.infer_lib.snap_shape(H, W)
    real = s4a.to_uint8(b4.infer_lib.resize_video(clip["frames"], (gh, gw)).permute(1, 0, 2, 3))

    if not (gdir / "group.npz").exists():
        T_b = po_episode.target_trajectory(clip, a.theta)
        zbs = [zbuffer(clip["depth"][t], clip["K"], clip["T_cw"][t], clip["K"], T_b[t], (W, H)) for t in range(len(T_b))]
        px, z, state = po_episode.target_view_gt(clip, T_b, lambda t: zbs[t])
        V_all = clip["val"] & in_frame(px, (W, H)) & (z > 0) & (state == VIS_VISIBLE)
        ok0 = V_all[0] & np.isfinite(clip["X"]).all(-1).all(0)
        motion = np.nanmax(np.linalg.norm(clip["X"] - clip["X"][:1], axis=-1), axis=0)
        dyn_all = motion > 0.02 * np.maximum(z[0], 1e-3)
        ids = s4a.b4_sample(np.random.default_rng(0), ok0, dyn_all, 1024)
        X = clip["X"][:, ids]
        Y = transform(T_b[0], X).transpose(1, 0, 2)[:, :T_D4RT]
        Y_loc = transform(T_b[:, None], X).transpose(1, 0, 2)[:, :T_D4RT]
        V = V_all[:, ids].T[:, :T_D4RT]
        px_t_grid = s4a.grid_px(px[:T_D4RT][:, ids], (W, H), (gw, gh), "pc")
        uv0 = px_t_grid[0] * np.array([256.0 / gw, 256.0 / gh]) / 255.0
        # source side: source camera, GT visibility in the source
        T_a = clip["T_cw"]
        px_s, z_s = project(clip["K"][None, None], transform(T_a[:, None], X))
        src_ok = clip["vis"][:, ids] & clip["val"][:, ids] & in_frame(px_s, (W, H)) & (z_s > 0)
        V_src = src_ok.T[:, :T_D4RT]
        px_s_grid = s4a.grid_px(px_s[:T_D4RT], (W, H), (gw, gh), "pc")
        uv0_src = px_s_grid[0] * np.array([256.0 / gw, 256.0 / gh]) / 255.0
        Y_src = transform(T_a[0], X).transpose(1, 0, 2)[:, :T_D4RT]
        Y_loc_src = transform(T_a[:, None], X).transpose(1, 0, 2)[:, :T_D4RT]
        pr_src, pl_src = s4b.predict(critic, real, uv0_src)
        person = np.zeros(len(ids), bool)
        if a.person_id >= 0:
            decided = np.zeros(len(ids), bool)
            for t in range(len(clip["idx"])):
                m = np.array(Image.open(scene / f"masks/mask_{clip['idx'][t]:05d}.png"))
                okt = clip["vis"][t, ids] & clip["val"][t, ids] & in_frame(px_s[t], (W, H)) & ~decided
                q = np.round(px_s[t][okt]).astype(int)
                person[np.flatnonzero(okt)] = m[q[:, 1], q[:, 0]] == a.person_id
                decided |= okt
        np.savez_compressed(gdir / "group.npz", ids=ids, dyn=dyn_all[ids], V=V, Y=Y, Y_loc=Y_loc, uv0=uv0,
                            px_t_grid=px_t_grid, px_s_grid=px_s_grid, src_ok=src_ok[:T_D4RT], V_src=V_src,
                            uv0_src=uv0_src, Y_src=Y_src, Y_loc_src=Y_loc_src, pr_src=pr_src, pl_src=pl_src,
                            src_vis0=V_src[:, 0], person=person, motion=motion[ids], T_b=T_b, real=real)
        (gdir / "meta.json").write_text(json.dumps({"scene": a.scene, "start": a.start, "theta": a.theta, "grid_hw": [gh, gw],
                                                    "person_id": a.person_id, "n_anchor_pool_t0": int(ok0.sum())}))
    g = np.load(gdir / "group.npz")
    T_b = g["T_b"]
    item = po_episode.av_item(clip, T_b)
    with torch.no_grad():
        entries, _ = b4.make_entries(item, vae, config, device)
    for seed in [int(x) for x in a.seeds.split(",")]:
        f = gdir / f"cand_{seed:02d}.npz"
        if f.exists():
            continue
        t0 = time.time()
        init = a.init_seed if a.init_seed >= 0 else seed
        r = rollout.sample(pipe, entries, seed=init, num_steps=35, mode="sde", a=0.7, noise_seed=seed)
        gen = s4a.to_uint8(b4.decode(vae, r.y0_pred_streams))
        pr, pl = s4b.predict(critic, gen, g["uv0"])
        rr = reward_d4rt_gt(pr, g["Y"], g["V"], 0, p, pl, g["Y_loc"])["r"]
        np.savez_compressed(f, video=gen, pr=pr, pl=pl, r=rr, init_seed=init, noise_seed=seed)
        print(json.dumps({"group": gdir.name, "init_seed": init, "noise_seed": seed, "r": rr, "sec": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
