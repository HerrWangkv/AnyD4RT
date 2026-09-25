#!/usr/bin/env python3
"""Critic-free geometric check of generated candidates (S4 independent reference, diagnostic).

For GT points visible in the source frame and in the target view (z-buffer), the candidate should show, at the point's
projected target position, the colour the point has in the source video. Per frame, search the 2D shift (dx, dy) of the
projected positions within +-R grid px that minimizes the colour difference; the best shift is how far that part of
the picture is misplaced. Reported per group (static points / person points): median |shift| over frames (grid px),
colour difference at zero shift and at the best shift. theta = 0 candidates calibrate the method (should be ~0 px).
The person mask is used only to split the groups for this diagnosis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AV = ROOT / "third_party" / "AnyView-DVS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AV))

from anyd4rt import po_episode  # noqa: E402
from anyd4rt.anyview_rl import rollout  # noqa: E402
from anyd4rt.geometry import VIS_VISIBLE, in_frame, project, transform, zbuffer  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("s4b", ROOT / "scripts" / "s4b_group_reward.py")
s4b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s4b)
s4a, b4 = s4b.s4a, s4b.s4a.b4
T_D4RT = 40


def norm(v):
    v = v.astype(np.float64)
    return (v - v.mean(axis=(0, 1, 2))) / (v.std(axis=(0, 1, 2)) + 1e-6)


def best_shift(S, G, src_px, tgt_px, R, step=2):
    """S, G: normalized [H,W,3] source/candidate frames at the grid; src_px, tgt_px: [n,2] grid pixels.
    Returns (dx, dy) minimizing the mean |colour diff|, the diff at (0,0) and at the best shift."""
    gh, gw = G.shape[:2]
    def box(img, p):
        x = np.clip(np.round(p[:, 0]).astype(int), 1, gw - 2); y = np.clip(np.round(p[:, 1]).astype(int), 1, gh - 2)
        return sum(img[y + dy, x + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    cs = box(S, src_px)
    best, d0 = (0, 0, np.inf), None
    for dy in range(-R, R + 1, step):
        for dx in range(-R, R + 1, step):
            d = float(np.abs(box(G, tgt_px + np.array([dx, dy])) - cs).mean())
            if dx == 0 and dy == 0:
                d0 = d
            if d < best[2]:
                best = (dx, dy, d)
    return best, d0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="cnb_dlab_0215_ego2")
    ap.add_argument("--theta", type=float, default=10.0)
    ap.add_argument("--seeds", default="14,11,22,1,4,5,7,0,2")
    ap.add_argument("--person-id", type=int, default=62)
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "s4e"))
    a = ap.parse_args()
    from anyview.config import AnyViewConfig
    from anyview.pipe import load_pipeline
    from anyview.vae import load_vae

    device = torch.device("cuda")
    config = AnyViewConfig(checkpoint_path=str(AV / "checkpoints/anyview_dvs_2b.pt"), tokenizer_path=str(AV / "checkpoints/tokenizer.pth"))
    vae = load_vae(config.tokenizer_path, device)
    pipe = load_pipeline(config, device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = Path("/tmp/kwang-mnt/pointodyssey/v2/val") / a.scene
    clip = po_episode.load_clip(scene, 0, 2)
    H, W = clip["frames"].shape[1:3]
    gh, gw = b4.infer_lib.snap_shape(H, W)
    real = s4a.to_uint8(b4.infer_lib.resize_video(clip["frames"], (gh, gw)).permute(1, 0, 2, 3))
    T_b = po_episode.target_trajectory(clip, a.theta)
    zbs = [zbuffer(clip["depth"][t], clip["K"], clip["T_cw"][t], clip["K"], T_b[t], (W, H)) for t in range(len(T_b))]
    px_t, z_t, st = po_episode.target_view_gt(clip, T_b, lambda t: zbs[t])
    px_s, _ = project(clip["K"][None, None], transform(clip["T_cw"][:, None], clip["X"]))
    motion = np.nanmax(np.linalg.norm(clip["X"] - clip["X"][:1], axis=-1), axis=0)
    with torch.no_grad():
        entries, _ = b4.make_entries(po_episode.av_item(clip, T_b), vae, config, device)
    frames_t = list(range(0, T_D4RT, 3))
    rng = np.random.default_rng(0)
    groups_t = []
    for t in frames_t:
        m = np.array(Image.open(scene / f"masks/mask_{clip['idx'][t]:05d}.png"))
        ok = (clip["vis"][t] & clip["val"][t] & in_frame(px_s[t], (W, H)) & in_frame(px_t[t], (W, H)) & (z_t[t] > 0) & (st[t] == VIS_VISIBLE))
        q = np.round(px_s[t][ok]).astype(int)
        is_p = np.zeros(len(ok), bool)
        is_p[np.flatnonzero(ok)] = m[q[:, 1], q[:, 0]] == a.person_id
        g = {}
        for name, sel in (("person", ok & is_p), ("static", ok & ~is_p & (motion < 0.01))):
            idx = np.flatnonzero(sel)
            idx = rng.choice(idx, min(400, len(idx)), replace=False) if len(idx) else idx
            g[name] = (s4a.grid_px(px_s[t][idx], (W, H), (gw, gh), "pc"), s4a.grid_px(px_t[t][idx], (W, H), (gw, gh), "pc"))
        groups_t.append(g)
    Sn = norm(real[:T_D4RT])
    rows = []
    for seed in [int(x) for x in a.seeds.split(",")]:
        r = rollout.sample(pipe, entries, seed=seed, num_steps=35, mode="sde", a=0.7, noise_seed=seed)
        gen = s4a.to_uint8(b4.decode(vae, r.y0_pred_streams))[:T_D4RT]
        Gn = norm(gen)
        row = {"seed": seed}
        for name in ("static", "person"):
            shifts, d0s, dbs = [], [], []
            for t, g in zip(frames_t, groups_t):
                sp, tp = g[name]
                if len(sp) < 20:
                    continue
                (dx, dy, db), d0 = best_shift(Sn[t], Gn[t], sp, tp, a.R)
                shifts.append(np.hypot(dx, dy)); d0s.append(d0); dbs.append(db)
            row[name] = {"median_shift_px": float(np.median(shifts)), "mean_shift_px": float(np.mean(shifts)),
                         "colour_diff_at_0": float(np.mean(d0s)), "colour_diff_at_best": float(np.mean(dbs)), "n_frames": len(shifts)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        (out / f"{a.scene}_theta{a.theta:g}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
