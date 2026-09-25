#!/usr/bin/env python3
"""S4 diagnosis of the "deleted person scores highest" case (cnb_dlab_0215_ego2, theta = 10 deg).

Exactly the s4b setup (same anchors, V, query chain, critic, seeds). The person's instance mask (PointOdyssey
masks/, id found in the source view) is used ONLY here, for diagnosis, never for scoring.
  Q1 per candidate: e / d / e_loc on person anchors, other dynamic anchors and static anchors; GT and predicted
     person motion (in cam_b(t0), metres and relative to depth); fitted scale; each group's share of the reward.
  Q2 person coverage: person GT points visible at t0 in the source, visible at t0 in the target (z-buffer),
     sampled as anchors, and per-time fraction of person anchors in V; z-buffer states of person points over time.
  Q3 presence/placement (diagnostic only): colour consistency between each GT point's colour in the source video
     and the candidate's colour at the point's projected target position, per group (person / static); plus a
     montage of every candidate with the projected person points. --extra-seeds adds seeds for this condition.
  I  interface: the same colour check on static points vs AnyView's scale factor (--scales) for 2 seeds.
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AV = ROOT / "third_party" / "AnyView-DVS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AV))

from anyd4rt import po_episode  # noqa: E402
from anyd4rt.anyview_rl import rollout  # noqa: E402
from anyd4rt.d4rt_critic import load_d4rt  # noqa: E402
from anyd4rt.geometry import VIS_OCCLUDED, VIS_UNKNOWN, VIS_VISIBLE, in_frame, project, transform, zbuffer  # noqa: E402
from anyd4rt.reward_d4rt_gt import RewardParams, reward_d4rt_gt  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("s4b", ROOT / "scripts" / "s4b_group_reward.py")
s4b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s4b)
s4a, b4 = s4b.s4a, s4b.s4a.b4
T_D4RT = 40


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="cnb_dlab_0215_ego2")
    ap.add_argument("--theta", type=float, default=10.0)
    ap.add_argument("--person-id", type=int, default=62)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--extra-seeds", type=int, default=0)
    ap.add_argument("--scales", default="0.25,0.125,0.0625")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "s4c"))
    return ap.parse_args()


def colour_consistency(src_grid, gen, px_src_g, px_tgt_g, mask):
    """Median over (i, t) in mask of |mean-normalized RGB difference| between the source colour at the point's source
    grid pixel and the candidate colour at its target grid pixel (3x3 box mean, per-video mean/std normalization)."""
    def norm(v):
        v = v.astype(np.float64)
        return (v - v.mean(axis=(0, 1, 2))) / (v.std(axis=(0, 1, 2)) + 1e-6)
    S, G = norm(src_grid), norm(gen)
    ti, ii = np.nonzero(mask)
    gh, gw = gen.shape[1:3]
    def sample(v, p):
        x = np.clip(np.round(p[..., 0]).astype(int), 1, gw - 2)
        y = np.clip(np.round(p[..., 1]).astype(int), 1, gh - 2)
        acc = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc = acc + v[ti, y + dy, x + dx]
        return acc / 9.0
    d = np.abs(sample(S, px_src_g[ti, ii]) - sample(G, px_tgt_g[ti, ii])).mean(-1)
    return float(np.median(d)) if len(d) else None


def main():
    a = parse_args()
    from anyview.config import AnyViewConfig
    from anyview.pipe import load_pipeline
    from anyview.vae import load_vae

    device = torch.device("cuda")
    config = AnyViewConfig(checkpoint_path=str(AV / "checkpoints/anyview_dvs_2b.pt"), tokenizer_path=str(AV / "checkpoints/tokenizer.pth"))
    vae = load_vae(config.tokenizer_path, device)
    pipe = load_pipeline(config, device)
    critic = load_d4rt(str(ROOT / "third_party/Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG"))
    p = RewardParams()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = Path("/tmp/kwang-mnt/pointodyssey/v2/val") / a.scene
    clip = po_episode.load_clip(scene, 0, 2)
    H, W = clip["frames"].shape[1:3]
    gh, gw = b4.infer_lib.snap_shape(H, W)
    real_grid = s4a.to_uint8(b4.infer_lib.resize_video(clip["frames"], (gh, gw)).permute(1, 0, 2, 3))

    # ---- identical to s4b ----
    T_b = po_episode.target_trajectory(clip, a.theta)
    zbs = [zbuffer(clip["depth"][t], clip["K"], clip["T_cw"][t], clip["K"], T_b[t], (W, H)) for t in range(len(T_b))]
    px, z, state = po_episode.target_view_gt(clip, T_b, lambda t: zbs[t])
    V_all = clip["val"] & in_frame(px, (W, H)) & (z > 0) & (state == VIS_VISIBLE)
    ok0 = V_all[0] & np.isfinite(clip["X"]).all(-1).all(0)
    motion = np.nanmax(np.linalg.norm(clip["X"] - clip["X"][:1], axis=-1), axis=0)
    dyn_all = motion > 0.02 * np.maximum(z[0], 1e-3)
    ids = s4a.b4_sample(np.random.default_rng(0), ok0, dyn_all, 1024)
    dyn = dyn_all[ids]
    Y = transform(T_b[0], clip["X"][:, ids]).transpose(1, 0, 2)[:, :T_D4RT]
    Y_loc = transform(T_b[:, None], clip["X"][:, ids]).transpose(1, 0, 2)[:, :T_D4RT]
    V = V_all[:, ids].T[:, :T_D4RT]
    uv0 = s4a.grid_px(px[0, ids], (W, H), (gw, gh), "pc") * np.array([256.0 / gw, 256.0 / gh]) / 255.0

    # ---- person membership (diagnostic only): mask id at the point's SOURCE pixel at the first frame it is GT-visible
    px_s, z_s = project(clip["K"][None, None], transform(clip["T_cw"][:, None], clip["X"]))
    person = np.zeros(clip["X"].shape[1], bool)
    decided = np.zeros_like(person)
    for t in range(len(clip["idx"])):
        m = np.array(Image.open(scene / f"masks/mask_{clip['idx'][t]:05d}.png"))
        okt = clip["vis"][t] & clip["val"][t] & in_frame(px_s[t], (W, H)) & ~decided
        q = np.round(px_s[t][okt]).astype(int)
        person[np.flatnonzero(okt)] = m[q[:, 1], q[:, 0]] == a.person_id
        decided |= okt
    pa = person[ids]
    groups = {"person": pa, "other_dyn": dyn & ~pa, "static": ~dyn & ~pa}

    # ---- Q2 coverage
    src_vis0 = clip["vis"][0] & clip["val"][0] & in_frame(px_s[0], (W, H))
    per_t_person_V = V[pa].mean(0) if pa.any() else np.zeros(T_D4RT)
    st_person = state[:T_D4RT][:, ids[pa]] if pa.any() else np.zeros((T_D4RT, 0))
    q2 = {"person_gt_points_total": int(person.sum()), "person_visible_t0_source": int((person & src_vis0).sum()),
          "person_visible_t0_target_zbuffer": int((person & ok0).sum()),
          "anchor_pool_visible_t0_target": int(ok0.sum()), "anchors": len(ids), "person_anchors": int(pa.sum()),
          "person_frac_of_anchors": float(pa.mean()), "dyn_anchors": int(dyn.sum()), "person_frac_of_dyn_anchors": float(pa[dyn].mean()) if dyn.any() else None,
          "person_anchor_times_in_V_frac": float(V[pa].mean()) if pa.any() else None,
          "person_frac_of_V": float(V[pa].sum() / V.sum()),
          "person_V_frac_per_t_[0,10,20,30,39]": [float(per_t_person_V[t]) for t in (0, 10, 20, 30, 39)],
          "person_zbuffer_states_over_t_{visible,occluded,unknown}": [float((st_person == s).mean()) for s in (VIS_VISIBLE, VIS_OCCLUDED, VIS_UNKNOWN)],
          "person_gt_motion_over_clip_median_m": float(np.median(motion[person])) if person.any() else None}

    # grid pixels for the colour check: source view (source camera) and target view (target camera)
    src_g = s4a.grid_px(px_s[:T_D4RT][:, ids], (W, H), (gw, gh), "pc")
    tgt_g = s4a.grid_px(px[:T_D4RT][:, ids], (W, H), (gw, gh), "pc")
    src_ok = (clip["vis"][:T_D4RT][:, ids] & clip["val"][:T_D4RT][:, ids]).astype(bool)
    colour_mask = {g: src_ok & V.T & m[None] for g, m in groups.items()}

    item = po_episode.av_item(clip, T_b)
    with torch.no_grad():
        entries, _ = b4.make_entries(item, vae, config, device)
    nt = (np.arange(T_D4RT) != 0)[None]

    def analyse(seed, ent, tag):
        r = rollout.sample(pipe, ent, seed=seed, num_steps=35, mode="sde", a=0.7, noise_seed=seed)
        gen = s4a.to_uint8(b4.decode(vae, r.y0_pred_streams))
        pr, pl = s4b.predict(critic, gen, uv0)
        o = reward_d4rt_gt(pr, Y, V, 0, p, pl, Y_loc)
        row = {"seed": seed, "tag": tag, "r": o["r"], "s": o["s"], "e_mean": o["e_mean"], "d_mean": o["d_mean"], "eloc_mean": o["eloc_mean"]}
        nV, nVd, nVl = V.sum(), (V & nt & V[:, :1]).sum(), (V & nt).sum()
        for g, m in groups.items():
            mm = m[:, None]
            Vg, Vdg, Vlg = V & mm, V & nt & V[:, :1] & mm, V & nt & mm
            row[g] = {"n_anchors": int(m.sum()),
                      "e": float(o["e"][Vg].mean()) if Vg.any() else None, "d": float(o["d"][Vdg].mean()) if Vdg.any() else None,
                      "eloc": float(o["e_loc"][Vlg].mean()) if Vlg.any() else None,
                      "share_of_r": float(-(o["e"][Vg].sum() / nV + o["d"][Vdg].sum() / nVd + o["e_loc"][Vlg].sum() / nVl) / o["r"]),
                      "colour_diff": colour_consistency(real_grid[:T_D4RT], gen[:T_D4RT], src_g, tgt_g, colour_mask[g])}
            if Vdg.any():
                zbar = np.maximum(Y[:, :1, 2], p.z_min)
                gt_mv = np.linalg.norm(Y - Y[:, :1], axis=-1)
                pr_mv = np.linalg.norm(o["s"] * (pr - pr[:, :1]), axis=-1)
                row[g].update(gt_motion_m=float(gt_mv[Vdg].mean()), pred_motion_m=float(pr_mv[Vdg].mean()),
                              gt_motion_rel=float((gt_mv / zbar)[Vdg].mean()), pred_motion_rel=float((pr_mv / zbar)[Vdg].mean()))
        return row, gen

    rows, gens = [], {}
    for seed in range(a.G + a.extra_seeds):
        row, gen = analyse(seed, entries, "main")
        rows.append(row); gens[seed] = gen
        pr_ = row["person"]
        print(json.dumps({"seed": seed, "r": round(row["r"], 3), "s": round(row["s"], 3),
                          **{g: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row[g].items()} for g in groups}}), flush=True)
        (out / "q1_q3.json").write_text(json.dumps({"q2": q2, "rows": rows}, indent=1))

    # montage: every candidate at t = 20 with the projected person points (red) and a few static points (green)
    t = 20
    tiles = []
    for seed, gen in gens.items():
        im = np.ascontiguousarray(gen[t]).copy()
        okp = V[:, t] & pa
        for x, y in tgt_g[t][okp][::3]:
            cv2.circle(im, (int(round(x)), int(round(y))), 1, (255, 0, 0), -1)
        cv2.putText(im, f"seed {seed} r={rows[seed]['r']:.3f} col(person)={rows[seed]['person']['colour_diff']:.2f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        tiles.append(im)
    src = np.ascontiguousarray(real_grid[t]).copy()
    cv2.putText(src, "input (source camera), t=20", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    tiles = [src] + tiles
    while len(tiles) % 3:
        tiles.append(np.zeros_like(src))
    mont = np.concatenate([np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)], 0)
    cv2.imwrite(str(out / "montage_t20.jpg"), cv2.cvtColor(mont, cv2.COLOR_RGB2BGR))

    # I: scale-factor check (static colour consistency, reward) for 2 seeds
    scale_rows = []
    for sf in [float(x) for x in a.scales.split(",")]:
        it = dict(item); it["scale_factor"] = sf
        with torch.no_grad():
            ent, _ = b4.make_entries(it, vae, config, device)
        for seed in (0, 1):
            row, _ = analyse(seed, ent, f"scale{sf}")
            scale_rows.append({"scale": sf, "seed": seed, "r": row["r"], "static_colour": row["static"]["colour_diff"],
                               "person_colour": row["person"]["colour_diff"], "static_eloc": row["static"]["eloc"], "person_e": row["person"]["e"]})
            print(json.dumps(scale_rows[-1]), flush=True)
    (out / "q1_q3.json").write_text(json.dumps({"q2": q2, "rows": rows, "scale_check": scale_rows}, indent=1))
    print("Q2 " + json.dumps(q2))


if __name__ == "__main__":
    main()
