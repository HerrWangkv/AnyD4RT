#!/usr/bin/env python3
"""Controlled edits on REAL video at theta = 0 (target = source camera): does the reward systematically prefer deleting
the main moving object over a slightly misplaced one? Diagnostic only (README §5.1(a)): no labels are derived.

Per scene: the main moving instance (the mask id holding most dynamic GT points at t0; masks used only here) is
  deleted:   dilated mask inpainted (cv2 Telea) in every frame;
  shifted:   the object's pixels pasted dx grid-px to the right over the deleted background (dx in --shifts).
All variants are scored with the identical anchors, V (z-buffer, same camera), query chain and critic as the original.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anyd4rt import po_episode  # noqa: E402
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
    ap.add_argument("--scenes", default="cnb_dlab_0215_ego2,r1_new_f,ani10_new_f")
    ap.add_argument("--shifts", default="4,8,16,32")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "s4d"))
    a = ap.parse_args()
    critic = load_d4rt(str(ROOT / "third_party/Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG"))
    p = RewardParams()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in a.scenes.split(","):
        scene = Path("/tmp/kwang-mnt/pointodyssey/v2/val") / name
        clip = po_episode.load_clip(scene, 0, 2)
        H, W = clip["frames"].shape[1:3]
        gh, gw = b4.infer_lib.snap_shape(H, W)
        real = s4a.to_uint8(b4.infer_lib.resize_video(clip["frames"], (gh, gw)).permute(1, 0, 2, 3))[:T_D4RT]
        T_b = clip["T_cw"]
        zbs = [zbuffer(clip["depth"][t], clip["K"], T_b[t], clip["K"], T_b[t], (W, H)) for t in range(len(T_b))]
        px, z, state = po_episode.target_view_gt(clip, T_b, lambda t: zbs[t])
        V_all = clip["val"] & in_frame(px, (W, H)) & (z > 0) & (state == VIS_VISIBLE)
        ok0 = V_all[0] & np.isfinite(clip["X"]).all(-1).all(0)
        motion = np.nanmax(np.linalg.norm(clip["X"] - clip["X"][:1], axis=-1), axis=0)
        dyn_all = motion > 0.02 * np.maximum(z[0], 1e-3)
        ids = s4a.b4_sample(np.random.default_rng(0), ok0, dyn_all, 1024)
        Y = transform(T_b[0], clip["X"][:, ids]).transpose(1, 0, 2)[:, :T_D4RT]
        Y_loc = transform(T_b[:, None], clip["X"][:, ids]).transpose(1, 0, 2)[:, :T_D4RT]
        V = V_all[:, ids].T[:, :T_D4RT]
        uv0 = s4a.grid_px(px[0, ids], (W, H), (gw, gh), "pc") * np.array([256.0 / gw, 256.0 / gh]) / 255.0
        masks = [np.array(Image.open(scene / f"masks/mask_{clip['idx'][t]:05d}.png")) for t in range(T_D4RT)]
        q = np.round(px[0][ok0 & dyn_all]).astype(int)
        mids, cnt = np.unique(masks[0][q[:, 1], q[:, 0]], return_counts=True)
        obj = int(mids[np.argmax(cnt)])
        om = np.stack([cv2.resize((m == obj).astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST) for m in masks]).astype(bool)
        pa = np.zeros(len(ids), bool)
        q0 = np.round(px[0, ids]).astype(int)
        pa[:] = masks[0][q0[:, 1], q0[:, 0]] == obj
        kern = np.ones((7, 7), np.uint8)
        deleted = np.stack([cv2.inpaint(f, cv2.dilate(m.astype(np.uint8), kern), 5, cv2.INPAINT_TELEA) for f, m in zip(real, om)])

        def score(frames):
            pr, pl = s4b.predict(critic, frames, uv0)
            o = reward_d4rt_gt(pr, Y, V, 0, p, pl, Y_loc)
            Vp = V & pa[:, None]
            return {"r": o["r"], "e": o["e_mean"], "d": o["d_mean"], "eloc": o["eloc_mean"],
                    "obj_e": float(o["e"][Vp].mean()), "obj_eloc": float(o["e_loc"][Vp & (np.arange(T_D4RT) != 0)[None]].mean()),
                    "rest_e": float(o["e"][V & ~pa[:, None]].mean())}

        row = {"scene": name, "object_id": obj, "object_area_frac_t0": float(om[0].mean()), "object_anchors": int(pa.sum()),
               "object_frac_of_V": float((V & pa[:, None]).sum() / V.sum()), "original": score(real), "deleted": score(deleted)}
        vis = [real[20].copy(), deleted[20].copy()]
        for dx in [int(x) for x in a.shifts.split(",")]:
            sh = deleted.copy()
            for t in range(T_D4RT):
                ys, xs = np.nonzero(om[t])
                xs2 = np.clip(xs + dx, 0, gw - 1)
                sh[t, ys, xs2] = real[t, ys, xs]
            row[f"shift{dx}"] = score(sh)
            if dx in (8, 32):
                vis.append(sh[20].copy())
        rows.append(row)
        print(json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in row.items()}), flush=True)
        labs = ["original", "object deleted (inpaint)", "shifted +8 px", "shifted +32 px"]
        for im, lab in zip(vis, labs):
            cv2.putText(im, f"{name}: {lab}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imwrite(str(out / f"{name}_edits_t20.jpg"), cv2.cvtColor(np.concatenate([np.concatenate(vis[:2], 1), np.concatenate(vis[2:], 1)], 0), cv2.COLOR_RGB2BGR))
        (out / "edits.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
