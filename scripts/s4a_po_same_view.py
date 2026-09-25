#!/usr/bin/env python3
"""S4 prerequisites on PointOdyssey with the target camera equal to the source camera (theta = 0).

Per clip (41 frames, stride 2):
  Z. z-buffer visibility (source depth splatted into the target = same camera) vs GT `visibs`, for GT points in frame.
  P. pixel convention of GENERATED video: AnyView output at theta = 0 vs the input resized to the same grid with
     the released LANCZOS (pixel-center) resize; sub-pixel shift by phase correlation, per frame.
     If AnyView's content follows the pixel-center convention the shift is ~0; if it follows the multiplicatively
     scaled intrinsics it is 0.5 * (1 - s) px (x: +0.20, y: +0.22 for 960x540 -> 576x304).
  R. reward (README §3.5, combined) on the real input (noise floor, this torch 2.7 environment) and on generated
     candidates, through the full chain: grid image -> 256^2 (cv2 INTER_AREA, pixel-center) -> query. The native
     GT pixel x maps to the generated grid by "pc" x_g = (x + 0.5) s - 0.5 or "mult" x_g = s x; the query is the
     OpenD4RT training convention on the 256 input: u = (256 / W_g) x_g / 255.
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

ROOT = Path(__file__).resolve().parents[1]
AV = ROOT / "third_party" / "AnyView-DVS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AV))

from anyd4rt import po_episode  # noqa: E402
from anyd4rt.anyview_rl import rollout  # noqa: E402
from anyd4rt.d4rt_critic import encode, load_d4rt, prepare_video, run_queries  # noqa: E402
from anyd4rt.geometry import VIS_OCCLUDED, VIS_UNKNOWN, VIS_VISIBLE, in_frame, project, transform, zbuffer  # noqa: E402
from anyd4rt.reward_d4rt_gt import RewardParams, reward_d4rt_gt  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("b4", ROOT / "scripts" / "b4_sampler_checks.py")
b4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b4)
T_D4RT = 40


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/tmp/kwang-mnt/pointodyssey/v2/val")
    ap.add_argument("--scenes", default="ani10_new_f,cnb_dlab_0215_ego2,r1_new_f")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--a", type=float, default=0.7)
    ap.add_argument("--num-anchors", type=int, default=1024)
    ap.add_argument("--exp-dir", default=str(ROOT / "third_party/Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "s4a"))
    return ap.parse_args()


def phase_shift(gen, ref):
    """Per-frame sub-pixel shift (dx, dy) of gen relative to ref (both [T,H,W,3] uint8), Hanning-windowed."""
    win = cv2.createHanningWindow(ref.shape[2:0:-1], cv2.CV_64F)
    out = []
    for g, r in zip(gen, ref):
        (dx, dy), resp = cv2.phaseCorrelate(cv2.cvtColor(r, cv2.COLOR_RGB2GRAY).astype(np.float64),
                                           cv2.cvtColor(g, cv2.COLOR_RGB2GRAY).astype(np.float64), win)
        out.append((dx, dy, resp))
    return np.array(out)


def to_uint8(pred):  # [T,3,H,W] float in [0,1] -> [T,H,W,3] uint8
    return (pred.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255 + 0.5).astype(np.uint8)


def grid_px(px, native_wh, grid_wh, conv):
    s = np.array(grid_wh, dtype=np.float64) / np.array(native_wh, dtype=np.float64)
    return (px + 0.5) * s - 0.5 if conv == "pc" else px * s


def reward_on_video(model, frames_grid, clip, ids, V, Y, Y_loc, px0_native, native_wh, conv, p):
    """Combined reward (§3.5) with OpenD4RT on a video given at the grid resolution (first 40 frames)."""
    gh, gw = frames_grid.shape[1:3]
    video, aspect = prepare_video(frames_grid[:T_D4RT], "cuda")
    mem = encode(model, video, aspect)
    xg = grid_px(px0_native, native_wh, (gw, gh), conv)
    uv0 = xg * np.array([256.0 / gw, 256.0 / gh]) / 255.0
    n, T = len(ids), T_D4RT
    t_tgt = np.tile(np.arange(T), n)
    uv = np.repeat(uv0, T, 0)
    pr = run_queries(model, video, aspect, uv, np.zeros_like(t_tgt), t_tgt, np.zeros_like(t_tgt), memory=mem)["xyz_3d"].reshape(n, T, 3)
    pl = run_queries(model, video, aspect, uv, np.zeros_like(t_tgt), t_tgt, t_tgt, memory=mem)["xyz_3d"].reshape(n, T, 3)
    out = reward_d4rt_gt(pr, Y[:, :T], V[:, :T], 0, p, pl, Y_loc[:, :T])
    return {k: out.get(k) for k in ("r", "e_mean", "d_mean", "eloc_mean", "s", "scale_valid", "frac_capped_e", "frac_capped_eloc")}


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
    rows = []
    for name in a.scenes.split(","):
        t0 = time.time()
        clip = po_episode.load_clip(Path(a.data_root) / name, 0, a.stride)
        H, W = clip["frames"].shape[1:3]
        T_cw = clip["T_cw"]
        # Z: z-buffer vs GT visibility, same camera
        zbs = [zbuffer(clip["depth"][t], clip["K"], T_cw[t], clip["K"], T_cw[t], (W, H), stride=2) for t in range(len(T_cw))]
        px, z, state = po_episode.target_view_gt(clip, T_cw, lambda t: zbs[t])
        infr = clip["val"] & in_frame(px, (W, H)) & (z > 0)
        gv = clip["vis"]
        zrow = {"n": int(infr.sum()),
                "P(zb visible | GT visible)": float((state[infr & gv] == VIS_VISIBLE).mean()),
                "P(zb occluded | GT occluded)": float((state[infr & ~gv] == VIS_OCCLUDED).mean()),
                "P(GT visible | zb visible)": float(gv[infr & (state == VIS_VISIBLE)].mean()),
                "P(GT occluded | zb occluded)": float((~gv)[infr & (state == VIS_OCCLUDED)].mean()),
                "frac unknown": float((state[infr] == VIS_UNKNOWN).mean())}
        # anchors and references (target = source camera); V from the z-buffer, not from GT visibs
        rng = np.random.default_rng(0)
        V_all = infr & (state == VIS_VISIBLE)
        ok0 = V_all[0] & np.isfinite(clip["X"]).all(-1).all(0)
        motion = np.nanmax(np.linalg.norm(clip["X"] - clip["X"][:1], axis=-1), axis=0)
        dyn = motion > 0.02 * np.maximum(z[0], 1e-3)
        ids = b4_sample(rng, ok0, dyn, a.num_anchors)
        Y = transform(T_cw[0], clip["X"][:, ids]).transpose(1, 0, 2)
        Y_loc = transform(T_cw[:, None], clip["X"][:, ids]).transpose(1, 0, 2)
        V = V_all[:, ids].T
        row = {"scene": name, "zbuffer_vs_gt": zrow, "n_anchor": len(ids), "n_dyn": int(dyn[ids].sum())}
        # R0: real input at the AnyView grid, noise floor through the same chain
        from anyview import cameras  # noqa: F401
        hw = b4.infer_lib.snap_shape(H, W)
        real_grid = to_uint8(b4.infer_lib.resize_video(clip["frames"], tuple(hw)).permute(1, 0, 2, 3))
        row["reward_real_grid"] = {conv: reward_on_video(critic, real_grid, clip, ids, V, Y, Y_loc, px[0, ids], (W, H), conv, p) for conv in ("pc", "mult")}
        # generation at theta = 0
        item = po_episode.av_item(clip, T_cw)
        with torch.no_grad():
            entries, _ = b4.make_entries(item, vae, config, device)
        cands = []
        for g in range(a.G):
            r = rollout.sample(pipe, entries, seed=g, num_steps=35, mode="sde", a=a.a, noise_seed=g)
            gen = to_uint8(b4.decode(vae, r.y0_pred_streams))
            sh = phase_shift(gen, real_grid)
            psnr = float(np.mean([cv2.PSNR(x, y) for x, y in zip(gen, real_grid)]))
            cands.append({"seed": g, "psnr_vs_input_grid": psnr,
                          "shift_px_median": [float(np.median(sh[:, 0])), float(np.median(sh[:, 1]))],
                          "shift_px_mean": [float(sh[:, 0].mean()), float(sh[:, 1].mean())], "phasecorr_response_median": float(np.median(sh[:, 2])),
                          "reward": {conv: reward_on_video(critic, gen, clip, ids, V, Y, Y_loc, px[0, ids], (W, H), conv, p) for conv in ("pc", "mult")}})
            if g == 0:
                cv2.imwrite(str(out / f"{name}_frame20_input_vs_gen.jpg"), cv2.cvtColor(np.concatenate([real_grid[20], gen[20]], 1), cv2.COLOR_RGB2BGR))
        row["candidates"] = cands
        row["grid_hw"] = list(hw)
        row["sec"] = round(time.time() - t0, 1)
        rows.append(row)
        print(json.dumps({"scene": name, "zbuffer": zrow, "real": row["reward_real_grid"],
                          "gen": [{k: c[k] for k in ("seed", "psnr_vs_input_grid", "shift_px_median")} | {"r_pc": c["reward"]["pc"]["r"], "r_mult": c["reward"]["mult"]["r"]} for c in cands],
                          "sec": row["sec"]}), flush=True)
        (out / "rows.json").write_text(json.dumps(rows, indent=1))


def b4_sample(rng, ok0, dynamic, n):
    dyn, sta = np.flatnonzero(ok0 & dynamic), np.flatnonzero(ok0 & ~dynamic)
    nd = min(len(dyn), n // 2)
    ns = min(len(sta), n - nd)
    return np.concatenate([rng.choice(dyn, nd, replace=False), rng.choice(sta, ns, replace=False)])


if __name__ == "__main__":
    main()
