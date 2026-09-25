#!/usr/bin/env python3
"""B4 (README §6 S3): stochastic transition inside the real AnyView sampler.

Per episode (AnyViewBench, released preprocessing unchanged, 35 steps, seed 0):
  C1 wrapper parity:   anyd4rt rollout in mode "ab2" vs pipe.generate (same seed): max |diff| of the rgb0 latent.
  C2 first step:       on the real first network output, Euler in z coordinates vs the released scheduler step.
  C3 conditioning:     recompute a recorded step with every non-output stream/channel zeroed vs randomized;
                       the prediction on the output region must not change.
  C4 log-prob:         SDE rollout (a = --check-a) records every step; recompute log p(z_{i+1}|z_i) with a fresh
                       network call in eval() and in train() mode; report |logp_new - logp_old| per step, and the
                       standardized residual (z_{i+1} - mu)/std on the output region.
  C5 quality:          ab2 / euler / sde(a in --sde-a) at 35 steps: PSNR / SSIM / LPIPS vs the target GT.
C3 and C4 run only on the first --checks-episodes episodes of each split (they cost extra network calls).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
AV = ROOT / "third_party" / "AnyView-DVS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AV))

from anyd4rt.anyview_rl import rollout, sde  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


infer_lib = _load("av_infer", AV / "scripts" / "infer.py")
eval_lib = _load("av_eval_avb", AV / "scripts" / "eval_avb.py")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="zeroshot:droid_OOD_LR,indist:kubric5d")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--checks-episodes", type=int, default=2)
    ap.add_argument("--sde-a", default="0.3,0.5,0.7")
    ap.add_argument("--check-a", type=float, default=0.7)
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "b4"))
    return ap.parse_args()


def make_entries(item, vae, config, device):
    """generate_target_view() up to the entries, unchanged (released resize, intrinsics scaling, re-anchoring)."""
    from anyview.cameras import resolve_scale_factor
    from anyview.logistics import build_dvs_entries

    frames = eval_lib.as_uint8_video(item["input"])
    gt = eval_lib.as_uint8_video(item["target_gt"])
    native_hw = (tuple(frames.shape[1:3]), tuple(gt.shape[1:3]))
    cams = {"intrinsics": {0: item["intrinsics_target"].float(), 1: item["intrinsics_input"].float()},
            "world2cam": {0: item["world2cam_target"].float(), 1: item["world2cam_input"].float()}}
    cams["world2cam"] = infer_lib.reanchor_world2cam(cams["world2cam"])
    scale_factor = resolve_scale_factor(item)
    hw1 = infer_lib.snap_shape(*frames.shape[1:3])
    hw0 = infer_lib.snap_shape(*gt.shape[1:3])
    video1 = infer_lib.resize_video(frames, hw1)
    rgb1 = vae.encode_rgb(video1.unsqueeze(0).to(device))
    c0 = infer_lib.prepare_cams_latent(vae, cams["intrinsics"][0], cams["world2cam"][0], native_hw[1], hw0, scale_factor, device)
    c1 = infer_lib.prepare_cams_latent(vae, cams["intrinsics"][1], cams["world2cam"][1], native_hw[0], hw1, scale_factor, device)
    entries = build_dvs_entries(None, rgb1, c0, c1)
    gt_resized = infer_lib.resize_video(gt, tuple(hw0)).permute(1, 0, 2, 3)
    return entries, gt_resized


def decode(vae, y0_pred_streams):
    from anyview.logistics import unpack_entries_from_streams
    rgb0 = unpack_entries_from_streams(y0_pred_streams)["rgb0"]
    return vae.decode_rgb(rgb0)[0].to(torch.float32).clamp(0, 1).permute(1, 0, 2, 3).cpu()


def rgb0_latent(y0_pred_streams):
    return y0_pred_streams["v0"][:, 0:16].float()


@torch.no_grad()
def check_first_step(pipe, entries):
    """C2: one real network call, then released scheduler step vs Euler in z coordinates."""
    from anyview.pipe import scheduler_step_streams
    x0s, masks, cross, state = rollout._setup(pipe, entries, 0, 35)
    _, y0 = pipe.denoise(x0s, state, masks, rollout.sigma_in(pipe, 0, state), cross)
    ref, _ = scheduler_step_streams(pipe.scheduler, x0_pred_streams=y0, i=0, sample_streams=dict(state))
    s_t, s_s = float(pipe.scheduler.sigmas[0]), float(pipe.scheduler.sigmas[1])
    om = masks["output"]["v0"].bool()
    z_next, _, _ = sde.step(sde.x_to_z(state["v0"].double(), s_t), y0["v0"].double(), sde.tau_of(s_t), sde.tau_of(s_s), 0.0)
    ours = sde.z_to_x(z_next, s_s).float()
    d = (ours - ref["v0"]).abs()[om]
    return {"max_abs": float(d.max()), "max_rel": float(d.max() / ref["v0"].abs()[om].max())}


@torch.no_grad()
def check_conditioning(pipe, entries, rec, a):
    """C3: prediction on the output region with other channels zeroed vs randomized."""
    x0s, masks, _, start = rollout._setup(pipe, entries, 0, 35)
    g = torch.Generator(device=start["v0"].device).manual_seed(123)
    rand = {k: torch.randn(v.shape, generator=g, device=v.device) * 50 for k, v in start.items()}
    lp0, mu0 = rollout.recompute_log_prob(pipe, entries, rec, a=a)
    lp1, mu1 = rollout.recompute_log_prob(pipe, entries, rec, a=a, other_state=rand)
    om = masks["output"][rec["stream"]].bool()
    return {"i": rec["i"], "max_abs_mu_diff": float((mu0 - mu1).abs()[om].max()), "logp_diff": float((lp0 - lp1).abs().max())}


@torch.no_grad()
def check_log_prob(pipe, entries, steps, a):
    """C4: recompute every recorded step in eval() and train(); standardized residuals of the rollout itself."""
    rows = []
    for mode in ("eval", "train"):
        pipe.dit.train(mode == "train")
        for rec in steps:
            lp, mu = rollout.recompute_log_prob(pipe, entries, rec, a=a)
            row = {"mode": mode, "i": rec["i"], "logp_old": float(rec["logp_old"][0]), "logp_new": float(lp[0]),
                   "abs_diff": float((lp - rec["logp_old"]).abs()[0])}
            if mode == "eval":
                x0s, masks, _, _ = rollout._setup(pipe, entries, 0, 35)
                om = masks["output"][rec["stream"]].bool()
                r = ((rec["z_next"] - mu) / rec["std"])[om]
                row.update(resid_mean=float(r.mean()), resid_var=float(r.var()))
            rows.append(row)
    pipe.dit.eval()
    return rows


def main():
    a = parse_args()
    from anyview.avb_dataset import AVBDataset
    from anyview.config import AnyViewConfig
    from anyview.metrics import RGBEvaluation
    from anyview.pipe import load_pipeline
    from anyview.vae import load_vae

    device = torch.device("cuda")
    config = AnyViewConfig(checkpoint_path=str(AV / "checkpoints/anyview_dvs_2b.pt"), tokenizer_path=str(AV / "checkpoints/tokenizer.pth"))
    vae = load_vae(config.tokenizer_path, device)
    pipe = load_pipeline(config, device)
    evaluator = RGBEvaluation()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sde_as = [float(x) for x in a.sde_a.split(",")]
    rows = []
    for spec in a.splits.split(","):
        tier, split = spec.split(":")
        ds = AVBDataset(str(AV / "data" / f"AnyViewBench_{tier}"), split)
        for idx in range(min(a.episodes, len(ds))):
            t0 = time.time()
            item = ds[idx]
            ep = item.get("episode", f"{split}_ep{idx:03d}")
            with torch.no_grad():
                entries, gt = make_entries(item, vae, config, device)
            row = {"split": split, "episode": ep, "metrics": {}}
            with torch.no_grad():
                ref = pipe.generate(entries, seed=0, num_sampling_steps=a.num_steps)
            ours = rollout.sample(pipe, entries, seed=0, num_steps=a.num_steps, mode="ab2")
            row["C1_ab2_parity_max_abs"] = float((rgb0_latent(ours.y0_pred_streams) - rgb0_latent(ref["y0_pred_streams"])).abs().max())
            row["metrics"]["ab2"] = eval_lib.score_episode(evaluator, decode(vae, ours.y0_pred_streams), gt, device)
            row["C2_first_step"] = check_first_step(pipe, entries)
            eu = rollout.sample(pipe, entries, seed=0, num_steps=a.num_steps, mode="euler")
            row["metrics"]["euler"] = eval_lib.score_episode(evaluator, decode(vae, eu.y0_pred_streams), gt, device)
            do_checks = idx < a.checks_episodes
            for av in sde_as:
                rec = do_checks and abs(av - a.check_a) < 1e-9
                r = rollout.sample(pipe, entries, seed=0, num_steps=a.num_steps, mode="sde", a=av, noise_seed=idx, record=rec)
                row["metrics"][f"sde_a{av}"] = eval_lib.score_episode(evaluator, decode(vae, r.y0_pred_streams), gt, device)
                if rec:
                    picks = [r.steps[j] for j in (0, len(r.steps) // 2, len(r.steps) - 1)]
                    row["C3_conditioning"] = [check_conditioning(pipe, entries, s, av) for s in picks]
                    row["C4_logprob"] = check_log_prob(pipe, entries, r.steps, av)
            row["sec"] = round(time.time() - t0, 1)
            rows.append(row)
            m = row["metrics"]
            print(f"{ep}: parity {row['C1_ab2_parity_max_abs']:.2e} | " +
                  " | ".join(f"{k} {v['psnr']:.2f}/{v['ssim']:.3f}/{v['lpips']:.3f}" for k, v in m.items()) + f" | {row['sec']}s", flush=True)
            (out / "episodes.json").write_text(json.dumps(rows, indent=1, default=float))

    summary = {"n_episodes": len(rows), "num_steps": a.num_steps, "seed": 0}
    for split in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == split]
        summary[split] = {k: {m: float(np.mean([r["metrics"][k][m] for r in rs])) for m in ("psnr", "ssim", "lpips")} for k in rs[0]["metrics"]}
    summary["C1_ab2_parity_max_abs"] = max(r["C1_ab2_parity_max_abs"] for r in rows)
    summary["C2_first_step_max_rel"] = max(r["C2_first_step"]["max_rel"] for r in rows)
    c3 = [c for r in rows for c in r.get("C3_conditioning", [])]
    summary["C3_conditioning_max_abs_mu_diff"] = max((c["max_abs_mu_diff"] for c in c3), default=None)
    c4 = [c for r in rows for c in r.get("C4_logprob", [])]
    for mode in ("eval", "train"):
        d = [c["abs_diff"] for c in c4 if c["mode"] == mode]
        summary[f"C4_logprob_abs_diff_{mode}"] = {"max": max(d), "mean": float(np.mean(d)), "n": len(d)} if d else None
    res = [c for c in c4 if c["mode"] == "eval"]
    if res:
        summary["C4_resid_mean_range"] = [min(c["resid_mean"] for c in res), max(c["resid_mean"] for c in res)]
        summary["C4_resid_var_range"] = [min(c["resid_var"] for c in res), max(c["resid_var"] for c in res)]
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
