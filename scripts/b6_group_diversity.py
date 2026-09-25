#!/usr/bin/env python3
"""Within-group diversity (README §6 S3/S4 prep): same scene, same cameras, G candidates differing only by seed.

Per scene: G candidates with SDE (a = --a; seed g sets both the starting noise and the per-step noise) and
G candidates with the released AB2 sampler (seed g sets the starting noise only), 35 steps, released
preprocessing unchanged. Reports per candidate PSNR/SSIM/LPIPS vs GT, their within-group spread, and
pairwise PSNR/LPIPS between candidates of the same group (how different the candidates are from each other).
"""

from __future__ import annotations

import argparse
import itertools
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

from anyd4rt.anyview_rl import rollout  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("b4", ROOT / "scripts" / "b4_sampler_checks.py")
b4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b4)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="zeroshot:droid_OOD_LR")
    ap.add_argument("--episodes", default="0,1,2")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--a", type=float, default=0.7)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "b6"))
    return ap.parse_args()


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
    ev = RGBEvaluation()
    tier, split = a.split.split(":")
    ds = AVBDataset(str(AV / "data" / f"AnyViewBench_{tier}"), split)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in [int(x) for x in a.episodes.split(",")]:
        t0 = time.time()
        item = ds[idx]
        with torch.no_grad():
            entries, gt = b4.make_entries(item, vae, config, device)
        row = {"split": split, "episode": item.get("episode", f"{split}_ep{idx:03d}")}
        for mode in ("sde", "ab2"):
            preds = []
            for g in range(a.G):
                r = rollout.sample(pipe, entries, seed=g, num_steps=35, mode=mode, a=a.a if mode == "sde" else 0.0, noise_seed=g)
                preds.append(b4.decode(vae, r.y0_pred_streams))
            vs_gt = [b4.eval_lib.score_episode(ev, p, gt, device) for p in preds]
            pair = [b4.eval_lib.score_episode(ev, preds[i], preds[j], device)
                    for i, j in itertools.combinations(range(a.G), 2)]
            stat = lambda xs, m: {"mean": float(np.mean([x[m] for x in xs])), "std": float(np.std([x[m] for x in xs], ddof=1)),  # noqa: E731
                                  "min": float(np.min([x[m] for x in xs])), "max": float(np.max([x[m] for x in xs]))}
            row[mode] = {"vs_gt": {m: stat(vs_gt, m) for m in ("psnr", "ssim", "lpips")},
                         "pairwise": {m: stat(pair, m) for m in ("psnr", "lpips")},
                         "per_candidate_vs_gt": vs_gt}
        row["sec"] = round(time.time() - t0, 1)
        rows.append(row)
        s, b = row["sde"], row["ab2"]
        print(f"{row['episode']}: SDE vsGT PSNR {s['vs_gt']['psnr']['mean']:.2f}±{s['vs_gt']['psnr']['std']:.2f} "
              f"[{s['vs_gt']['psnr']['min']:.2f},{s['vs_gt']['psnr']['max']:.2f}] pairLPIPS {s['pairwise']['lpips']['mean']:.3f} | "
              f"AB2 vsGT PSNR {b['vs_gt']['psnr']['mean']:.2f}±{b['vs_gt']['psnr']['std']:.2f} pairLPIPS {b['pairwise']['lpips']['mean']:.3f} | {row['sec']}s", flush=True)
        (out / f"{split}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
