#!/usr/bin/env python3
"""Within-group exploration: fixed starting latent + varying SDE noise vs both varying (cached S4 groups).

Per group (8 candidates, reward config A):
  sigma_group   std of the candidates' rewards (the signal GRPO standardizes);
  sigma_query   per-candidate std of the reward over random anchor halves (40 half-rewards from 20 splits), averaged
                over candidates (score noise from the query subset);
  snr           sigma_group / sigma_query; split-half Kendall tau of the ranking (mean over splits);
  appearance    pairwise PSNR / LPIPS between candidates (AnyView's RGBEvaluation, 576 grid);
  geometry      critic-free alignment reference per candidate (static / dynamic shift, grid px, search +-32);
                a shift >= 30 px counts as saturated; the reference is "usable" for the group only if at most 2 of 8
                candidates are saturated on static points; otherwise its ranking is reported as not judgeable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
AV = ROOT / "third_party" / "AnyView-DVS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AV))

from anyd4rt.reward_d4rt_gt import RewardParams, reward_d4rt_gt  # noqa: E402

import importlib.util  # noqa: E402

_g = importlib.util.spec_from_file_location("s4g", ROOT / "scripts" / "s4g_gate_offline.py")
s4g = importlib.util.module_from_spec(_g)
_g.loader.exec_module(s4g)
_e = importlib.util.spec_from_file_location("av_eval", AV / "scripts" / "eval_avb.py")
eval_lib = importlib.util.module_from_spec(_e)
_e.loader.exec_module(eval_lib)
P = RewardParams()
SAT = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "outputs" / "s4cache"))
    ap.add_argument("--groups", required=True, help="comma-separated group dirs; '<dir>:0-7' limits candidates")
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "s4h" / "noise.json"))
    a = ap.parse_args()
    from anyview.metrics import RGBEvaluation
    ev = RGBEvaluation()
    res = {}
    for spec in a.groups.split(","):
        name, _, rng_s = spec.partition(":")
        d = Path(a.cache) / name
        g = dict(np.load(d / "group.npz"))
        files = sorted(d.glob("cand_*.npz"))
        if rng_s:
            lo, hi = map(int, rng_s.split("-"))
            files = [f for f in files if lo <= int(f.stem.split("_")[1]) <= hi]
        C = [dict(np.load(f)) for f in files]
        seeds = [int(f.stem.split("_")[1]) for f in files]
        r = np.array([reward_d4rt_gt(c["pr"], g["Y"], g["V"], 0, P, c["pl"], g["Y_loc"])["r"] for c in C])
        rng = np.random.default_rng(1)
        half_r, taus = [], []
        for _ in range(a.splits):
            perm = rng.permutation(len(g["ids"]))
            hs = []
            for h in (perm[: len(perm) // 2], perm[len(perm) // 2:]):
                m = np.zeros(len(g["ids"]), bool); m[h] = True
                hs.append([reward_d4rt_gt(c["pr"], g["Y"], g["V"], 0, P, c["pl"], g["Y_loc"], anchor_gate=m)["r"] for c in C])
            half_r += hs
            taus.append(s4g.kendall(hs[0], hs[1]))
        half_r = np.array(half_r)  # [40, G]
        sigma_q = float(half_r.std(axis=0, ddof=1).mean())
        sigma_g = float(r.std(ddof=1))
        pair = []
        for i, j in itertools.combinations(range(len(C)), 2):
            vi = torch.from_numpy(C[i]["video"][:40]).permute(0, 3, 1, 2).float() / 255.0
            vj = torch.from_numpy(C[j]["video"][:40]).permute(0, 3, 1, 2).float() / 255.0
            pair.append(eval_lib.score_episode(ev, vi, vj, "cuda"))
        ref_path = d / ("alignment_ref.json" if not rng_s else "alignment_ref.json")
        ref = json.loads(ref_path.read_text()) if ref_path.exists() else {}
        for s_, c in zip(seeds, C):
            if str(s_) not in ref:
                ref[str(s_)] = s4g.alignment_reference(g, c["video"])
        ref_path.write_text(json.dumps(ref))
        st = np.array([ref[str(s_)]["static"] if ref[str(s_)]["static"] is not None else np.nan for s_ in seeds])
        dy = np.array([ref[str(s_)]["dynamic"] if ref[str(s_)]["dynamic"] is not None else np.nan for s_ in seeds])
        n_sat = int(np.sum(st >= SAT))
        usable = n_sat <= 2
        res[spec] = {"seeds": seeds, "rewards": r.tolist(), "sigma_group": sigma_g, "sigma_query": sigma_q, "snr": sigma_g / sigma_q,
                     "reward_range": float(r.max() - r.min()), "split_half_tau": float(np.nanmean(taus)),
                     "pair_psnr_mean": float(np.mean([x["psnr"] for x in pair])), "pair_lpips_mean": float(np.mean([x["lpips"] for x in pair])),
                     "ref_static_px": st.tolist(), "ref_dynamic_px": dy.tolist(), "ref_static_std": float(np.nanstd(st)), "ref_dynamic_std": float(np.nanstd(dy)),
                     "ref_n_saturated_static": n_sat, "ref_usable": usable,
                     "tau_reward_vs_ref_static": s4g.kendall(r, -st) if usable else None,
                     "tau_reward_vs_ref_dynamic": s4g.kendall(r, -dy) if usable else None}
        print(json.dumps({"group": spec, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[spec].items() if k not in ("seeds", "rewards", "ref_static_px", "ref_dynamic_px")}}), flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
