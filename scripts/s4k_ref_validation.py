#!/usr/bin/env python3
"""Minimal validation of the critic-free alignment reference (s4g.alignment_reference) before using it as a yardstick.

1. Known shifts: theta = 0 groups (target = source camera), the REAL source video shifted by a known (dx, dy) and
   optionally degraded (Gaussian blur sigma 1.5 / 3, additive noise sigma 8/255) plays the candidate; the reference
   should recover |(dx, dy)| on static and dynamic points (step 2 px, search +-32).
2. Split-half consistency on generated theta = 10 candidates: the static (and dynamic) shift computed on two disjoint
   halves of the reference points; |difference| per candidate. A large difference means the reference is not
   reliable for that candidate even if it is below saturation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402

_g = importlib.util.spec_from_file_location("s4g", ROOT / "scripts" / "s4g_gate_offline.py")
s4g = importlib.util.module_from_spec(_g)
_g.loader.exec_module(s4g)


def shift_video(v, dx, dy, blur=0.0, noise=0.0, seed=0):
    h, w = v.shape[1:3]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out = np.stack([cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REFLECT) for f in v])
    if blur > 0:
        out = np.stack([cv2.GaussianBlur(f, (0, 0), blur) for f in out])
    if noise > 0:
        out = np.clip(out + np.random.default_rng(seed).normal(0, noise * 255, out.shape), 0, 255).astype(np.uint8)
    return out


def split_ref(g, video, frames=range(0, 40, 3)):
    S, G = s4g.norm(g["real"][:40]), s4g.norm(video[:40])
    ok = g["src_ok"] & g["V"].T
    idx = np.arange(len(g["ids"]))
    out = {}
    for name, sel in (("static", g["motion"] < 0.01), ("dynamic", g["dyn"])):
        halves = []
        for h in (idx % 2 == 0, idx % 2 == 1):
            sh = [s4g.align_shift(S[t], G[t], g["px_s_grid"][t][ok[t] & sel & h], g["px_t_grid"][t][ok[t] & sel & h])
                  for t in frames if (ok[t] & sel & h).sum() >= 20]
            halves.append(float(np.median(sh)) if sh else None)
        out[name] = halves
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "outputs" / "s4cache"))
    ap.add_argument("--theta0-groups", default="cnb_dlab_0215_ego2@0_theta0,r1_new_f@0_theta0,ani10_new_f@0_theta0")
    ap.add_argument("--split-groups", default="")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "s4v2" / "ref_validation.json"))
    a = ap.parse_args()
    res = {"known_shifts": {}, "split_half": {}}
    shifts = [(0, 0), (4, 0), (0, 4), (8, 0), (8, 8), (16, 0), (0, 16), (24, 0), (20, 20)]
    for name in a.theta0_groups.split(","):
        g = dict(np.load(Path(a.cache) / name / "group.npz"))
        rows = []
        for deg in ((0.0, 0.0), (1.5, 0.0), (3.0, 8 / 255)):
            for dx, dy in shifts:
                r = s4g.alignment_reference(g, shift_video(g["real"], dx, dy, *deg))
                rows.append({"blur": deg[0], "noise": deg[1], "true_px": float(np.hypot(dx, dy)), "static": r["static"], "dynamic": r["dynamic"]})
        res["known_shifts"][name] = rows
        err = [abs(x["static"] - x["true_px"]) for x in rows if x["static"] is not None]
        errd = [abs(x["dynamic"] - x["true_px"]) for x in rows if x["dynamic"] is not None]
        print(json.dumps({"group": name, "static_abs_err_median": float(np.median(err)), "static_abs_err_max": float(np.max(err)),
                          "dynamic_abs_err_median": float(np.median(errd)) if errd else None, "dynamic_abs_err_max": float(np.max(errd)) if errd else None}), flush=True)
    for name in [x for x in a.split_groups.split(",") if x]:
        d = Path(a.cache) / name
        g = dict(np.load(d / "group.npz"))
        rows = {}
        for f in sorted(d.glob("cand_*.npz")):
            rows[int(f.stem.split("_")[1])] = split_ref(g, np.load(f)["video"])
        res["split_half"][name] = rows
        diffs = {k: [abs(v[k][0] - v[k][1]) for v in rows.values() if None not in v[k]] for k in ("static", "dynamic")}
        print(json.dumps({"group": name, **{f"{k}_halves_absdiff_median": float(np.median(x)) if x else None for k, x in diffs.items()},
                          **{f"{k}_halves_absdiff_max": float(np.max(x)) if x else None for k, x in diffs.items()}}), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
