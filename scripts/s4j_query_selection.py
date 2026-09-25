#!/usr/bin/env python3
"""Can source-side reliability and per-point view change select trustworthy novel-view queries? (cached groups)

Same candidates, critic and original query pool (anchors, V) per group; only the scored subset changes:
  N   no filter (reward config A);
  S   source-side reliable anchors: position support (e_i <= 0.15 and e_loc_i <= 0.15) for every anchor; dynamic
      anchors additionally need displacement support (d_i / m_i <= 0.5) unless they are low-motion (m_i < 0.03),
      which are kept on position support alone;
  SV  S, and additionally only anchor-times whose source/target viewing-angle difference is <= 10 deg.
Thresholds are fixed here (not tuned on the deleted-person case). Per config and group:
  correctness   Kendall tau vs the critic-free reference, only where that reference is valid (s4_common), with n:
                full reward vs static / dynamic reference, and the static-part reward vs the static reference, the
                dynamic-part reward vs the dynamic reference;
  stability     split-half Kendall tau over random anchor halves (20 splits);
  coverage      retained fraction of dynamic / static anchors and of V entries.
With 8 candidates a Kendall tau of about 0.57 is needed for p < 0.05 (two-sided); single-group values are noisy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anyd4rt import po_episode  # noqa: E402
from anyd4rt.reward_d4rt_gt import RewardParams, reward_d4rt_gt  # noqa: E402

import importlib.util  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


s4c = _load("s4_common", ROOT / "scripts" / "s4_common.py")
s4g = _load("s4g", ROOT / "scripts" / "s4g_gate_offline.py")
P = RewardParams()
T = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "outputs" / "s4cache"))
    ap.add_argument("--groups", required=True)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--max-view-deg", type=float, default=10.0)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "s4v2" / "query_selection.json"))
    a = ap.parse_args()
    res = {}
    for name in a.groups.split(","):
        d = Path(a.cache) / name
        g = dict(np.load(d / "group.npz"))
        meta = json.loads((d / "meta.json").read_text())
        clip = po_episode.load_clip(Path("/tmp/kwang-mnt/pointodyssey/v2/val") / meta["scene"], meta["start"], 2, load_depth=False)
        X = clip["X"][:, g["ids"]]
        c_a = np.linalg.inv(clip["T_cw"])[:T, :3, 3][:, None]
        c_b = np.linalg.inv(g["T_b"])[:T, :3, 3][:, None]
        va, vb = X[:T] - c_a, X[:T] - c_b
        ang = np.degrees(np.arccos(np.clip((va * vb).sum(-1) / (np.linalg.norm(va, axis=-1) * np.linalg.norm(vb, axis=-1) + 1e-12), -1, 1))).T
        rel = s4g.source_reliability(g)
        dyn = g["dyn"].astype(bool)
        Vs = g["V_src"].astype(bool)
        nt = (np.arange(Vs.shape[1]) != 0)[None]
        zbar = np.maximum(g["Y_src"][:, :1, 2], P.z_min)
        md = Vs & nt & Vs[:, :1]
        m_i = np.where(md.any(1), ((np.linalg.norm(g["Y_src"] - g["Y_src"][:, :1], axis=-1) / zbar) * md).sum(1) / np.maximum(md.sum(1), 1), np.nan)
        pos_ok = rel["src_vis0"] & (np.nan_to_num(np.maximum(rel["e_i"], rel["l_i"]), nan=np.inf) <= 0.15)
        low = dyn & (np.nan_to_num(m_i, nan=0.0) < 0.03)
        disp_ok = np.nan_to_num(rel["d_i"] / np.maximum(m_i, 1e-9), nan=np.inf) <= 0.5
        S = pos_ok & (~dyn | low | disp_ok)
        V = g["V"].astype(bool)
        masks = {"N": V, "S": V & S[:, None], "SV": V & S[:, None] & (ang <= a.max_view_deg)}
        files = sorted(d.glob("cand_*.npz"))
        C = [dict(np.load(f)) for f in files]
        seeds = [int(f.stem.split("_")[1]) for f in files]
        ref = json.loads((d / "alignment_ref.json").read_text())
        st, dy = s4c.ref_values(ref, seeds, "static"), s4c.ref_values(ref, seeds, "dynamic")
        vst, vdy = s4c.ref_validity(st), s4c.ref_validity(dy)
        out = {"n_candidates": len(C), "ref_static": vst, "ref_dynamic": vdy, "configs": {}}
        for cname, Vm in masks.items():
            def rew(M, part=None):
                if part is not None:
                    M = M & (part[:, None])
                if M.sum() == 0:
                    return [np.nan] * len(C)
                return [reward_d4rt_gt(c["pr"], g["Y"], M, 0, P, c["pl"], g["Y_loc"])["r"] for c in C]
            r_all, r_sta, r_dyn = rew(Vm), rew(Vm, ~dyn), rew(Vm, dyn)
            rng = np.random.default_rng(1)
            taus = []
            anchors = np.flatnonzero(Vm.any(1))
            for _ in range(a.splits):
                perm = rng.permutation(anchors)
                h1, h2 = np.zeros(len(dyn), bool), np.zeros(len(dyn), bool)
                h1[perm[: len(perm) // 2]] = True; h2[perm[len(perm) // 2:]] = True
                taus.append(s4c.kendall(rew(Vm, h1), rew(Vm, h2))[0])
            ent = {"coverage": {"dynamic_anchors": float(Vm[dyn].any(1).mean()) if dyn.any() else None,
                                "static_anchors": float(Vm[~dyn].any(1).mean()) if (~dyn).any() else None,
                                "dynamic_V": float(Vm[dyn].sum() / max(V[dyn].sum(), 1)), "static_V": float(Vm[~dyn].sum() / max(V[~dyn].sum(), 1)),
                                "n_dynamic_anchors": int(Vm[dyn].any(1).sum()), "n_static_anchors": int(Vm[~dyn].any(1).sum())},
                   "split_half_tau": float(np.nanmean(taus))}
            for lab, rr, rv in (("full_vs_static", r_all, st), ("full_vs_dynamic", r_all, dy), ("static_part_vs_static", r_sta, st), ("dynamic_part_vs_dynamic", r_dyn, dy)):
                tv = s4c.tau_vs_ref(rr, rv)
                ent[lab] = {"tau": tv["tau"], "n": tv["n"]}
            out["configs"][cname] = ent
        res[name] = out
        print(json.dumps({"group": name, "ref_static_valid": vst["valid"], "ref_dynamic_valid": vdy["valid"],
                          **{c: {k: (v if not isinstance(v, dict) else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in e.items()} for c, e in out["configs"].items()}}), flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
