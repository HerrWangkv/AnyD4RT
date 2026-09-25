#!/usr/bin/env python3
"""Per-point factors of target-side score trustworthiness: source reconstruction error x view change (cached groups).

For every anchor i and time t in V (target, z-buffer visible):
  source error  e_i (per trajectory, real source video, one scale fit; s4g.source_reliability), binned
                low < 0.05 <= mid <= 0.15 < high, or "none" (no source measurement: not GT-visible in the source at t0);
  view change   angle between the rays from the source camera c_a(t) and the target camera c_b(t) to X_i(t), binned
                small < 5 deg <= mid <= 10 deg < large;
  boundary      the target z-buffer's 3x3 neighbourhood around the point spans a depth range larger than the visibility
                tolerance (near an occlusion boundary); reported as its own bin;
  unknown       anchor-times whose target visibility is "unknown" are not in V; only counted.
Per cell (dynamic / static separately), over the group's candidates (config A reward terms, e and e_loc):
  n             anchor-times in V;
  split-half tau  stability of the candidate ranking computed from this cell's points only (random anchor halves);
  tau vs ref    Kendall tau of the cell ranking vs the critic-free alignment reference (static cells vs the static
                shift, dynamic cells vs the dynamic shift); each reference is validated separately (s4_common), and the
                number of candidates used is reported;
  min residual  the minimum over the CURRENT candidate set of the cell's mean error. It mixes generation and critic
                errors and depends on how many candidates there are (24 for cnb theta 10, 8 elsewhere; --max-cands
                gives a same-size comparison); it is not a critic noise floor.
Also a per-group source-support report (fixed rules): position support e_i <= 0.15 and e_loc_i <= 0.15; for dynamic
anchors, displacement support d_i / m_i <= 0.5 where m_i is the anchor's mean relative GT motion over its source-
visible times, only for m_i >= 0.03 (lower-motion anchors are flagged "low motion": the ratio is unstable); counts and
time coverage (fraction of the 40 frames in V_src / V) of the supported anchors; usable if >= 64 supported anchors.
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
from anyd4rt.geometry import VIS_UNKNOWN, in_frame, project, transform, zbuffer, zbuffer_visibility  # noqa: E402
from anyd4rt.reward_d4rt_gt import RewardParams, reward_d4rt_gt  # noqa: E402

import importlib.util  # noqa: E402

_c = importlib.util.spec_from_file_location("s4_common", ROOT / "scripts" / "s4_common.py")
s4c = importlib.util.module_from_spec(_c)
_c.loader.exec_module(s4c)
_g = importlib.util.spec_from_file_location("s4g", ROOT / "scripts" / "s4g_gate_offline.py")
s4g = importlib.util.module_from_spec(_g)
_g.loader.exec_module(s4g)
P = RewardParams()
T = 40
SAT = 30.0


def src_bin(e):
    return np.where(np.isnan(e), "none", np.where(e < 0.05, "low", np.where(e <= 0.15, "mid", "high")))


def view_bin(a):
    return np.where(a < 5, "small", np.where(a <= 10, "mid", "large"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "outputs" / "s4cache"))
    ap.add_argument("--groups", required=True)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--min-cell", type=int, default=200, help="cells with fewer anchor-times are listed but not ranked")
    ap.add_argument("--max-cands", type=int, default=0, help="use only the first N candidates (0 = all)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "s4h" / "point_factors.json"))
    a = ap.parse_args()
    res = {}
    for name in a.groups.split(","):
        d = Path(a.cache) / name
        g = dict(np.load(d / "group.npz"))
        meta = json.loads((d / "meta.json").read_text())
        clip = po_episode.load_clip(Path("/tmp/kwang-mnt/pointodyssey/v2/val") / meta["scene"], meta["start"], 2)
        H, W = clip["frames"].shape[1:3]
        T_a, T_b = clip["T_cw"], g["T_b"]
        X = clip["X"][:, g["ids"]]  # [41,N,3]
        rel = s4g.source_reliability(g)
        e_src = np.where(rel["src_vis0"], rel["e_i"], np.nan)
        dyn = g["dyn"].astype(bool)
        # view change per anchor-time
        c_a = np.linalg.inv(T_a)[:T, :3, 3][:, None]
        c_b = np.linalg.inv(T_b)[:T, :3, 3][:, None]
        va, vb = X[:T] - c_a, X[:T] - c_b
        cosang = (va * vb).sum(-1) / (np.linalg.norm(va, axis=-1) * np.linalg.norm(vb, axis=-1) + 1e-12)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1))).T  # [N,T]
        # boundary / unknown from the target z-buffer (same construction as s4b)
        px, z = project(clip["K"][None, None], transform(T_b[:, None], X))
        boundary = np.zeros((len(g["ids"]), T), bool)
        unknown = np.zeros_like(boundary)
        for t in range(T):
            zb = zbuffer(clip["depth"][t], clip["K"], T_a[t], clip["K"], T_b[t], (W, H))
            st = zbuffer_visibility(zb, px[t], z[t], (W, H))
            unknown[:, t] = (st == VIS_UNKNOWN) & clip["val"][t, g["ids"]] & in_frame(px[t], (W, H))
            gh, gw = zb.shape
            pad = np.pad(zb, 1, constant_values=np.nan)
            stack = np.stack([pad[dy:dy + gh, dx:dx + gw] for dy in range(3) for dx in range(3)])
            with np.errstate(invalid="ignore"):
                fin = np.where(np.isfinite(stack), stack, np.nan)
                rng_ = np.nanmax(fin, 0) - np.nanmin(fin, 0)
            cx = np.clip((px[t][:, 0] * gw / W).astype(int), 0, gw - 1)
            cy = np.clip((px[t][:, 1] * gh / H).astype(int), 0, gh - 1)
            tol = np.maximum(0.05, 0.02 * z[t])
            boundary[:, t] = np.nan_to_num(rng_[cy, cx], nan=0.0) > tol
        V = g["V"].astype(bool)
        files = sorted(d.glob("cand_*.npz"))
        if a.max_cands:
            files = files[: a.max_cands]
        C = [dict(np.load(f)) for f in files]
        seeds = [int(f.stem.split("_")[1]) for f in files]
        outs = [reward_d4rt_gt(c["pr"], g["Y"], g["V"], 0, P, c["pl"], g["Y_loc"]) for c in C]
        E = np.stack([(o["e"] + np.where(np.isfinite(o["e_loc"]), o["e_loc"], o["e"])) / 2 for o in outs])  # [G,N,T]
        ref_path = d / "alignment_ref.json"
        ref = json.loads(ref_path.read_text()) if ref_path.exists() else {}
        st_ref = s4c.ref_values(ref, seeds, "static")
        dy_ref = s4c.ref_values(ref, seeds, "dynamic")
        val_st, val_dy = s4c.ref_validity(st_ref), s4c.ref_validity(dy_ref)
        sb = np.broadcast_to(src_bin(e_src)[:, None], V.shape)
        vb_ = view_bin(ang)
        cells = {}
        rng = np.random.default_rng(1)
        for motion_name, mm in (("dynamic", dyn), ("static", ~dyn)):
            for s_lab in ("low", "mid", "high", "none"):
                for v_lab in ("small", "mid", "large", "boundary"):
                    m = V & mm[:, None] & (sb == s_lab) & (boundary if v_lab == "boundary" else ((vb_ == v_lab) & ~boundary))
                    n = int(m.sum())
                    cell = {"n": n, "n_anchors": int(m.any(1).sum())}
                    if n >= a.min_cell:
                        score = np.array([-(E[k][m].mean()) for k in range(len(C))])
                        cell["min_residual_current_set"] = float(-score.max())
                        cell["mean_error"] = float(-score.mean())
                        anchors = np.flatnonzero(m.any(1))
                        taus = []
                        for _ in range(a.splits):
                            perm = rng.permutation(anchors)
                            h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
                            m1, m2 = m.copy(), m.copy()
                            m1[h2] = False; m2[h1] = False
                            if m1.any() and m2.any():
                                taus.append(s4c.kendall([-(E[k][m1].mean()) for k in range(len(C))], [-(E[k][m2].mean()) for k in range(len(C))])[0])
                        cell["split_half_tau"] = float(np.nanmean(taus)) if taus else None
                        tv = s4c.tau_vs_ref(score, st_ref if motion_name == "static" else dy_ref)
                        cell["tau_vs_ref"], cell["tau_vs_ref_n"] = tv["tau"], tv["n"]
                    cells[f"{motion_name}|src:{s_lab}|view:{v_lab}"] = cell
        # source-side support (fixed rules)
        Vs = g["V_src"].astype(bool)
        nt = (np.arange(Vs.shape[1]) != 0)[None]
        zbar = np.maximum(g["Y_src"][:, :1, 2], P.z_min)
        rel_mot = np.linalg.norm(g["Y_src"] - g["Y_src"][:, :1], axis=-1) / zbar
        md = Vs & nt & Vs[:, :1]
        m_i = np.where(md.any(1), (rel_mot * md).sum(1) / np.maximum(md.sum(1), 1), np.nan)
        pos_ok = rel["src_vis0"] & (np.nan_to_num(np.maximum(rel["e_i"], rel["l_i"]), nan=np.inf) <= 0.15)
        low_motion = dyn & (np.nan_to_num(m_i, nan=0.0) < 0.03)
        disp_ok = dyn & ~low_motion & rel["src_vis0"] & (np.nan_to_num(rel["d_i"] / np.maximum(m_i, 1e-9), nan=np.inf) <= 0.5)
        cov = lambda m, VV: float(VV[m].mean()) if m.any() else None  # noqa: E731
        support = {"static": {"n": int((~dyn).sum()), "n_source_measured": int((rel["src_vis0"] & ~dyn).sum()),
                              "n_position_supported": int((pos_ok & ~dyn).sum()),
                              "time_cov_source": cov(pos_ok & ~dyn, Vs), "time_cov_target": cov(pos_ok & ~dyn, V)},
                   "dynamic": {"n": int(dyn.sum()), "n_source_measured": int((rel["src_vis0"] & dyn).sum()),
                               "n_position_supported": int((pos_ok & dyn).sum()), "n_low_motion": int(low_motion.sum()),
                               "n_displacement_supported": int(disp_ok.sum()),
                               "n_position_and_displacement_supported": int((pos_ok & disp_ok).sum()),
                               "time_cov_source_pos": cov(pos_ok & dyn, Vs), "time_cov_target_pos": cov(pos_ok & dyn, V),
                               "time_cov_target_pos_disp": cov(pos_ok & disp_ok, V)}}
        usable_flags = {"static_position_scoring_usable": support["static"]["n_position_supported"] >= 64,
                        "dynamic_position_scoring_usable": support["dynamic"]["n_position_supported"] >= 64,
                        "dynamic_displacement_scoring_usable": support["dynamic"]["n_position_and_displacement_supported"] >= 64}
        res[name] = {"n_candidates": len(C), "reference_static": val_st, "reference_dynamic": val_dy,
                     "source_support": support, "usable": usable_flags,
                     "unknown_anchor_times": int(unknown.sum()), "boundary_frac_of_V": float((boundary & V).sum() / V.sum()),
                     "view_angle_deg_pct[10,50,90]": [float(x) for x in np.percentile(ang[V], [10, 50, 90])],
                     "cells": cells}
        print(json.dumps({"group": name, "n_cands": len(C), "ref_static_valid": val_st["valid"], "ref_dynamic_valid": val_dy["valid"],
                          **usable_flags, "support": support,
                          "view_pct": [round(x, 1) for x in res[name]["view_angle_deg_pct[10,50,90]"]], "boundary_frac": round(res[name]["boundary_frac_of_V"], 3)}), flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
