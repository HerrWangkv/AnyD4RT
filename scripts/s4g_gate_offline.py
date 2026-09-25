#!/usr/bin/env python3
"""Offline source-side reliability gating on cached S4 groups (outputs/s4cache/*), configs A-D.

Source reliability (per GT trajectory = anchor): OpenD4RT on the REAL source video (source camera, same grid -> 256
chain), queries at the anchor's source pixel at t0; ONE scale fit over all source-visible anchors (never refit after
gating); per-trajectory mean e (over V_src), d and e_loc (over V_src, t != t0). An anchor passes the gate if it is
GT-visible in the source at t0 and max(e_i, e_loc_i) <= tau and d_i <= tau (one uniform tau per run, --taus).
The gate is carried to the target view by anchor identity and is the same for every candidate of the group.

Configs:  A no gate + pooled mean   B gate + pooled   C no gate + static/dynamic balance   D gate + balance.
Group reliability: gated anchors >= --min-anchors (and, for C/D-type balance, both groups >= --min-group or the
balance falls back to the non-empty group, reported). An unreliable group is reported as such; no low-reliability
anchors are added back.

Per group and config: retention (overall / dynamic / static / person), reward terms, ranking, split-half Kendall tau
over anchor halves, Kendall tau vs the critic-free alignment reference (colour-alignment shift of static and dynamic
anchors, computed from the cached videos), and for cnb theta 10 the agreement with the user's visual pairs.
Also: motion recovery of OpenD4RT on the real source video (GT vs predicted displacement per dynamic / person anchor).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anyd4rt.reward_d4rt_gt import RewardParams, reward_d4rt_gt  # noqa: E402

import importlib.util  # noqa: E402

_c = importlib.util.spec_from_file_location("s4_common", ROOT / "scripts" / "s4_common.py")
s4c = importlib.util.module_from_spec(_c)
_c.loader.exec_module(s4c)

P = RewardParams()
VISUAL_PAIRS = {"cnb_dlab_0215_ego2@0_theta10": [[[4, 5], [7], [0, 2, 3, 6]], [[11, 22], [14], [4, 5], [1]]]}


def kendall(x, y):
    n, s, c = len(x), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if a != 0 and b != 0:
                s += a * b; c += 1
    return s / c if c else float("nan")


def norm(v):
    v = v.astype(np.float64)
    return (v - v.mean(axis=(0, 1, 2))) / (v.std(axis=(0, 1, 2)) + 1e-6)


def align_shift(S, G, sp, tp, R=32, step=2):
    gh, gw = G.shape[:2]
    def box(img, p):
        x = np.clip(np.round(p[:, 0]).astype(int), 1, gw - 2); y = np.clip(np.round(p[:, 1]).astype(int), 1, gh - 2)
        return sum(img[y + dy, x + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    cs = box(S, sp)
    best = (0, 0, np.inf)
    for dy in range(-R, R + 1, step):
        for dx in range(-R, R + 1, step):
            d = float(np.abs(box(G, tp + np.array([dx, dy])) - cs).mean())
            if d < best[2]:
                best = (dx, dy, d)
    return float(np.hypot(best[0], best[1]))


def alignment_reference(g, video, frames=range(0, 40, 3)):
    """Median per-frame best shift (grid px) for static (GT motion < 1 cm) and dynamic anchors; critic-free."""
    S, G = norm(g["real"][:40]), norm(video[:40])
    ok = g["src_ok"] & g["V"].T  # [T,N]: visible in the source frame and in the target view at t
    out = {}
    for name, sel in (("static", g["motion"] < 0.01), ("dynamic", g["dyn"])):
        sh = []
        for t in frames:
            m = ok[t] & sel
            if m.sum() >= 20:
                sh.append(align_shift(S[t], G[t], g["px_s_grid"][t][m], g["px_t_grid"][t][m]))
        out[name] = float(np.median(sh)) if sh else None
    return out


def source_reliability(g):
    o = reward_d4rt_gt(g["pr_src"], g["Y_src"], g["V_src"], 0, P, g["pl_src"], g["Y_loc_src"])
    V = g["V_src"]
    nt = (np.arange(V.shape[1]) != 0)[None]
    def per_traj(err, mask):
        num, den = (err * mask).sum(1), mask.sum(1)
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)
    e_i = per_traj(o["e"], V)
    d_i = per_traj(o["d"], V & nt & V[:, :1])
    l_i = per_traj(o["e_loc"], V & nt)
    return {"s": o["s"], "r_source": o["r"], "e_i": e_i, "d_i": d_i, "l_i": l_i, "src_vis0": g["src_vis0"].astype(bool)}


def motion_recovery(g, rel, mask_anchor):
    """GT vs predicted displacement on the REAL source video (cam_a(t0)), over V_src entries with t != t0."""
    o = reward_d4rt_gt(g["pr_src"], g["Y_src"], g["V_src"], 0, P, g["pl_src"], g["Y_loc_src"])
    V = g["V_src"] & mask_anchor[:, None] & (np.arange(g["V_src"].shape[1]) != 0)[None] & g["V_src"][:, :1]
    if not V.any():
        return None
    gt = (g["Y_src"] - g["Y_src"][:, :1])[V]
    pr = (o["s"] * (g["pr_src"] - g["pr_src"][:, :1]))[V]
    ng, npr = np.linalg.norm(gt, axis=-1), np.linalg.norm(pr, axis=-1)
    big = ng > 0.02
    cos = (gt * pr).sum(-1)[big] / (ng[big] * npr[big] + 1e-9)
    zbar = np.maximum(g["Y_src"][:, :1, 2], P.z_min)
    return {"n_entries": int(V.sum()), "gt_motion_m_mean": float(ng.mean()), "pred_motion_m_mean": float(npr.mean()),
            "pred_over_gt_magnitude": float(npr.mean() / max(ng.mean(), 1e-9)),
            "cosine_mean_where_gt_motion>2cm": float(cos.mean()) if big.any() else None,
            "frac_entries_gt_motion>2cm": float(big.mean()),
            "d_source_mean": float(o["d"][V].mean()),
            "gt_motion_rel_mean": float((np.linalg.norm(g["Y_src"] - g["Y_src"][:, :1], axis=-1) / zbar)[V].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "outputs" / "s4cache"))
    ap.add_argument("--taus", default="0.10,0.20")
    ap.add_argument("--min-anchors", type=int, default=128)
    ap.add_argument("--min-group", type=int, default=32)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--groups", default="", help="comma-separated group dir names (default: all)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "s4g"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gdirs = sorted(d for d in Path(a.cache).iterdir() if (d / "group.npz").exists())
    if a.groups:
        gdirs = [d for d in gdirs if d.name in a.groups.split(",")]
    taus = [float(x) for x in a.taus.split(",")]
    report = {"taus": taus, "groups": {}, "source_error_distribution": {}}

    # 1) source reliability distributions (before choosing / applying any threshold)
    rels = {}
    for d in gdirs:
        g = dict(np.load(d / "group.npz"))
        rels[d.name] = source_reliability(g)
        r = rels[d.name]
        v = r["src_vis0"]
        q = lambda x: [float(np.nanpercentile(x[v], k)) for k in (25, 50, 75, 90)]  # noqa: E731
        report["source_error_distribution"][d.name] = {"s": r["s"], "r_source": r["r_source"], "n_src_vis0": int(v.sum()),
                                                        "e_i_pct[25,50,75,90]": q(r["e_i"]), "d_i_pct": q(r["d_i"]), "l_i_pct": q(r["l_i"]),
                                                        "dyn_e_i_median": float(np.nanmedian(r["e_i"][v & g["dyn"].astype(bool)])) if (v & g["dyn"].astype(bool)).any() else None,
                                                        "static_e_i_median": float(np.nanmedian(r["e_i"][v & ~g["dyn"].astype(bool)]))}
        report["groups"][d.name] = {"motion_recovery_source": {
            "dynamic": motion_recovery(g, r, g["dyn"].astype(bool)),
            "person": motion_recovery(g, r, g["person"].astype(bool)) if g["person"].any() else None}}
    print(json.dumps(report["source_error_distribution"], indent=1), flush=True)

    # 2) configs per group
    for d in gdirs:
        g = dict(np.load(d / "group.npz"))
        rel = rels[d.name]
        dyn, person = g["dyn"].astype(bool), g["person"].astype(bool)
        cands = sorted(d.glob("cand_*.npz"))
        C = [dict(np.load(f)) for f in cands]
        seeds = [int(f.stem.split("_")[1]) for f in cands]
        ref_path = d / "alignment_ref.json"
        if ref_path.exists():
            ref = {int(k): v for k, v in json.loads(ref_path.read_text()).items()}
        else:
            ref = {s: alignment_reference(g, c["video"]) for s, c in zip(seeds, C)}
            ref_path.write_text(json.dumps(ref))
        ref_static = s4c.ref_values({str(k): v for k, v in ref.items()}, seeds, "static")
        ref_dyn = s4c.ref_values({str(k): v for k, v in ref.items()}, seeds, "dynamic")
        gr = report["groups"][d.name]
        gr["seeds"] = seeds
        gr["alignment_ref"] = {s: ref[s] for s in seeds}
        V = g["V"].astype(bool)
        gates = {"none": np.ones(len(dyn), bool)}
        for tau in taus:
            ok = rel["src_vis0"] & (np.nan_to_num(np.maximum(rel["e_i"], rel["l_i"]), nan=np.inf) <= tau) & (np.nan_to_num(rel["d_i"], nan=np.inf) <= tau)
            gates[f"tau{tau:g}"] = ok
        gr["configs"] = {}
        for gname, gate in gates.items():
            for bal in (False, True):
                cfg = ("A" if not bal else "C") if gname == "none" else (("B" if not bal else "D") + f"_{gname}")
                Vg = V & gate[:, None]
                ret = {k: float(gate[m].mean()) if m.any() else None for k, m in (("all", np.ones_like(gate)), ("dynamic", dyn), ("static", ~dyn), ("person", person))}
                ret_V = {k: float(Vg[m].sum() / max(V[m].sum(), 1)) if m.any() else None for k, m in (("all", np.ones_like(gate)), ("dynamic", dyn), ("static", ~dyn), ("person", person))}
                n_dyn, n_sta = int((gate & dyn).sum()), int((gate & ~dyn).sum())
                reliable = int(gate.sum()) >= a.min_anchors
                bal_note = None
                if bal and (n_dyn < a.min_group or n_sta < a.min_group):
                    bal_note = f"balance falls back: dyn {n_dyn}, static {n_sta} (< {a.min_group})"
                b_arr = dyn & gate if bal else None
                if bal and bal_note:
                    b_arr = np.zeros_like(dyn) if n_dyn < a.min_group else np.ones_like(dyn)
                rows = []
                for c in C:
                    o = reward_d4rt_gt(c["pr"], g["Y"], V, 0, P, c["pl"], g["Y_loc"], anchor_gate=gate, balance=b_arr)
                    rows.append({k: o.get(k) for k in ("r", "e_mean", "d_mean", "eloc_mean", "s", "scale_valid")})
                rs = np.array([x["r"] for x in rows], dtype=float)
                rng = np.random.default_rng(1)
                elig = np.flatnonzero(gate)
                taus_split = []
                for _ in range(a.splits):
                    perm = rng.permutation(elig)
                    h = [perm[: len(perm) // 2], perm[len(perm) // 2:]]
                    rr = []
                    for half in h:
                        m = np.zeros_like(gate); m[half] = True
                        rr.append([reward_d4rt_gt(c["pr"], g["Y"], V, 0, P, c["pl"], g["Y_loc"], anchor_gate=m, balance=b_arr)["r"] for c in C])
                    taus_split.append(kendall(rr[0], rr[1]))
                ent = {"reliable": reliable, "n_gated_anchors": int(gate.sum()), "n_dyn": n_dyn, "n_static": n_sta, "balance_note": bal_note,
                       "retention_anchors": ret, "retention_V": ret_V,
                       "person_frac_of_V_after": float((Vg & person[:, None]).sum() / max(Vg.sum(), 1)) if person.any() else None,
                       "rewards": dict(zip(seeds, rs.tolist())), "order": [seeds[i] for i in np.argsort(-rs)],
                       "terms_mean": {k: float(np.mean([x[k] for x in rows if x[k] is not None])) for k in ("e_mean", "d_mean", "eloc_mean")},
                       "split_half_tau_mean": float(np.nanmean(taus_split)),
                       **{f"tau_vs_ref_{k}": (tv := s4c.tau_vs_ref(rs, vals))["tau"] for k, vals in (("static", ref_static), ("dynamic", ref_dyn))},
                       **{f"n_tau_{k}": s4c.tau_vs_ref(rs, vals)["n"] for k, vals in (("static", ref_static), ("dynamic", ref_dyn))},
                       "ref_static_validity": s4c.ref_validity(ref_static), "ref_dynamic_validity": s4c.ref_validity(ref_dyn)}
                if d.name in VISUAL_PAIRS:
                    pairs = set()
                    for ch in VISUAL_PAIRS[d.name]:
                        for i, j in itertools.combinations(range(len(ch)), 2):
                            pairs |= {(x, y) for x in ch[i] for y in ch[j]}
                    R = dict(zip(seeds, rs))
                    ent["visual_pairs_agree"] = f"{sum(R[x] > R[y] for x, y in pairs if x in R and y in R)}/{len(pairs)}"
                    ent["rank_of_seed1"] = ent["order"].index(1) + 1
                gr["configs"][cfg] = ent
        print(json.dumps({"group": d.name, **{k: {kk: gr["configs"][k][kk] for kk in ("reliable", "n_gated_anchors", "retention_anchors", "split_half_tau_mean", "tau_vs_ref_static", "n_tau_static", "tau_vs_ref_dynamic", "n_tau_dynamic") + (("visual_pairs_agree", "rank_of_seed1") if d.name in VISUAL_PAIRS else ())} for k in gr["configs"]}}), flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
