#!/usr/bin/env python3
"""Audit source-side query selection on cached generated groups.

The script reports two scale protocols separately. ``refit`` preserves the
historical behaviour (fit a scale for every scored configuration/subset), while
``fixed`` fits one scale per candidate on the unfiltered N calibration set and
reuses it for N/S/SV, static/dynamic subsets, and split-half checks.

Reliability is term-specific and fixed before candidate generation: position
(``e``), displacement (``d``), and local-coordinate (``loc``) support masks are
reported independently. Low-motion dynamic anchors are not automatically
accepted for displacement. No universal significance threshold is used; all
correlations are standard Kendall tau-b values with their p-values and n.
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
from anyd4rt.reward_d4rt_gt import RewardParams, fit_scale, reward_d4rt_gt  # noqa: E402

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


def _term_masks(V, support, angle, config, subset, max_view_deg):
    """Build fixed [N,T] masks without looking at candidate predictions."""
    subset = np.asarray(subset, bool)
    base = V & subset[:, None]
    not_t0 = (np.arange(V.shape[1]) != 0)[None]
    term_base = {
        "e": base,
        "d": base & not_t0 & V[:, :1],
        "loc": base & not_t0,
    }
    if config == "N":
        anchor = {k: np.ones(len(subset), bool) for k in ("e", "d", "loc")}
        view = np.ones_like(V, bool)
    else:
        anchor = {k: np.asarray(support[k], bool) for k in ("e", "d", "loc")}
        view = np.ones_like(V, bool) if config == "S" else (angle <= float(max_view_deg))
    return {k: term_base[k] & anchor[k][:, None] & view for k in ("e", "d", "loc")}


def _calibration_mask(V, masks):
    """Union of active term anchors; used only by the per-subset refit protocol."""
    anchors = np.zeros(V.shape[0], bool)
    for mask in masks.values():
        anchors |= mask.any(1)
    return V & anchors[:, None]


def _term_base_masks(V):
    not_t0 = (np.arange(V.shape[1]) != 0)[None]
    return {"e": V, "d": V & not_t0 & V[:, :1], "loc": V & not_t0}


def _coverage(V, masks, dyn):
    base = _term_base_masks(V)
    out = {}
    for key in ("e", "d", "loc"):
        bm, mm = base[key], masks[key]
        anchor_base, anchor_kept = bm.any(1), mm.any(1)
        out[key] = {
            "entries_kept": int(mm.sum()),
            "entries_base": int(bm.sum()),
            "entries_fraction": float(mm.sum() / max(bm.sum(), 1)),
            "anchors_kept": int(anchor_kept.sum()),
            "anchors_base": int(anchor_base.sum()),
            "anchors_fraction": float(anchor_kept.sum() / max(anchor_base.sum(), 1)),
            "dynamic_anchors_fraction": float(anchor_kept[dyn].sum() / max(anchor_base[dyn].sum(), 1)) if dyn.any() else None,
            "static_anchors_fraction": float(anchor_kept[~dyn].sum() / max(anchor_base[~dyn].sum(), 1)) if (~dyn).any() else None,
        }
    return out


def _score(candidate, group, V, masks, mode, fixed_fit=None):
    if mode == "fixed":
        if fixed_fit is None:
            raise ValueError("fixed scale mode requires a precomputed fit")
        scale = fixed_fit["s"] if fixed_fit["scale_valid"] else np.nan
        return reward_d4rt_gt(candidate["pr"], group["Y"], V, 0, P, candidate["pl"], group["Y_loc"],
                              term_masks=masks, scale_mask=V, scale_override=scale)
    if mode == "refit":
        return reward_d4rt_gt(candidate["pr"], group["Y"], V, 0, P, candidate["pl"], group["Y_loc"],
                              term_masks=masks, scale_mask=_calibration_mask(V, masks))
    raise ValueError(mode)


def _score_many(C, group, V, masks, mode, fixed_fits):
    return [_score(c, group, V, masks, mode, fixed_fits[i] if mode == "fixed" else None)
            for i, c in enumerate(C)]


def _scale_summary(rows):
    vals = np.asarray([r.get("s", np.nan) for r in rows], float)
    n_valid = int(np.isfinite(vals).sum())
    return {"n": int(len(vals)), "n_valid": n_valid,
            "mean": float(np.nanmean(vals)) if n_valid else None,
            "std": float(np.nanstd(vals, ddof=1)) if n_valid > 1 else None,
            "min": float(np.nanmin(vals)) if n_valid else None,
            "max": float(np.nanmax(vals)) if n_valid else None,
            "values": [float(x) if np.isfinite(x) else None for x in vals]}


def _tau_entry(rewards, ref):
    tv = s4c.tau_vs_ref(rewards, ref)
    return {"tau_b": tv["tau"], "pvalue": tv["pvalue"], "n": tv["n"], "validity": tv["validity"]}


def _score_entry(C, group, V, masks, mode, fixed_fits, static_ref, dynamic_ref):
    rows = _score_many(C, group, V, masks, mode, fixed_fits)
    rewards = np.asarray([r["r"] if r.get("r") is not None else np.nan for r in rows], float)
    terms = {}
    for key in ("e_mean", "d_mean", "eloc_mean"):
        values = [r[key] for r in rows if r.get(key) is not None]
        terms[key] = float(np.mean(values)) if values else None
    return {
        "scale": _scale_summary(rows),
        "rewards": [float(x) if np.isfinite(x) else None for x in rewards],
        "terms_mean": terms,
        "tau_vs_ref_static": _tau_entry(rewards, static_ref),
        "tau_vs_ref_dynamic": _tau_entry(rewards, dynamic_ref),
    }, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "outputs" / "s4cache"))
    ap.add_argument("--groups", required=True)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--max-view-deg", type=float, default=10.0)
    ap.add_argument("--pos-tau", type=float, default=0.10)
    ap.add_argument("--loc-tau", type=float, default=0.10)
    ap.add_argument("--disp-abs-tau", type=float, default=0.10)
    ap.add_argument("--disp-ratio", type=float, default=0.5)
    ap.add_argument("--min-motion", type=float, default=0.03)
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
        V = g["V"].astype(bool)
        support = s4g.source_support_masks(
            g, rel, pos_tau=a.pos_tau, loc_tau=a.loc_tau, disp_abs_tau=a.disp_abs_tau,
            disp_ratio=a.disp_ratio, min_motion=a.min_motion,
        )
        files = sorted(d.glob("cand_*.npz"))
        C = [dict(np.load(f)) for f in files]
        seeds = [int(f.stem.split("_")[1]) for f in files]
        fixed_fits = [fit_scale(c["pr"], g["Y"], V, 0, P) for c in C]
        ref = json.loads((d / "alignment_ref.json").read_text())
        st, dy = s4c.ref_values(ref, seeds, "static"), s4c.ref_values(ref, seeds, "dynamic")
        out = {
            "n_candidates": len(C),
            "seeds": seeds,
            "reference_static": s4c.ref_validity(st),
            "reference_dynamic": s4c.ref_validity(dy),
            "mask_policy": {
                "all_fixed_before_candidate_generation": True,
                "position": "source src_vis0 and e_i <= pos_tau",
                "displacement": "static d_i <= disp_abs_tau; dynamic m_i >= min_motion and d_i/m_i <= disp_ratio",
                "local": "source src_vis0 and e_loc_i <= loc_tau",
                "low_motion_dynamic_is_not_accepted": True,
                "thresholds": {"pos_tau": a.pos_tau, "loc_tau": a.loc_tau, "disp_abs_tau": a.disp_abs_tau,
                               "disp_ratio": a.disp_ratio, "min_motion": a.min_motion, "max_view_deg": a.max_view_deg},
            },
            "source_support": {
                key: {"anchors": int(support[key].sum()), "fraction": float(support[key].mean())}
                for key in ("e", "d", "loc")
            },
            "low_motion_dynamic": {"anchors": int(support["low_motion_dynamic"].sum()),
                                    "fraction_of_dynamic": float(support["low_motion_dynamic"][dyn].mean()) if dyn.any() else None},
            "fixed_scale_calibration": {
                "mask": "unfiltered N V, fixed for every config and subset",
                "per_seed": {str(seed): {k: (float(v) if isinstance(v, (float, np.floating))
                                           else bool(v) if isinstance(v, (bool, np.bool_))
                                           else int(v) if isinstance(v, (int, np.integer)) else v)
                                           for k, v in fit.items() if k in ("s", "scale_valid", "n_Q0", "n_Vplus")}
                             for seed, fit in zip(seeds, fixed_fits)},
            },
            "configs": {},
        }
        rng = np.random.default_rng(1)
        for cname in ("N", "S", "SV"):
            full_masks = _term_masks(V, support, ang, cname, np.ones(len(dyn), bool), a.max_view_deg)
            subset_masks = {
                name: _term_masks(V, support, ang, cname, subset, a.max_view_deg)
                for name, subset in (("static", ~dyn), ("dynamic", dyn))
            }
            ent = {"coverage": _coverage(V, full_masks, dyn),
                   "subset_coverage": {name: _coverage(V, masks, dyn)
                                       for name, masks in subset_masks.items()},
                   "scale_modes": {}}
            for mode in ("refit", "fixed"):
                full, _ = _score_entry(C, g, V, full_masks, mode, fixed_fits, st, dy)
                ent["scale_modes"][mode] = {"full": full, "subsets": {}, "split_half_tau": {}}
                for subset_name, masks in subset_masks.items():
                    subset_entry, _ = _score_entry(C, g, V, masks, mode, fixed_fits, st, dy)
                    ent["scale_modes"][mode]["subsets"][subset_name] = subset_entry
                eligible = np.flatnonzero(np.logical_or.reduce([m.any(1) for m in full_masks.values()]))
                split_taus = []
                for _ in range(a.splits):
                    perm = rng.permutation(eligible)
                    h1 = np.zeros(len(dyn), bool); h2 = np.zeros(len(dyn), bool)
                    h1[perm[: len(perm) // 2]] = True; h2[perm[len(perm) // 2:]] = True
                    m1 = _term_masks(V, support, ang, cname, h1, a.max_view_deg)
                    m2 = _term_masks(V, support, ang, cname, h2, a.max_view_deg)
                    r1, _ = _score_entry(C, g, V, m1, mode, fixed_fits, st, dy)
                    r2, _ = _score_entry(C, g, V, m2, mode, fixed_fits, st, dy)
                    split_taus.append(s4c.kendall(r1["rewards"], r2["rewards"])[0])
                ent["scale_modes"][mode]["split_half_tau"] = {
                    "mean_tau_b": float(np.nanmean(split_taus)) if split_taus else None,
                    "values_tau_b": [float(x) if np.isfinite(x) else None for x in split_taus],
                }
            out["configs"][cname] = ent
        res[name] = out
        print(json.dumps({"group": name, "source_support": out["source_support"],
                          "configs": {c: {m: ent["scale_modes"][m]["full"]["tau_vs_ref_static"]
                                          for m in ("refit", "fixed")} for c, ent in out["configs"].items()}}, indent=1), flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
