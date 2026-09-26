"""D4RT-GT branch reward (README §3.5, §7.2).

All sets that define denominators (V, Q0, V_d, V_loc) depend on GT only and are shared by every candidate in a group.
Model outputs only decide which entries get the cap c, and whether the scale fit is valid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardParams:
    # All values are placeholders ([待定] in README) until S1 calibrates them.
    z_min: float = 1e-3
    c: float = 1.0
    n_min: int = 32
    kappa: float = 0.5
    w_p: float = 1.0
    w_d: float = 1.0
    w_loc: float = 1.0


def reward_d4rt_gt(pred_xc: np.ndarray, Y: np.ndarray, V: np.ndarray, t0: int, p: RewardParams = RewardParams(),
                   pred_loc: np.ndarray | None = None, Y_loc: np.ndarray | None = None,
                   anchor_gate: np.ndarray | None = None, balance: np.ndarray | None = None,
                   term_masks: dict[str, np.ndarray] | None = None,
                   scale_mask: np.ndarray | None = None,
                   scale_override: float | None = None) -> dict:
    """Scalar reward for one candidate.

    pred_xc:  [N,T,3] OpenD4RT xyz for queries (t_src=t0, t_tgt=t, t_cam=t0), in cam_b(t0) frame, unknown scale.
    Y:        [N,T,3] GT reference R_b(t0)^T (X_GT(i,t) - c_b(t0)).
    V:        [N,T] bool, GT-valid & target-view visible & in frame. Fixed before seeing the candidate.
    pred_loc: [N,T,3] optional, queries (t_src=t0, t_tgt=t, t_cam=t), in cam_b(t) frame.
    Y_loc:    [N,T,3] optional, GT reference R_b(t)^T (X_GT(i,t) - c_b(t)).
    Without pred_loc the local term is off (w_loc ignored).
    anchor_gate: [N] bool, optional. Backward-compatible combined gate: anchors with False are removed from V,
                 and therefore from the scale fit and every denominator. New analyses should use term_masks instead.
    balance:     [N] bool (dynamic flag from GT motion), optional. Each term is the mean of its static-group mean and
                 its dynamic-group mean (one group empty -> the other group's mean) instead of the pooled mean.
    term_masks: optional fixed masks for the three terms, with keys ``e``, ``d`` and ``loc``. A False entry removes
                that point-time entry only from the corresponding term; it does not remove it from the other terms.
                Masks are expected to be fixed before candidate generation and shared within a group.
    scale_mask: optional [N,T] GT-only calibration mask used to fit the scale. It is independent of term_masks.
                This is useful for the refit-vs-fixed-scale analysis.
    scale_override: optional per-candidate scale. If provided, no scale is re-fitted; scale_mask still defines the
                    anchor set used to decide which anchor-time predictions are valid.
    """
    N, T, _ = Y.shape
    V = V.astype(bool)
    if anchor_gate is not None:
        V = V & np.asarray(anchor_gate, bool)[:, None]
    not_t0 = (np.arange(T) != t0)[None, :]
    if term_masks is None:
        mask_e = np.ones_like(V)
        mask_d = np.ones_like(V)
        mask_loc = np.ones_like(V)
    else:
        missing = {"e", "d", "loc"} - set(term_masks)
        if missing:
            raise ValueError(f"term_masks is missing keys: {sorted(missing)}")
        mask_e = np.asarray(term_masks["e"], bool)
        mask_d = np.asarray(term_masks["d"], bool)
        mask_loc = np.asarray(term_masks["loc"], bool)
        if any(m.shape != V.shape for m in (mask_e, mask_d, mask_loc)):
            raise ValueError("each term mask must have the same [N,T] shape as V")

    V_e = V & mask_e
    V_d = V & not_t0 & V[:, t0:t0 + 1] & mask_d
    V_loc = V & not_t0 & mask_loc if pred_loc is not None else np.zeros_like(V)
    n_v, n_ve, n_vd, n_vl = int(V.sum()), int(V_e.sum()), int(V_d.sum()), int(V_loc.sum())
    out = {"n_V": n_v, "n_Ve": n_ve, "n_Vd": n_vd, "n_Vloc": n_vl,
           "use_e": n_ve > 0, "use_d": n_vd > 0, "use_loc": n_vl > 0}
    if n_v == 0 or (n_ve == 0 and n_vd == 0 and n_vl == 0):
        out.update(r=None, skipped="empty V")
        return out

    # Scale from a fixed, GT-only calibration set when requested. The term masks must not silently change it.
    fit_V = V if scale_mask is None else np.asarray(scale_mask, bool)
    if fit_V.shape != V.shape:
        raise ValueError("scale_mask must have the same [N,T] shape as V")
    Q0 = fit_V[:, t0] & (Y[:, t0, 2] > p.z_min)
    finite = np.isfinite(pred_xc).all(-1)
    anchor_ok = finite[:, t0] & (pred_xc[:, t0, 2] > p.z_min)
    Vplus = Q0 & anchor_ok
    if scale_override is None:
        s = float(np.median(Y[Vplus, t0, 2] / pred_xc[Vplus, t0, 2])) if Vplus.any() else float("nan")
    else:
        s = float(scale_override)
    out.update(n_Q0=int(Q0.sum()), n_Vplus=int(Vplus.sum()), s=s)
    scale_valid = (np.isfinite(s) and s > 0)
    if scale_override is None:
        scale_valid = scale_valid and Vplus.sum() >= max(p.n_min, p.kappa * Q0.sum())
    if not scale_valid:
        active_penalty = p.w_p * (n_ve > 0) + p.w_d * (n_vd > 0) + p.w_loc * (n_vl > 0)
        out.update(r=-p.c * active_penalty, scale_valid=False,
                   e_mean=p.c if n_ve else None, d_mean=p.c if n_vd else None,
                   eloc_mean=p.c if n_vl else None)
        return out

    zbar = np.maximum(Y[:, t0, 2], p.z_min)[:, None]
    anchor_bad = ~anchor_ok[:, None]

    def capped(err, pred_finite):
        # Invalid prediction -> cap; V unchanged. Anchor-level invalidity caps the whole row.
        invalid = anchor_bad | ~pred_finite | ~np.isfinite(err)
        return np.where(invalid, p.c, np.minimum(p.c, err)), invalid

    with np.errstate(invalid="ignore"):
        e, inv_e = capped(np.linalg.norm(s * pred_xc - Y, axis=-1) / zbar, finite)
        d, inv_d = capped(np.linalg.norm(s * (pred_xc - pred_xc[:, t0:t0 + 1]) - (Y - Y[:, t0:t0 + 1]), axis=-1) / zbar, finite)
        if n_vl:
            e_loc, inv_l = capped(np.linalg.norm(s * pred_loc - Y_loc, axis=-1) / zbar, np.isfinite(pred_loc).all(-1))
        else:
            e_loc = inv_l = None

    def agg(err, mask):
        if balance is None:
            return float(err[mask].mean())
        parts = [err[mask & g[:, None]] for g in (~np.asarray(balance, bool), np.asarray(balance, bool))]
        parts = [x.mean() for x in parts if x.size]
        return float(np.mean(parts))

    e_mean = agg(e, V_e) if n_ve else None
    d_mean = agg(d, V_d) if n_vd else None
    eloc_mean = agg(e_loc, V_loc) if n_vl else None
    r = -((p.w_p * e_mean if n_ve else 0.0)
           + (p.w_d * d_mean if n_vd else 0.0)
           + (p.w_loc * eloc_mean if n_vl else 0.0))
    out.update(r=r, scale_valid=True, e_mean=e_mean, d_mean=d_mean, eloc_mean=eloc_mean, e=e, d=d, e_loc=e_loc)
    # Diagnostics, per term over its own set: invalid predictions, and all entries sitting at the cap c
    # (invalid ones plus finite errors >= c). A high cap fraction means the term is saturated.
    for name, err, inv, mask, n in (("e", e, inv_e, V_e, n_ve), ("d", d, inv_d, V_d, n_vd), ("eloc", e_loc, inv_l, V_loc, n_vl)):
        out[f"n_invalid_{name}"] = int((inv & mask).sum()) if n else 0
        out[f"frac_capped_{name}"] = float((err[mask] >= p.c).mean()) if n else None
    return out


def fit_scale(pred_xc: np.ndarray, Y: np.ndarray, V: np.ndarray, t0: int,
              p: RewardParams = RewardParams()) -> dict:
    """Fit the reward scale once on a fixed GT-only calibration mask.

    The returned scale can be passed as ``scale_override`` to repeatedly score
    different term masks or query subsets without silently changing the scale.
    Candidate validity at the calibration anchor time remains candidate-specific,
    but the calibration set itself is fixed and shared.
    """
    V = np.asarray(V, bool)
    finite = np.isfinite(pred_xc).all(-1)
    q0 = V[:, t0] & (Y[:, t0, 2] > p.z_min)
    vplus = q0 & finite[:, t0] & (pred_xc[:, t0, 2] > p.z_min)
    s = float(np.median(Y[vplus, t0, 2] / pred_xc[vplus, t0, 2])) if vplus.any() else float("nan")
    valid = bool(np.isfinite(s) and s > 0 and vplus.sum() >= max(p.n_min, p.kappa * q0.sum()))
    return {"s": s, "scale_valid": valid, "n_Q0": int(q0.sum()), "n_Vplus": int(vplus.sum()),
            "q0": q0, "vplus": vplus, "anchor_ok": finite[:, t0] & (pred_xc[:, t0, 2] > p.z_min)}
