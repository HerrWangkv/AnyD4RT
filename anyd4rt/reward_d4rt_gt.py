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
                   pred_loc: np.ndarray | None = None, Y_loc: np.ndarray | None = None) -> dict:
    """Scalar reward for one candidate.

    pred_xc:  [N,T,3] OpenD4RT xyz for queries (t_src=t0, t_tgt=t, t_cam=t0), in cam_b(t0) frame, unknown scale.
    Y:        [N,T,3] GT reference R_b(t0)^T (X_GT(i,t) - c_b(t0)).
    V:        [N,T] bool, GT-valid & target-view visible & in frame. Fixed before seeing the candidate.
    pred_loc: [N,T,3] optional, queries (t_src=t0, t_tgt=t, t_cam=t), in cam_b(t) frame.
    Y_loc:    [N,T,3] optional, GT reference R_b(t)^T (X_GT(i,t) - c_b(t)).
    Without pred_loc the local term is off (w_loc ignored).
    """
    N, T, _ = Y.shape
    V = V.astype(bool)
    not_t0 = (np.arange(T) != t0)[None, :]
    V_d = V & not_t0 & V[:, t0:t0 + 1]
    V_loc = V & not_t0 if pred_loc is not None else np.zeros_like(V)
    n_v, n_vd, n_vl = int(V.sum()), int(V_d.sum()), int(V_loc.sum())
    out = {"n_V": n_v, "n_Vd": n_vd, "n_Vloc": n_vl, "use_d": n_vd > 0, "use_loc": n_vl > 0}
    if n_v == 0:
        out.update(r=None, skipped="empty V")
        return out

    # Scale from anchor time only (§3.5): Q0 by GT, V+ subset where prediction is fit for the fit.
    Q0 = V[:, t0] & (Y[:, t0, 2] > p.z_min)
    finite = np.isfinite(pred_xc).all(-1)
    anchor_ok = finite[:, t0] & (pred_xc[:, t0, 2] > p.z_min)
    Vplus = Q0 & anchor_ok
    s = float(np.median(Y[Vplus, t0, 2] / pred_xc[Vplus, t0, 2])) if Vplus.any() else float("nan")
    out.update(n_Q0=int(Q0.sum()), n_Vplus=int(Vplus.sum()), s=s)
    if Vplus.sum() < max(p.n_min, p.kappa * Q0.sum()) or not (np.isfinite(s) and s > 0):
        out.update(r=-p.c * (p.w_p + p.w_d * (n_vd > 0) + p.w_loc * (n_vl > 0)), scale_valid=False,
                   e_mean=p.c, d_mean=p.c if n_vd else None, eloc_mean=p.c if n_vl else None)
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

    e_mean = float(e[V].mean())
    d_mean = float(d[V_d].mean()) if n_vd else None
    eloc_mean = float(e_loc[V_loc].mean()) if n_vl else None
    r = -(p.w_p * e_mean + (p.w_d * d_mean if n_vd else 0.0) + (p.w_loc * eloc_mean if n_vl else 0.0))
    out.update(r=r, scale_valid=True, e_mean=e_mean, d_mean=d_mean, eloc_mean=eloc_mean, e=e, d=d, e_loc=e_loc)
    # Diagnostics, per term over its own set: invalid predictions, and all entries sitting at the cap c
    # (invalid ones plus finite errors >= c). A high cap fraction means the term is saturated.
    for name, err, inv, mask, n in (("e", e, inv_e, V, n_v), ("d", d, inv_d, V_d, n_vd), ("eloc", e_loc, inv_l, V_loc, n_vl)):
        out[f"n_invalid_{name}"] = int((inv & mask).sum()) if n else 0
        out[f"frac_capped_{name}"] = float((err[mask] >= p.c).mean()) if n else None
    return out
