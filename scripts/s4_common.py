"""Shared helpers for the S4 offline analyses: critic-free reference handling and Kendall tau with missing values.

Reference validity is decided separately for the static and the dynamic reference of a group:
  a candidate's value is OK if it is present (not None / NaN) and below the search saturation (< SAT px);
  the reference is valid if >= VALID_FRAC of the group's candidates are OK AND the OK values actually vary
  (spread >= MIN_SPREAD px; e.g. theta = 0, where every faithful candidate is at 0 px, carries no ranking information).
Missing values are never treated as unsaturated. Correlations use only the candidates whose value is OK and report n.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

SAT = 30.0          # search range is +-32 px (step 2): >= 30 px counts as saturated
VALID_FRAC = 0.75   # >= 6 of 8 candidates OK
MIN_SPREAD = 1.0    # px


def ref_values(ref: dict, seeds, key: str) -> np.ndarray:
    """ref: {str(seed): {"static": float | None, "dynamic": float | None}}; explicit None -> NaN (0.0 stays 0.0)."""
    out = []
    for s in seeds:
        v = ref.get(str(s), {}).get(key)
        out.append(np.nan if v is None else float(v))
    return np.array(out, dtype=float)


def ref_validity(vals: np.ndarray) -> dict:
    ok = np.isfinite(vals) & (vals < SAT)
    n = len(vals)
    spread = float(np.std(vals[ok])) if ok.any() else 0.0
    valid = bool(ok.sum() >= VALID_FRAC * n and spread >= MIN_SPREAD)
    reason = None if valid else ("too few usable candidates" if ok.sum() < VALID_FRAC * n else "no spread (no ranking information)")
    return {"valid": valid, "n": n, "n_ok": int(ok.sum()), "n_missing": int((~np.isfinite(vals)).sum()),
            "n_saturated": int((np.isfinite(vals) & (vals >= SAT)).sum()), "spread_px": spread, "reason": reason}


def _kendall_result(x, y) -> tuple[float, int, float]:
    """Return standard Kendall tau-b, the finite-pair count, and its two-sided p-value."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 2:
        return float("nan"), n, float("nan")
    result = kendalltau(x, y, variant="b", nan_policy="omit")
    return float(result.statistic), n, float(result.pvalue)


def kendall(x, y) -> tuple[float, int]:
    """Kendall tau-b over entries where both x and y are finite; returns (tau_b, n_used)."""
    tau, n, _ = _kendall_result(x, y)
    return tau, n


def tau_vs_ref(reward, ref_vals) -> dict:
    """Kendall tau between rewards and a critic-free reference (smaller shift = better), only on OK candidates."""
    v = ref_validity(ref_vals)
    if not v["valid"]:
        return {"tau": None, "pvalue": None, "n": 0, "validity": v}
    ok = np.isfinite(ref_vals) & (ref_vals < SAT)
    t, n, p = _kendall_result(np.asarray(reward, float)[ok], -ref_vals[ok])
    return {"tau": t, "pvalue": p, "n": n, "validity": v}
