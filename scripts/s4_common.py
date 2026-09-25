"""Shared helpers for the S4 offline analyses: critic-free reference handling and Kendall tau with missing values.

Reference validity is decided separately for the static and the dynamic reference of a group:
  a candidate's value is OK if it is present (not None / NaN) and below the search saturation (< SAT px);
  the reference is valid if >= VALID_FRAC of the group's candidates are OK AND the OK values actually vary
  (spread >= MIN_SPREAD px; e.g. theta = 0, where every faithful candidate is at 0 px, carries no ranking information).
Missing values are never treated as unsaturated. Correlations use only the candidates whose value is OK and report n.
"""

from __future__ import annotations

import numpy as np

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


def kendall(x, y) -> tuple[float, int]:
    """Kendall tau over the entries where both x and y are finite; returns (tau, n_used). tau is NaN if n < 3."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    s = c = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if a != 0 and b != 0:
                s += a * b
                c += 1
    return (float(s / c) if c and n >= 3 else float("nan")), n


def tau_vs_ref(reward, ref_vals) -> dict:
    """Kendall tau between rewards and a critic-free reference (smaller shift = better), only on OK candidates."""
    v = ref_validity(ref_vals)
    if not v["valid"]:
        return {"tau": None, "n": 0, "validity": v}
    ok = np.isfinite(ref_vals) & (ref_vals < SAT)
    t, n = kendall(np.asarray(reward, float)[ok], -ref_vals[ok])
    return {"tau": t, "n": n, "validity": v}
