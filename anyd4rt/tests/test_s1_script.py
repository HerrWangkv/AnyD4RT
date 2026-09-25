"""Pure-NumPy helpers of scripts/s1_checks.py (no model, no GPU)."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from anyd4rt.reward_d4rt_gt import RewardParams

_spec = importlib.util.spec_from_file_location("s1_checks", Path(__file__).resolve().parents[2] / "scripts" / "s1_checks.py")
s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s1)


def test_clip_starts_never_negative():
    span = 2 * (48 - 1) + 1  # 95
    assert s1.clip_starts(80, span, 2) == []  # 80-frame video, stride 2: no clip fits (was start -15 before)
    assert s1.clip_starts(95, span, 2) == [0]
    starts = s1.clip_starts(1669, span, 2)
    assert starts == [0, 1669 - span] and all(st + span <= 1669 for st in starts)


def test_absrel_reports_invalid_predictions():
    assert s1.absrel(np.array([1.0, np.nan]), np.array([1.0, 1.0])) == (0.0, 0.5)
    assert s1.absrel(np.array([np.nan, -1.0]), np.array([1.0, 1.0])) == (None, 0.0)


def _row(r, e_sta, depth=0.1, finite=1.0):
    rew = {"r": r, "e_mean": 0.1, "d_mean": 0.05, "eloc_mean": 0.1, "e_sta": e_sta, "scale_valid": True}
    return {"proj_err_px_median": 0.1, "depth_absrel_T48": depth, "depth_finite_T48": finite,
            "depth_absrel_T40": depth, "depth_finite_T40": finite,
            **{k: dict(rew) for k in s1.REWARD_KEYS}}


def test_summary_handles_missing_static_and_failed_scale(tmp_path):
    rows = [_row(-0.1, 0.05), _row(-0.2, None, depth=None, finite=0.0)]
    rows[1]["reward_frozen_T40"] = {"r": -0.3, "scale_valid": False}  # scale fit failed: most fields absent
    rows[0]["reward_frozen_T40"]["r"] = -0.5
    s1.write_summary(rows, tmp_path, RewardParams(), 2, [{"scene": "x", "reason": "short"}])
    out = json.loads((tmp_path / "summary.json").read_text())
    assert out["frozen_static_e_above_real"] == {"frac": 0.0, "n_compared": 1}  # row 1 has no e_sta on either side
    assert out["frozen_below_real"] == {"frac": 1.0, "n_compared": 2}
    assert out["depth_T48"]["n_clips_no_valid_prediction"] == 1 and out["depth_T48"]["valid_frac_mean"] == 0.5
    assert out["skipped"] == [{"scene": "x", "reason": "short"}]
    assert out["T40_vs_T48_common40"]["combined"]["r"]["n_compared"] == 2
