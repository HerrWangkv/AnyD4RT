import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("s4_common", Path(__file__).resolve().parents[2] / "scripts" / "s4_common.py")
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)


def test_zero_shift_is_kept_and_none_is_missing():
    ref = {"0": {"static": 0.0, "dynamic": None}, "1": {"static": 3.0, "dynamic": 2.0}}
    st = c.ref_values(ref, [0, 1], "static")
    dy = c.ref_values(ref, [0, 1], "dynamic")
    assert st.tolist() == [0.0, 3.0]
    assert np.isnan(dy[0]) and dy[1] == 2.0


def test_validity_counts_missing_and_saturation_and_spread():
    v = c.ref_validity(np.array([1, 2, 3, 4, 5, 6, 31, np.nan], float))
    assert v["n_ok"] == 6 and v["n_missing"] == 1 and v["n_saturated"] == 1 and v["valid"]
    v = c.ref_validity(np.array([1, 2, 3, 4, 5, np.nan, np.nan, 31], float))  # missing is not "unsaturated"
    assert not v["valid"] and v["reason"] == "too few usable candidates"
    v = c.ref_validity(np.zeros(8))  # theta = 0: all 0 px
    assert not v["valid"] and v["reason"].startswith("no spread")


def test_kendall_ignores_missing_and_reports_n():
    t, n = c.kendall([1, 2, np.nan, 4], [1, 2, 3, 4])
    assert n == 3 and t == 1.0
    r = c.tau_vs_ref([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], np.array([1, 2, 3, 4, 5, 6, 7, np.nan]))
    assert r["n"] == 7 and r["tau"] == 1.0 and r["pvalue"] is not None  # higher reward <-> smaller shift


def test_kendall_uses_tau_b_with_ties():
    # The old concordant-pair ratio dropped all tied pairs and returned 1.0.
    # Standard tau-b applies the tie correction and returns 0.5 here.
    t, n = c.kendall([1, 1, 2], [1, 2, 2])
    assert n == 3 and np.isclose(t, 0.5)
