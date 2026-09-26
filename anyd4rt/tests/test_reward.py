import numpy as np

from anyd4rt.reward_d4rt_gt import RewardParams, fit_scale, reward_d4rt_gt

P = RewardParams(z_min=1e-3, c=1.0, n_min=4, kappa=0.5)


def _scene(n=64, t=8, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(n, t, 3))
    Y[:, 0, 2] = rng.uniform(1.0, 5.0, size=n)  # anchors in front of the camera at t0
    Y[:, 1:] = Y[:, :1] + 0.1 * rng.normal(size=(n, t - 1, 3)).cumsum(1)
    return Y, np.ones((n, t), bool)


def test_exact_up_to_scale_gives_zero():
    Y, V = _scene()
    out = reward_d4rt_gt(Y / 3.7, Y, V, t0=0, p=P)
    assert out["scale_valid"] and abs(out["s"] - 3.7) < 1e-9
    assert abs(out["r"]) < 1e-9


def test_nonfinite_prediction_keeps_V_and_is_capped():
    Y, V = _scene()
    pred = Y.copy()
    pred[:10, 3] = np.nan
    out = reward_d4rt_gt(pred, Y, V, t0=0, p=P)
    assert out["n_V"] == V.sum()
    assert np.all(out["e"][:10, 3] == P.c) and np.all(out["d"][:10, 3] == P.c)
    assert out["r"] < 0


def test_negative_z_after_t0_not_penalized():
    Y, V = _scene()
    Y[:5, 4:, 2] = -2.0  # object moves behind the t0 camera plane
    out = reward_d4rt_gt(Y.copy(), Y, V, t0=0, p=P)
    assert abs(out["r"]) < 1e-9


def test_empty_Vd_disables_displacement():
    Y, V = _scene()
    V[:, 1:] = False
    out = reward_d4rt_gt(Y.copy(), Y, V, t0=0, p=P)
    assert out["n_Vd"] == 0 and not out["use_d"] and out["d_mean"] is None
    assert np.isfinite(out["r"])


def test_scale_invalid_gets_max_penalty():
    Y, V = _scene()
    pred = Y.copy()
    pred[:40, 0, 2] = -1.0  # most anchors unfit for scale fitting -> |V+| < kappa*|Q0|
    out = reward_d4rt_gt(pred, Y, V, t0=0, p=P)
    assert out["scale_valid"] is False and out["r"] == -(P.w_p + P.w_d) * P.c


def test_denominators_do_not_depend_on_prediction():
    Y, V = _scene()
    rng = np.random.default_rng(1)
    a = reward_d4rt_gt(Y.copy(), Y, V, t0=0, p=P)
    pred = Y + rng.normal(size=Y.shape)
    pred[rng.random(pred.shape[:2]) < 0.3] = np.inf
    b = reward_d4rt_gt(pred, Y, V, t0=0, p=P)
    assert (a["n_V"], a["n_Vd"], a["n_Q0"]) == (b["n_V"], b["n_Vd"], b["n_Q0"])
    assert b["r"] < a["r"]


def _static_scene_moving_camera(n=64, t=8, seed=0):
    """Static world points, camera translating along x. Returns Y (t0 frame), Y_loc (per-frame), V."""
    rng = np.random.default_rng(seed)
    Xw = np.stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(3, 6, n)], -1)
    cam_x = np.linspace(0, 1.0, t)  # camera centers c(t) = (cam_x, 0, 0), identity rotation
    Y = np.repeat(Xw[:, None], t, 1)  # cam(t0) = world
    Y_loc = Xw[:, None] - np.stack([cam_x, np.zeros(t), np.zeros(t)], -1)[None]
    return Y, Y_loc, np.ones((n, t), bool)


def test_local_term_penalizes_not_following_camera():
    Y, Y_loc, V = _static_scene_moving_camera()
    # "Frozen" candidate: world looks static and the camera never moves -> local prediction stays at the t0 view.
    frozen_loc = np.repeat(Y_loc[:, :1], Y.shape[1], 1)
    t0_only = reward_d4rt_gt(Y.copy(), Y, V, t0=0, p=P)
    frozen = reward_d4rt_gt(Y.copy(), Y, V, t0=0, p=P, pred_loc=frozen_loc, Y_loc=Y_loc)
    follow = reward_d4rt_gt(Y.copy(), Y, V, t0=0, p=P, pred_loc=Y_loc.copy(), Y_loc=Y_loc)
    assert abs(t0_only["r"]) < 1e-9  # the t0-frame terms alone cannot see it (S1 finding)
    assert abs(follow["r"]) < 1e-9
    assert frozen["eloc_mean"] > 0.05 and frozen["r"] < follow["r"]


def test_local_term_uses_t0_scale_and_excludes_t0():
    Y, Y_loc, V = _static_scene_moving_camera()
    out = reward_d4rt_gt(Y / 2.0, Y, V, t0=0, p=P, pred_loc=Y_loc / 2.0, Y_loc=Y_loc)
    assert abs(out["s"] - 2.0) < 1e-9 and abs(out["r"]) < 1e-9
    assert out["n_Vloc"] == V[:, 1:].sum()


def test_local_nonfinite_capped_and_empty_Vloc_disabled():
    Y, Y_loc, V = _static_scene_moving_camera()
    bad = Y_loc.copy()
    bad[:8, 2] = np.nan
    out = reward_d4rt_gt(Y.copy(), Y, V, t0=0, p=P, pred_loc=bad, Y_loc=Y_loc)
    assert np.all(out["e_loc"][:8, 2] == P.c) and out["n_Vloc"] == V[:, 1:].sum()
    V1 = V.copy()
    V1[:, 1:] = False
    out = reward_d4rt_gt(Y.copy(), Y, V1, t0=0, p=P, pred_loc=Y_loc.copy(), Y_loc=Y_loc)
    assert not out["use_loc"] and out["eloc_mean"] is None and np.isfinite(out["r"])


def test_scale_invalid_penalty_includes_local_term():
    Y, Y_loc, V = _static_scene_moving_camera()
    pred = Y.copy()
    pred[:40, 0, 2] = -1.0
    out = reward_d4rt_gt(pred, Y, V, t0=0, p=P, pred_loc=Y_loc.copy(), Y_loc=Y_loc)
    assert out["scale_valid"] is False and out["r"] == -P.c * (P.w_p + P.w_d + P.w_loc)


def test_cap_diagnostics_count_large_finite_errors():
    Y, V = _scene()
    pred = Y.copy()
    pred[:24, 2:] += 100.0  # finite but far beyond the cap
    out = reward_d4rt_gt(pred, Y, V, t0=0, p=P)
    assert out["n_invalid_e"] == 0
    assert np.isclose(out["frac_capped_e"], (24 * (Y.shape[1] - 2)) / V.sum())
    pred[:4, 5] = np.nan
    out = reward_d4rt_gt(pred, Y, V, t0=0, p=P)
    assert out["n_invalid_e"] == 4 and out["n_invalid_d"] == 4


def test_cap_diagnostics_local_term():
    Y, Y_loc, V = _static_scene_moving_camera()
    bad = Y_loc + 50.0
    bad[:3, 1] = np.nan
    out = reward_d4rt_gt(Y.copy(), Y, V, t0=0, p=P, pred_loc=bad, Y_loc=Y_loc)
    assert out["n_invalid_eloc"] == 3 and out["frac_capped_eloc"] == 1.0
    assert out["frac_capped_e"] == 0.0


def test_anchor_gate_removes_anchors_everywhere():
    Y, V = _scene()
    pred = Y.copy()
    pred[:8] += 10.0  # the first 8 anchors are badly predicted
    gate = np.ones(len(Y), bool)
    gate[:8] = False
    full = reward_d4rt_gt(pred, Y, V, t0=0, p=P)
    gated = reward_d4rt_gt(pred, Y, V, t0=0, p=P, anchor_gate=gate)
    ref = reward_d4rt_gt(pred[8:], Y[8:], V[8:], t0=0, p=P)
    assert gated["n_V"] == V[8:].sum() and gated["n_Q0"] == ref["n_Q0"]
    assert np.isclose(gated["r"], ref["r"]) and gated["r"] > full["r"]


def test_balance_equal_weights_static_dynamic():
    Y, V = _scene(n=64)
    dyn = np.zeros(64, bool)
    dyn[:16] = True  # 16 dynamic, 48 static
    pred = Y.copy()
    pred[:16, 1:] += 0.5  # only dynamic anchors are wrong after t0
    out = reward_d4rt_gt(pred, Y, V, t0=0, p=P, balance=dyn)
    e = out["e"]
    expect = 0.5 * (e[16:][V[16:]].mean() + e[:16][V[:16]].mean())
    assert np.isclose(out["e_mean"], expect)
    pooled = reward_d4rt_gt(pred, Y, V, t0=0, p=P)
    assert out["e_mean"] > pooled["e_mean"]  # the minority group now counts for half
    only_static = reward_d4rt_gt(pred, Y, V, t0=0, p=P, balance=np.zeros(64, bool))
    assert np.isclose(only_static["e_mean"], pooled["e_mean"])  # one empty group -> plain mean


def test_term_masks_are_independent():
    Y, Y_loc, V = _static_scene_moving_camera(n=64)
    pred = Y.copy()
    pred[:8, 1:] += 1.0
    masks = {"e": V.copy(), "d": V.copy(), "loc": V.copy()}
    masks["e"][:8] = False
    masks["loc"][8:16] = False
    out = reward_d4rt_gt(pred, Y, V, t0=0, p=P, pred_loc=Y_loc.copy(), Y_loc=Y_loc, term_masks=masks)
    assert out["n_Ve"] == int(masks["e"].sum())
    assert out["n_Vloc"] == int((masks["loc"] & (np.arange(V.shape[1]) != 0)[None]).sum())
    assert out["n_Vd"] == int((masks["d"] & (np.arange(V.shape[1]) != 0)[None] & V[:, :1]).sum())
    # Excluding an anchor from e must not remove it from displacement.
    full_d = reward_d4rt_gt(pred, Y, V, t0=0, p=P, pred_loc=Y_loc.copy(), Y_loc=Y_loc)
    assert out["d_mean"] == full_d["d_mean"]


def test_fixed_scale_reuses_unfiltered_calibration():
    Y, V = _scene()
    pred = Y / 2.5
    fit = fit_scale(pred, Y, V, 0, P)
    subset = np.ones(len(Y), bool)
    subset[:16] = False
    masks = {k: V & subset[:, None] for k in ("e", "d", "loc")}
    out = reward_d4rt_gt(pred, Y, V, t0=0, p=P, term_masks=masks,
                         scale_mask=V, scale_override=fit["s"])
    assert fit["scale_valid"] and abs(fit["s"] - 2.5) < 1e-9
    assert abs(out["s"] - fit["s"]) < 1e-9 and abs(out["r"]) < 1e-9
