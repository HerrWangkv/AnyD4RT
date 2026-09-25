import numpy as np

from anyd4rt.geometry import in_frame, project, scale_intrinsics, transform, uv_normalize


def _cam():
    K = np.array([[576.0, 0, 480.0], [0, 576.0, 270.0], [0, 0, 1]])
    ang = 0.3
    R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    T_cw = np.eye(4)
    T_cw[:3, :3], T_cw[:3, 3] = R, [0.1, -0.2, 3.0]
    return K, T_cw


def test_project_roundtrip():
    K, T_cw = _cam()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    px, z = project(K, transform(T_cw, X))
    assert np.all(z > 0)
    # Back-project with depth and T_wc, must recover world points.
    ray = np.linalg.solve(K, np.concatenate([px, np.ones((100, 1))], 1).T).T * z[:, None]
    assert np.allclose(transform(np.linalg.inv(T_cw), ray), X, atol=1e-9)


def test_scale_intrinsics_center_is_exact_for_pixel_centers():
    K, T_cw = _cam()
    src, dst = (960, 540), (256, 256)
    X = np.random.default_rng(1).normal(size=(50, 3))
    px_src, _ = project(K, transform(T_cw, X))
    px_dst, _ = project(scale_intrinsics(K, src, dst, "center"), transform(T_cw, X))
    s = np.array(dst) / np.array(src)
    assert np.allclose(px_dst, (px_src + 0.5) * s - 0.5)


def test_multiply_vs_center_offset():
    K, _ = _cam()
    src, dst = (960, 540), (256, 256)
    d = scale_intrinsics(K, src, dst, "multiply")[:2, 2] - scale_intrinsics(K, src, dst, "center")[:2, 2]
    s = np.array(dst) / np.array(src)
    assert np.allclose(d, -0.5 * (s - 1))


def test_uv_normalize_is_resolution_free_only_up_to_half_pixel():
    # u = x/(W-1) maps the grid 0..W-1 to [0,1]; corners are exact.
    assert np.allclose(uv_normalize(np.array([[0.0, 0.0], [959.0, 539.0]]), (960, 540)), [[0, 0], [1, 1]])
    assert in_frame(np.array([[959.0, 539.0]]), (960, 540))[0] and not in_frame(np.array([[959.5, 0.0]]), (960, 540))[0]
