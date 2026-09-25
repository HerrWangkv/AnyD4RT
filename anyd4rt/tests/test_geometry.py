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


def _scene_two_planes():
    """Camera at origin looking +z; back plane at z=5, a front square at z=2 covering the center."""
    K = np.array([[100.0, 0, 63.5], [0, 100.0, 63.5], [0, 0, 1]])
    depth = np.full((128, 128), 5.0)
    depth[40:88, 40:88] = 2.0
    return K, np.eye(4), depth


def test_zbuffer_same_view_three_states():
    from anyd4rt.geometry import VIS_OCCLUDED, VIS_UNKNOWN, VIS_VISIBLE, zbuffer, zbuffer_visibility
    K, T, depth = _scene_two_planes()
    zb = zbuffer(depth, K, T, K, T, (128, 128), grid_wh=(64, 64))
    px = np.array([[64.0, 64.0], [64.0, 64.0], [10.0, 10.0], [200.0, 10.0]])
    z = np.array([2.0, 5.0, 5.0, 5.0])  # front surface / point behind the square / background / off-frame
    vis = zbuffer_visibility(zb, px, z, (128, 128))
    assert vis.tolist() == [VIS_VISIBLE, VIS_OCCLUDED, VIS_VISIBLE, VIS_UNKNOWN]


def test_zbuffer_unknown_where_source_never_saw():
    from anyd4rt.geometry import VIS_UNKNOWN, zbuffer, zbuffer_visibility, orbit_camera
    K, T, depth = _scene_two_planes()
    depth[:, :64] = 0.0  # source has no depth on the left half
    zb = zbuffer(depth, K, T, K, T, (128, 128), grid_wh=(64, 64))
    vis = zbuffer_visibility(zb, np.array([[10.0, 64.0]]), np.array([5.0]), (128, 128))
    assert vis[0] == VIS_UNKNOWN


def test_orbit_camera_identity_and_pivot():
    from anyd4rt.geometry import orbit_camera
    _, T = _cam()
    Ts = np.stack([T, T])
    assert np.allclose(orbit_camera(Ts, np.array([0.0, 0, 4]), [0, 1, 0], 0.0), Ts)
    Tb = orbit_camera(Ts, np.array([0.0, 0, 4]), [0, 1, 0], 30.0)
    c_a, c_b = np.linalg.inv(T)[:3, 3], np.linalg.inv(Tb[0])[:3, 3]
    p = np.array([0.0, 0, 4])
    assert np.isclose(np.linalg.norm(c_a - p), np.linalg.norm(c_b - p))  # stays on the orbit around the pivot
    # the pivot projects to the same pixel in both views (camera rotates about it)
    K = np.eye(3)
    pa, _ = project(K, transform(T, p)); pb, _ = project(K, transform(Tb[0], p))
    assert np.allclose(pa, pb, atol=1e-9)


def test_zbuffer_background_next_to_depth_edge_is_visible():
    from anyd4rt.geometry import VIS_VISIBLE, zbuffer, zbuffer_visibility
    K, T, depth = _scene_two_planes()  # front square spans pixels 40..87
    zb = zbuffer(depth, K, T, K, T, (128, 128), grid_wh=(64, 64))
    # background pixel 2 px outside the square: its own cell only sees the background
    vis = zbuffer_visibility(zb, np.array([[36.0, 64.0], [91.0, 64.0]]), np.array([5.0, 5.0]), (128, 128))
    assert vis.tolist() == [VIS_VISIBLE, VIS_VISIBLE]
