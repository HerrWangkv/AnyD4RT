import numpy as np

from scripts.dynamic_replica_stereo_calibration import (
    _random_equal_coverage,
    _time_count_signature,
)


def test_random_control_matches_strata_and_effective_time_counts():
    dynamic = np.array([True, True, True, True, False, False, False, False])
    base = np.ones((8, 6), dtype=bool)
    base[0, 5] = False
    base[1, 4:] = False
    base[4, 5] = False
    base[6, 3:] = False

    selected = np.zeros_like(base)
    selected[0, :4] = True
    selected[1, :3] = True
    selected[4, :5] = True
    selected[6, :2] = True
    errors = np.arange(base.size, dtype=float).reshape(base.shape)

    result = _random_equal_coverage(
        errors, selected, base, dynamic, np.random.default_rng(7), draws=20
    )

    assert result["successful_draws"] == 20
    assert result["n_dynamic_anchors"] == 2
    assert result["n_static_anchors"] == 2
    assert result["exact_time_count_match"]
    assert _time_count_signature(selected, dynamic) == ((3, 4), (2, 5))
