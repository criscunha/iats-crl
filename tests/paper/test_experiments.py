"""Cheap paper-experiment checks: frozen theory ratio, episodic sub-linear scaling.

The full powered run is on-demand via ``experiments.paper.run``.
"""

from __future__ import annotations

import numpy as np
from experiments.paper.run import _e1_separation, _e4_horizon

from iats.analysis.stats import ratio_bootstrap_ci
from iats.bandit.trackandstop import track_and_stop


def test_e1_frozen_ratio():
    # The deterministic separation ratio at k=8 is frozen at 1.958 (1% band).
    e1 = _e1_separation()
    k8 = e1["ks"].index(8)
    assert abs(e1["ratio"][k8] - 1.958) < 0.02, e1["ratio"][k8]


def test_e4_episodic_scaling_is_sublinear():
    # Load-bearing: episode count grows sub-linearly in H (max_h), not linearly (sum_h).
    e4 = _e4_horizon(seeds=10, horizons_sim=[1, 8])
    assert 1.0 < e4["sim_growth"] < 4.0, e4["sim_growth"]  # 8 would be additive
    assert all(
        g["pass"] for g in e4["gates"] if "flat" in g["name"] or "linear" in g["name"]
    )


def test_control_is_near_one():
    # Permuted arms (same two arms reversed): speed-up ~1 -- a control that CAN fail.
    ia = np.array(
        [
            track_and_stop([1.0, 0.5], delta=0.01, rng=np.random.default_rng(s))[1]
            for s in range(20)
        ],
        dtype=float,
    )
    ctrl = np.array(
        [
            track_and_stop([0.5, 1.0], delta=0.01, rng=np.random.default_rng(s))[1]
            for s in range(20)
        ],
        dtype=float,
    )
    ratio, _, _ = ratio_bootstrap_ci(ctrl, ia, n_boot=1000)
    assert 0.8 <= ratio <= 1.25, ratio
