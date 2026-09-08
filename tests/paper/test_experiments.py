"""Cheap paper-experiment checks: frozen theory ratio, episodic sub-linear scaling.

The full powered run is on-demand via ``experiments.paper.run``.
"""

from __future__ import annotations

import numpy as np
from experiments.paper.run import _e1_separation, _e4_horizon
from experiments.sweeps.achievability import run_trials

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
    assert e4["sim_capped"] == 0
    assert (
        e4["stops_H8"].shape == (10, 8)
        and (e4["stops_H8"] <= e4["ns_H8"][:, None]).all()
    )


def test_run_trials_reports_capped_and_passes_threshold() -> None:
    """run_trials: per-seed arrays, a capped count, and the threshold is honoured."""
    r = run_trials([1.0, 0.0], 1.0, 0.1, 3, realloc_every=20, threshold="calibrated")
    assert r["capped"] == 0 and r["errors"] == int(np.sum(r["wrong"]))
    assert len(r["taus"]) == len(r["wrong"]) == len(r["forced"]) == 3
    # The analysed threshold is larger, so its runs stop later on the same seeds.
    r_an = run_trials([1.0, 0.0], 1.0, 0.1, 3, realloc_every=20)
    assert float(r_an["mean_tau"]) > float(r["mean_tau"])


def test_control_is_near_one():
    # Permuted arms (same two arms reversed): speed-up ~1 -- a control that CAN fail.
    ia = np.array(
        [
            track_and_stop(
                [1.0, 0.0], delta=0.01, rng=np.random.default_rng(s), realloc_every=20
            )[1]
            for s in range(20)
        ],
        dtype=float,
    )
    ctrl = np.array(
        [
            track_and_stop(
                [0.0, 1.0], delta=0.01, rng=np.random.default_rng(s), realloc_every=20
            )[1]
            for s in range(20)
        ],
        dtype=float,
    )
    ratio, _, _ = ratio_bootstrap_ci(ctrl, ia, n_boot=1000)
    assert 0.8 <= ratio <= 1.25, ratio
