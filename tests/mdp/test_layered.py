"""The layered-MDP lift composes per-stage T*, and the separation survives.

The load-bearing check is `test_episodic_is_max_not_sum`: a direct brute-force of
`sup_w min_h g_h(w_h)` over the product polytope must return the bottleneck `max_h`,
not the additive `sum_h`.
"""

from __future__ import annotations

import numpy as np

from iats.bandit.chartime import char_time
from iats.bandit.instances import chain_with_branches
from iats.mdp.layered import (
    brute_force_episodic_time,
    episodic_track_and_stop,
    mdp_char_time,
    mdp_separation,
)


def test_episodic_is_max_generative_is_sum():
    # Two stages, deliberately unequal T*_h, so max != sum and the law is testable.
    stage_arms = [[1.0, 0.5], [1.0, 0.7]]
    per_stage = [char_time(a) for a in stage_arms]
    assert mdp_char_time(stage_arms, budget="episodic") == max(per_stage)
    assert abs(mdp_char_time(stage_arms, budget="generative") - sum(per_stage)) < 1e-9


def test_episodic_is_max_not_sum():
    # The brute-force sup-min over the product polytope must land on max_h, not sum_h.
    stage_arms = [[1.0, 0.5], [1.0, 0.5]]  # identical stages: sum = 2 * max
    t_max = mdp_char_time(stage_arms, budget="episodic")
    t_sum = mdp_char_time(stage_arms, budget="generative")
    t_bf = brute_force_episodic_time(stage_arms, n_samples=200_000)
    # Within a few % of max; nowhere near the (2x larger) additive sum.
    assert abs(t_bf - t_max) / t_max < 0.05, (t_bf, t_max)
    assert t_bf < 0.6 * t_sum, (t_bf, t_sum)


def test_separation_is_flat_in_horizon_episodic():
    # Identical separating stages: episodic ratio = single-stage ratio for every H.
    stage = chain_with_branches(n_branches=6)
    single = mdp_separation([stage], budget="episodic")["ratio"]
    for h in (1, 2, 4, 8, 16):
        ratio = mdp_separation([stage] * h, budget="episodic")["ratio"]
        assert abs(ratio - single) < 1e-9, (h, ratio, single)
    assert single > 1.0  # the graph advantage is real


def test_episodic_track_and_stop_is_correct():
    # The episodic rollout identifies the best arm at every stage (delta-correct).
    stage = [1.0, 0.5]
    errors = 0
    for s in range(10):
        chosen, n = episodic_track_and_stop(
            [stage, stage], delta=0.02, rng=np.random.default_rng(s), realloc_every=20
        )
        errors += int(chosen != [0, 0])
        assert 2 <= n < 200_000
    assert errors == 0  # gap 0.5 at delta=0.02 -> reliably correct over these seeds


def test_separation_grows_with_branches():
    # At fixed H, the per-stage separation slope makes the ratio grow with branch count.
    prev = 0.0
    for k in (0, 2, 4, 8):
        ratio = mdp_separation([chain_with_branches(n_branches=k)] * 4)["ratio"]
        assert ratio >= prev
        prev = ratio
    assert prev > 1.0
