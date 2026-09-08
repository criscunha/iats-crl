"""The layered-MDP lift composes per-stage T*, and the separation survives.

The load-bearing check is `test_episodic_is_max_not_sum`: a direct brute-force of
`sup_w min_h g_h(w_h)` over the product polytope must return the bottleneck `max_h`,
not the additive `sum_h`. The episodic Track-and-Stop tests cover per-stage freezing.
"""

from __future__ import annotations

import numpy as np
import pytest

import iats.mdp.layered as layered
from iats.bandit.chartime import char_time
from iats.bandit.instances import chain_with_branches
from iats.bandit.trackandstop import _glr
from iats.mdp.layered import (
    brute_force_episodic_time,
    episodic_track_and_stop,
    mdp_char_time,
    mdp_separation,
)

_GLR_LOG: list[tuple[int, int, int]] = []


def _spy_glr(mu_hat: np.ndarray, counts: np.ndarray, sigma: float) -> tuple[float, int]:
    """Record ``(n_arms, episode, empirical best)`` at every GLR check, then delegate.

    Args:
        mu_hat: Empirical arm means of the stage being checked.
        counts: Per-arm pull counts (their sum is the current episode count).
        sigma: Gaussian standard deviation.

    Returns:
        The real ``(Z, best)`` from :func:`iats.bandit.trackandstop._glr`.
    """
    z, best = _glr(mu_hat, counts, sigma)
    _GLR_LOG.append((len(mu_hat), int(round(float(counts.sum()))), best))
    return z, best


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
    # The episodic rollout identifies the best arm at every stage (no errors on these
    # seeds); every stage's stop episode is at most the episode count.
    stage = [1.0, 0.5]
    errors = 0
    for s in range(6):
        chosen, n, stops = episodic_track_and_stop(
            [stage, stage], delta=0.02, rng=np.random.default_rng(s), realloc_every=50
        )
        errors += int(chosen != [0, 0])
        assert 2 <= n < 200_000
        assert max(stops) == n and min(stops) >= 2
    assert errors == 0  # gap 0.5 at delta=0.02 -> reliably correct over these seeds


def test_per_stage_freezing_fixes_the_empirical_best_at_freeze_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop episodes are <= n_episodes; each frozen arm is the empirical best then.

    Args:
        monkeypatch: Swaps :func:`iats.mdp.layered._glr` for a recording spy.
    """
    monkeypatch.setattr(layered, "_glr", _spy_glr)
    stages = [[1.0, 0.0], [1.0, 0.5, 0.2]]  # 2-arm fast stage, 3-arm slow stage
    for threshold in ("analysed", "calibrated"):
        _GLR_LOG.clear()
        chosen, n, stops = episodic_track_and_stop(
            stages,
            delta=0.05,
            rng=np.random.default_rng(7),
            realloc_every=50,
            threshold=threshold,
        )
        assert all(3 <= s <= n for s in stops) and max(stops) == n, (stops, n)
        assert stops[0] < stops[1], stops  # the easier stage freezes first
        for h, arms in enumerate(stages):
            checks = [
                best for k, ep, best in _GLR_LOG if k == len(arms) and ep == stops[h]
            ]
            assert checks == [chosen[h]], (h, checks, chosen)
        # No GLR check happens on a frozen stage after its freeze episode.
        assert not [1 for k, ep, _ in _GLR_LOG if k == 2 and ep > stops[0]]


def test_episodic_deterministic_and_capped() -> None:
    """Same seed -> same output; a capped run reports the cap for unfrozen stages."""
    kw = {"delta": 0.05, "realloc_every": 50}
    a = episodic_track_and_stop([[1.0, 0.0]] * 2, rng=np.random.default_rng(5), **kw)
    b = episodic_track_and_stop([[1.0, 0.0]] * 2, rng=np.random.default_rng(5), **kw)
    assert a == b and a[0] == [0, 0]
    chosen, n, stops = episodic_track_and_stop(
        [[1.0, 0.9], [1.0, 0.9]],
        delta=1e-3,
        rng=np.random.default_rng(0),
        max_episodes=5,
    )
    assert n == 5 and stops == [5, 5] and len(chosen) == 2
    with pytest.raises(ValueError, match="threshold"):
        episodic_track_and_stop(
            [[1.0, 0.0]], rng=np.random.default_rng(0), threshold="bogus"
        )


def test_separation_grows_with_branches():
    # At fixed H, the per-stage separation slope makes the ratio grow with branch count.
    prev = 0.0
    for k in (0, 2, 4, 8):
        ratio = mdp_separation([chain_with_branches(n_branches=k)] * 4)["ratio"]
        assert ratio >= prev
        prev = ratio
    assert prev > 1.0
