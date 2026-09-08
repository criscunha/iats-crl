"""Tests for the characteristic-time solver and the separation result."""

from __future__ import annotations

import numpy as np

from iats.bandit.chartime import char_time, separation
from iats.bandit.instances import chain_with_branches


def _brute_force_char_time(
    means: list[float], sigma: float = 1.0, grid: int = 400
) -> float:
    """Independent grid solver of the max-min program (validates char_time)."""
    mu = np.array(means, dtype=float)
    best = int(np.argmax(mu))
    others = [a for a in range(len(mu)) if a != best]
    d = {a: (mu[best] - mu[a]) ** 2 / (2 * sigma**2) for a in others}
    g = np.linspace(1e-3, 1 - 1e-3, grid)
    best_val = 0.0
    for w_best in g:
        rem = 1.0 - w_best
        if rem <= 0:
            continue
        # equal split of remaining weight across the suboptimal arms (optimal by
        # symmetry only when gaps are equal; we sweep w_best, which is the binding dof
        # for few arms) -- plus a finer 2-arm exact handling below.
        if len(others) == 1:
            w_a = rem
            val = d[others[0]] / (1 / w_best + 1 / w_a)
        else:
            # sweep the split between the two suboptimal arms too (3-arm case)
            val = 0.0
            for frac in np.linspace(1e-3, 1 - 1e-3, grid):
                wa, wb = rem * frac, rem * (1 - frac)
                v = min(
                    d[others[0]] / (1 / w_best + 1 / wa),
                    d[others[1]] / (1 / w_best + 1 / wb),
                )
                val = max(val, v)
        best_val = max(best_val, val)
    return 1.0 / best_val


def test_two_arm_closed_form():
    # gap D, sigma 1 -> T* = 8 / D^2.
    for D in (0.5, 1.0, 2.0):
        assert abs(char_time([D, 0.0]) - 8.0 / D**2) < 0.05 * (8.0 / D**2)


def test_matches_brute_force_three_arm():
    means = [1.0, 0.6, 0.3]
    t = char_time(means)
    t_bf = _brute_force_char_time(means)
    assert abs(t - t_bf) / t_bf < 0.03, (t, t_bf)


def test_tie_is_unidentifiable():
    assert char_time([1.0, 1.0, 0.5]) == float("inf")


def test_separation_matches_theorem_bound():
    # Per-branch floor: T*_blind >= (2 sigma^2/Delta_0^2)*k, Delta_0 = best - branch.
    best, competitor, branch = 1.0, 0.5, 0.2
    bound_slope = 2.0 / (best - branch) ** 2  # sigma = 1
    for k in (1, 2, 4, 8, 16, 32):
        t_blind = separation([best, competitor], branch, k)["t_blind"]
        assert t_blind >= bound_slope * k, (k, t_blind, bound_slope * k)


def test_separation_diverges_with_branches():
    inst0 = chain_with_branches(n_branches=0)
    t_graph = separation(inst0.relevant_means, inst0.branch_mean, 0)["t_graph"]
    ratios = []
    for k in (1, 2, 4, 8, 16, 32):
        ratios.append(separation([1.0, 0.5], 0.2, k)["ratio"])
    # graph-aware is flat; ratio is increasing and unbounded in k.
    assert all(ratios[i] < ratios[i + 1] for i in range(len(ratios) - 1))
    assert ratios[-1] > 3 * ratios[0]
    assert t_graph > 0
