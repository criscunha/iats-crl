"""IA-Track-and-Stop: error rate, rate vs T*, thresholds, and the blind comparison.

Simulation tests (a few seconds). Seed counts kept modest and the plug-in allocation is
cached (``realloc_every=20``) where many runs are needed; the on-demand
`experiments/sweeps/achievability.py` runs the higher-fidelity sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

from iats.bandit.chartime import char_time, optimal_allocation
from iats.bandit.trackandstop import (
    analysed_threshold,
    calibrated_threshold,
    track_and_stop,
)


def _run(arms: list[float], delta: float, seeds: int) -> tuple[float, float]:
    """Run Track-and-Stop over seeds; return (error_rate, mean_tau)."""
    best = int(np.argmax(arms))
    errs = 0
    taus = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        chosen, tau, _ = track_and_stop(arms, delta=delta, rng=rng, realloc_every=20)
        errs += int(chosen != best)
        taus.append(tau)
    return errs / seeds, float(np.mean(taus))


def test_optimal_allocation_is_a_distribution():
    w = optimal_allocation([1.0, 0.5, 0.2])
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert (w > 0).all()


def test_error_rate_at_most_delta():
    # delta=0.1: empirical error must stay <= delta (small margin for noise).
    err, _ = _run([1.0, 0.5], 0.1, 20)
    assert err <= 0.15, err


def test_ia_beats_blind():
    # IA (ancestors only) stops materially faster than blind (ancestors + 4 branches).
    _, t_ia = _run([1.0, 0.5], 0.1, 10)
    _, t_blind = _run([1.0, 0.5, 0.2, 0.2, 0.2, 0.2], 0.1, 10)
    assert t_blind > 1.3 * t_ia, (t_ia, t_blind)


def test_rate_is_right_order():
    # Z(t) ~ t/T* crosses beta_an(t), so tau / beta_an(tau, delta) sits within a small
    # factor of T* (finite-delta fluctuations + forced exploration make it larger).
    delta = 0.01
    err, t = _run([1.0, 0.5], delta, 12)
    tstar = char_time([1.0, 0.5])
    rate = t / analysed_threshold(t, delta, 2)
    assert 0.8 * tstar <= rate <= 2.0 * tstar, (rate, tstar)
    assert err <= 0.05


def test_analysed_union_bound_is_at_most_delta() -> None:
    """(K-1) * sum_{n,m<=2000} exp(-beta_an(n+m, delta)) <= delta, numerically."""
    n_max = 2000
    s = np.arange(2, 2 * n_max + 1)
    # number of pairs (n, m) in [1, n_max]^2 with n + m = s
    multiplicity = np.minimum(s - 1, 2 * n_max + 1 - s)
    for k in (2, 5, 34):
        for delta in (0.1, 0.01):
            tail = np.array([np.exp(-analysed_threshold(int(t), delta, k)) for t in s])
            total = (k - 1) * float((multiplicity * tail).sum())
            assert total <= delta, (k, delta, total)
            assert total > 0.2 * delta, (k, delta, total)  # bound is not vacuous


def test_thresholds_increase_and_analysed_dominates_calibrated() -> None:
    """Both thresholds increase in t; analysed >= calibrated for every t >= 2."""
    ts = np.unique(np.concatenate([np.arange(1, 1000), np.logspace(3, 8, 200)]))
    for k in (2, 5, 34):
        for delta in (0.5, 0.1, 1e-4):
            an = np.array([analysed_threshold(float(t), delta, k) for t in ts])
            cal = np.array([calibrated_threshold(float(t), delta, k) for t in ts])
            assert (np.diff(an) > 0).all(), (k, delta)
            assert (np.diff(cal) > 0).all(), (k, delta)
            assert (an[ts >= 2] >= cal[ts >= 2]).all(), (k, delta)


def test_unknown_threshold_raises() -> None:
    """An unknown threshold name is rejected before any sampling."""
    with pytest.raises(ValueError, match="threshold"):
        track_and_stop([1.0, 0.5], rng=np.random.default_rng(0), threshold="bogus")


def test_deterministic_under_fixed_seed() -> None:
    """The same seed gives the same (chosen, tau, forced) for either threshold."""
    for threshold in ("analysed", "calibrated"):
        runs = [
            track_and_stop(
                [1.0, 0.5, 0.2],
                delta=0.05,
                rng=np.random.default_rng(3),
                realloc_every=10,
                threshold=threshold,
            )
            for _ in range(2)
        ]
        assert runs[0] == runs[1], threshold
        assert runs[0][0] == 0


def test_default_per_round_reallocation_runs() -> None:
    """The default rule (allocation recomputed every round) stops and is correct."""
    chosen, tau, forced = track_and_stop(
        [1.0, 0.0], delta=0.1, rng=np.random.default_rng(1)
    )
    assert chosen == 0 and 2 < tau < 500_000 and 0 <= forced <= tau


def test_capped_run_reports_max_steps() -> None:
    """A run that hits the cap returns tau == max_steps (how callers detect it)."""
    chosen, tau, _ = track_and_stop(
        [1.0, 0.99], delta=1e-6, rng=np.random.default_rng(0), max_steps=40
    )
    assert tau == 40 and chosen in (0, 1)
