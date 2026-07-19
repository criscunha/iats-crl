"""IA-Track-and-Stop is delta-correct, attains T*, and beats the blind learner.

Simulation tests (a few seconds). Seed counts kept modest; the on-demand
`experiments/sweeps/achievability.py` runs the higher-fidelity sweep.
"""

from __future__ import annotations

import numpy as np

from iats.bandit.chartime import char_time, optimal_allocation
from iats.bandit.trackandstop import track_and_stop


def _run(arms: list[float], delta: float, seeds: int) -> tuple[float, float]:
    """Run Track-and-Stop over seeds; return (error_rate, mean_tau)."""
    best = int(np.argmax(arms))
    errs = 0
    taus = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        chosen, tau, _ = track_and_stop(arms, delta=delta, rng=rng)
        errs += int(chosen != best)
        taus.append(tau)
    return errs / seeds, float(np.mean(taus))


def test_optimal_allocation_is_a_distribution():
    w = optimal_allocation([1.0, 0.5, 0.2])
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert (w > 0).all()


def test_delta_correctness():
    # delta=0.1: empirical error must stay <= delta (small margin for noise).
    err, _ = _run([1.0, 0.5], 0.1, 40)
    assert err <= 0.15, err


def test_ia_beats_blind():
    # IA (ancestors only) stops materially faster than blind (ancestors + 4 branches).
    _, t_ia = _run([1.0, 0.5], 0.1, 20)
    _, t_blind = _run([1.0, 0.5, 0.2, 0.2, 0.2, 0.2], 0.1, 20)
    assert t_blind > 1.3 * t_ia, (t_ia, t_blind)


def test_rate_is_right_order():
    # tau / log(1/delta) is within a small factor of T* (converges to it as delta->0).
    delta = 0.01
    err, t = _run([1.0, 0.5], delta, 25)
    rate = t / np.log(1.0 / delta)
    tstar = char_time([1.0, 0.5])
    assert 0.8 * tstar <= rate <= 4.0 * tstar, (rate, tstar)
    assert err <= 0.05
