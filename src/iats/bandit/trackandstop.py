"""Track-and-Stop for Gaussian best-arm identification (Garivier-Kaufmann 2016).

Sampling rule: D-tracking toward the optimal allocation
(:func:`iats.bandit.chartime.optimal_allocation`) with forced exploration. Stopping
rule: the Gaussian GLR (Chernoff) statistic crossed against the
threshold ``beta(t, delta) = log((1 + log t)/delta)``. Output the empirical best arm.

The **intervention-aware** learner runs this on the graph-restricted reward-ancestor
arms only; the **structure-blind** learner runs the same algorithm on ancestors +
branches. Both are delta-correct; the blind learner pays the larger characteristic time
(the separation), which this module lets us confirm by simulation.
"""

from __future__ import annotations

import numpy as np

from iats.bandit.chartime import optimal_allocation


def _glr(mu_hat: np.ndarray, counts: np.ndarray, sigma: float) -> tuple[float, int]:
    """Gaussian GLR stopping statistic and the current empirical best arm.

    Args:
        mu_hat: Empirical arm means.
        counts: Per-arm pull counts.
        sigma: Gaussian standard deviation.

    Returns:
        ``(Z, best)`` where ``Z = min_{a != best} (mu_best-mu_a)^2 / (2 sigma^2
        (1/N_best + 1/N_a))`` and ``best = argmax mu_hat``.
    """
    best = int(np.argmax(mu_hat))
    z = float("inf")
    for a in range(len(mu_hat)):
        if a == best:
            continue
        gap = mu_hat[best] - mu_hat[a]
        z = min(z, gap * gap / (2 * sigma**2 * (1.0 / counts[best] + 1.0 / counts[a])))
    return z, best


def track_and_stop(
    true_means: np.ndarray | list[float],
    *,
    sigma: float = 1.0,
    delta: float = 0.01,
    rng: np.random.Generator,
    max_steps: int = 500_000,
    realloc_every: int = 20,
) -> tuple[int, int, int]:
    """Run Track-and-Stop on the given arm set; return (chosen, tau, forced_pulls).

    Args:
        true_means: True means of the arms this learner considers (the
            intervention-aware learner passes ancestor arms; the blind learner passes
            ancestors + branches).
        sigma: Gaussian standard deviation.
        delta: Target error probability.
        rng: Random generator (drives the Gaussian feedback).
        max_steps: Safety cap on the stopping time.
        realloc_every: D-tracking allocation-cache period.

    Returns:
        ``(chosen_arm, tau, forced_pulls)`` -- the index output, the pull count, and how
        many pulls were forced-exploration (the ``sqrt(t)`` term, which is ``o(tau)`` --
        so the leading cost is the ``T*`` gap, not an arm-count tax).
    """
    mu = np.asarray(true_means, dtype=float)
    n = mu.size
    if n < 2:
        return 0, 1, 0
    counts = np.ones(n)
    sums = rng.normal(mu, sigma)  # one initial pull per arm
    t = n
    forced = 0
    w_cache: np.ndarray | None = None
    while t < max_steps:
        mu_hat = sums / counts
        z, best = _glr(mu_hat, counts, sigma)
        beta = np.log((1.0 + np.log(t)) / delta)
        if z >= beta:
            return best, t, forced
        if counts.min() < np.sqrt(t) - n / 2.0:  # forced exploration
            i = int(np.argmin(counts))
            forced += 1
        else:  # D-tracking toward the optimal allocation (recomputed periodically:
            # w*(mu_hat) varies slowly, so caching it cuts the cost ~realloc_every-fold)
            if w_cache is None or t % realloc_every == 0:
                w_cache = optimal_allocation(mu_hat, sigma)
            i = int(np.argmax(t * w_cache - counts))
        sums[i] += rng.normal(mu[i], sigma)
        counts[i] += 1
        t += 1
    return int(np.argmax(sums / counts)), t, forced
