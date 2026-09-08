"""Track-and-Stop for Gaussian best-arm identification (Garivier & Kaufmann, 2016).

**Sampling rule** (D-tracking with forced exploration, GK 2016 Sec. 2). Pull every arm
once. At round ``t`` with pull counts ``N(t)``: if ``min_a N_a(t) < sqrt(t) - K/2`` pull
the least-sampled arm (forced exploration); otherwise pull
``argmax_a [t * w*_a(mu_hat(t)) - N_a(t)]``, where ``w*`` is the plug-in optimal
allocation :func:`iats.bandit.chartime.optimal_allocation`. The allocation is recomputed
every round by default; ``realloc_every > 1`` caches it between recomputations (a speed
knob for ad-hoc use, not the analysed rule).

**Stopping rule.** Stop at the first ``t`` with ``Z(t) >= beta(t, delta)`` and recommend
the empirical best arm ``b = argmax mu_hat(t)``, where
``Z(t) = min_{a != b} (mu_hat_b - mu_hat_a)^2 / (2 sigma^2 (1/N_b + 1/N_a))`` is the
Gaussian GLR (Chernoff) statistic. Two thresholds are available (``threshold=``):

- ``"analysed"`` (default), :func:`analysed_threshold`::

      beta_an(t, delta) = log((K - 1) zeta(2) t^3 / delta),    zeta(2) = pi^2 / 6.

  delta-correctness: a wrong stop at time ``t`` means the empirical best ``b != a*``
  satisfies ``Z_{b,a*}(t) >= beta_an(t, delta)`` with ``mu_hat_b > mu_hat_{a*}``. With
  ``n = N_{a*}(t)`` and ``m = N_b(t)`` (``n + m <= t``, and ``beta_an`` is increasing in
  ``t``) this event is contained in ``{mu_hat_{b,m} - mu_hat_{a*,n} >= sigma sqrt(2
  beta_an(n+m, delta) (1/n + 1/m))}``, whose probability under ``mu_b <= mu_{a*}`` is at
  most ``exp(-beta_an(n+m, delta))`` by the Gaussian tail bound. Here ``mu_hat_{a,m}``
  is the mean of the first ``m`` i.i.d. draws of arm ``a`` (the stack-of-rewards
  representation), which is what makes the union over count pairs valid for any
  sampling rule. The union over the ``K - 1`` rivals and all count pairs ``n, m >= 1``
  gives ``(K-1) sum_{n,m} delta / ((K-1) zeta(2) (n+m)^3) = (delta / zeta(2))
  sum_{s >= 2} (s-1) / s^3 <= delta``.

- ``"calibrated"``, :func:`calibrated_threshold`::

      beta_cal(t, delta) = log((1 + log t)(K - 1) / delta),

  the threshold Garivier & Kaufmann use in their experiments and describe as not covered
  by their delta-correctness proof. Its delta-correctness is **not proven**; it is kept
  for empirical comparison with the analysed threshold.

The intervention-aware learner runs this on the graph-restricted reward-ancestor arms
only; the structure-blind learner runs the same algorithm on ancestors + branches and
pays the larger characteristic time (the separation this module lets us measure).
"""

from __future__ import annotations

import typing as tp
from collections.abc import Callable

import numpy as np

from iats.bandit.chartime import optimal_allocation

ZETA_2: tp.Final[float] = float(np.pi**2 / 6)
MAX_STEPS: tp.Final[int] = 500_000
THRESHOLDS: tp.Final[tuple[str, ...]] = ("analysed", "calibrated")


def analysed_threshold(t: float, delta: float, n_arms: int) -> float:
    """Analysed stopping threshold ``log((K - 1) zeta(2) t^3 / delta)``.

    Args:
        t: Current round (total pulls so far), ``t >= 1``.
        delta: Target error probability.
        n_arms: Number of arms ``K`` (``K - 1`` rivals; clamped to ``>= 1`` so a
            single-arm problem, which has nothing to identify, keeps the log finite).

    Returns:
        ``beta_an(t, delta)``; the union bound in the module docstring makes the
        stopping rule delta-correct.
    """
    return float(np.log(max(n_arms - 1, 1) * ZETA_2 * float(t) ** 3 / delta))


def calibrated_threshold(t: float, delta: float, n_arms: int) -> float:
    """Calibrated threshold ``log((1 + log t)(K - 1) / delta)`` (GK 2016 experiments).

    Args:
        t: Current round (total pulls so far), ``t >= 1``.
        delta: Target error probability.
        n_arms: Number of arms ``K`` (``K - 1`` clamped to ``>= 1`` as in
            :func:`analysed_threshold`).

    Returns:
        ``beta_cal(t, delta)``. Its delta-correctness is not proven.
    """
    return float(np.log((1.0 + np.log(t)) * max(n_arms - 1, 1) / delta))


def threshold_fn(threshold: str) -> Callable[[float, float, int], float]:
    """Resolve a threshold name to its function.

    Args:
        threshold: ``"analysed"`` or ``"calibrated"``.

    Returns:
        :func:`analysed_threshold` or :func:`calibrated_threshold`.

    Raises:
        ValueError: If ``threshold`` is not one of :data:`THRESHOLDS`.
    """
    if threshold == "analysed":
        return analysed_threshold
    if threshold == "calibrated":
        return calibrated_threshold
    raise ValueError(f"threshold must be one of {THRESHOLDS}, got {threshold!r}")


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
    max_steps: int = MAX_STEPS,
    realloc_every: int = 1,
    threshold: str = "analysed",
) -> tuple[int, int, int]:
    """Run Track-and-Stop on the given arm set; return (chosen, tau, forced_pulls).

    Args:
        true_means: True means of the arms this learner considers (the
            intervention-aware learner passes ancestor arms; the blind learner passes
            ancestors + branches).
        sigma: Gaussian standard deviation.
        delta: Target error probability.
        rng: Random generator (drives the Gaussian feedback).
        max_steps: Safety cap on the stopping time. A capped run returns the pull
            count at the cap and the empirical best arm at that time; callers detect
            it by ``tau >= max_steps`` (the ``K`` warm-up pulls happen regardless, so
            ``tau`` exceeds a cap smaller than ``K``).
        realloc_every: Rounds between recomputations of the plug-in allocation
            ``w*(mu_hat)``; ``1`` (default) is the analysed D-tracking rule, larger
            values cache the allocation for speed in ad-hoc use.
        threshold: ``"analysed"`` (default) or ``"calibrated"``; see the module
            docstring.

    Returns:
        ``(chosen_arm, tau, forced_pulls)`` -- the index recommended, the pull count at
        stopping (``>= max_steps`` if capped), and how many pulls were forced
        exploration.

    Raises:
        ValueError: If ``threshold`` is not one of :data:`THRESHOLDS`.
    """
    beta_fn = threshold_fn(threshold)
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
        if z >= beta_fn(t, delta, n):
            return best, t, forced
        if counts.min() < np.sqrt(t) - n / 2.0:  # forced exploration
            i = int(np.argmin(counts))
            forced += 1
        else:  # D-tracking toward the plug-in optimal allocation
            if w_cache is None or t % realloc_every == 0:
                w_cache = optimal_allocation(mu_hat, sigma)
            i = int(np.argmax(t * w_cache - counts))
        sums[i] += rng.normal(mu[i], sigma)
        counts[i] += 1
        t += 1
    return int(np.argmax(sums / counts)), t, forced
