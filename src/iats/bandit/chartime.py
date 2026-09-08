"""Gaussian best-arm characteristic times (the Track-and-Stop sample-complexity rate).

The delta-correct best-arm identification sample complexity is ``T*(mu) log(1/delta)``,
where the characteristic time satisfies (Garivier & Kaufmann, 2016)::

    T*(mu)^{-1} = max_{w in simplex}  min_{a != best}
                      (mu_best - mu_a)^2 / (2 sigma^2 (1/w_best + 1/w_a)).

We compute ``T*`` for arbitrary mean vectors by a **derived** bisection (no scipy):
write ``c = T*^{-1}`` and ``d_a = (mu_best-mu_a)^2/(2 sigma^2)``. For a candidate ``c``,
``f_a(w) >= c`` forces ``w_a >= 1/(d_a/c - 1/w_best)``, so ``c`` is achievable iff

    min_{w_best in (c/d_min, 1)}  h(w_best) <= 1,
    h(w_best) = w_best + sum_a 1/(d_a/c - 1/w_best),

with ``h`` convex on that interval and ``min h`` increasing in ``c`` -- so a golden-
section inner minimisation inside an outer bisection on ``c`` returns ``T*`` exactly.
(Hand-check: two arms, gap ``D`` -> ``T* = 8 sigma^2 / D^2``.)

The headline use is :func:`separation`: a known graph removes the reward-non-ancestor
interventions from ``mu``, so ``T*_graph`` is flat while ``T*_blind`` grows with the
number of irrelevant branches -- an intrinsic, leading-order separation.
"""

from __future__ import annotations

import numpy as np

_GOLDEN = (5**0.5 - 1) / 2
# Iteration budget of the T* solve: 45 bisection steps on T*^{-1} (relative precision
# ~3e-14) x 60 golden-section steps on w_best (argmin to ~3e-13). This is the hot spot
# of Track-and-Stop (one solve per allocation update), so the budget is kept at what
# double precision can use rather than beyond it.
_BISECTION_ITERS = 45
_GOLDEN_ITERS = 60


def _h(w_best: float, d: np.ndarray, c: float) -> float:
    """Minimal total weight to achieve rate ``c`` at ``w_best`` (``inf`` if infeasible).

    Args:
        w_best: Allocation to the best arm.
        d: Per-suboptimal-arm KL coefficients ``(mu_best-mu_a)^2/(2 sigma^2)``.
        c: Candidate value of ``T*^{-1}``.

    Returns:
        ``w_best + sum_a 1/(d_a/c - 1/w_best)``, or ``inf`` if any denominator is
        non-positive (rate ``c`` not achievable at this ``w_best``).
    """
    inv = 1.0 / w_best
    denom = d / c - inv
    if np.any(denom <= 0):
        return float("inf")
    return float(w_best + np.sum(1.0 / denom))


def _golden(d: np.ndarray, c: float, iters: int = _GOLDEN_ITERS) -> tuple[float, float]:
    """Golden-section minimum of :func:`_h` over the feasible ``w_best`` interval.

    Args:
        d: Per-suboptimal-arm KL coefficients.
        c: Candidate value of ``T*^{-1}``.
        iters: Golden-section iterations.

    Returns:
        ``(min_value, argmin_w_best)``; ``(inf, nan)`` if the interval is empty.
    """
    dmin = float(np.min(d))
    lo, hi = c / dmin + 1e-12, 1.0 - 1e-12
    if lo >= hi:
        return float("inf"), float("nan")
    x1 = hi - _GOLDEN * (hi - lo)
    x2 = lo + _GOLDEN * (hi - lo)
    f1, f2 = _h(x1, d, c), _h(x2, d, c)
    for _ in range(iters):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - _GOLDEN * (hi - lo)
            f1 = _h(x1, d, c)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + _GOLDEN * (hi - lo)
            f2 = _h(x2, d, c)
    return (f1, x1) if f1 < f2 else (f2, x2)


def char_time(means: np.ndarray | list[float], sigma: float = 1.0) -> float:
    """Gaussian best-arm-identification characteristic time ``T*(means)``.

    Args:
        means: Arm means; the unique argmax is the best arm.
        sigma: Per-arm Gaussian standard deviation.

    Returns:
        ``T*`` (the multiplier on ``log(1/delta)``). ``inf`` if the best arm is not
        unique (the gap is zero, so it cannot be identified); ``0.0`` for a single arm.
    """
    mu = np.asarray(means, dtype=float)
    if mu.size < 2:
        return 0.0
    best = int(np.argmax(mu))
    gaps = np.array([mu[best] - mu[a] for a in range(mu.size) if a != best])
    if np.any(gaps <= 0):
        return float("inf")
    d = gaps**2 / (2 * sigma**2)
    dmin = float(np.min(d))
    # Largest c in (0, dmin) with min_w h(w; c) <= 1 is T*^{-1}; the min is increasing
    # in c.
    lo, hi = 1e-12, dmin * (1 - 1e-9)
    for _ in range(_BISECTION_ITERS):
        mid = 0.5 * (lo + hi)
        if _golden(d, mid)[0] <= 1.0:
            lo = mid
        else:
            hi = mid
    return 1.0 / lo


def optimal_allocation(
    means: np.ndarray | list[float], sigma: float = 1.0
) -> np.ndarray:
    """Optimal sampling proportions ``w*`` attaining the characteristic time.

    The allocation the Track-and-Stop sampling rule tracks. At the optimum ``c* =
    T*^{-1}``, the best arm gets the golden-section argmin ``w_best``, and each rival
    ``a`` gets the minimal weight ``1/(d_a/c* - 1/w_best)`` from ``_h``; the vector is
    renormalised to the simplex.

    Args:
        means: Arm means.
        sigma: Gaussian standard deviation.

    Returns:
        A probability vector over the arms (uniform if there are fewer than two arms
        or the best arm is not unique).
    """
    mu = np.asarray(means, dtype=float)
    n = mu.size
    if n < 2:
        return np.ones(n) / max(n, 1)
    best = int(np.argmax(mu))
    others = [a for a in range(n) if a != best]
    gaps = np.array([mu[best] - mu[a] for a in others])
    if np.any(gaps <= 0):
        return np.ones(n) / n
    d = gaps**2 / (2 * sigma**2)
    c = 1.0 / char_time(mu, sigma)
    w_best = _golden(d, c)[1]
    w = np.zeros(n)
    w[best] = w_best
    for a, da in zip(others, d, strict=True):
        w[a] = 1.0 / (da / c - 1.0 / w_best)
    total = w.sum()
    return w / total if total > 0 else np.ones(n) / n


def separation(
    relevant_means: list[float], branch_mean: float, n_branches: int, sigma: float = 1.0
) -> dict[str, float]:
    """Graph-aware vs structure-blind characteristic times for a causal-BPI instance.

    The graph-aware learner knows the ``n_branches`` reward-non-ancestor interventions
    sit at ``branch_mean`` (identified observationally, no exploration) and drops them;
    the blind learner must rule each out as a possible best arm.

    Args:
        relevant_means: Means of the reward-ancestor (relevant) interventions.
        branch_mean: Common value of every reward-non-ancestor (branch) intervention.
        n_branches: Number of branch interventions.
        sigma: Gaussian standard deviation.

    Returns:
        Mapping with ``t_graph``, ``t_blind`` and their ``ratio``.
    """
    t_graph = char_time(relevant_means, sigma)
    t_blind = char_time([*relevant_means, *([branch_mean] * n_branches)], sigma)
    return {"t_graph": t_graph, "t_blind": t_blind, "ratio": t_blind / t_graph}
