"""Statistical protocol for the experiments (numpy-only, no scipy).

The unpaired ratio-of-means bootstrap CI, the Mann-Whitney U test, the exact
Clopper-Pearson upper bound, and Holm-Bonferroni correction -- the rigor the
experiments report stopping-time / sample-complexity comparisons with. Dependency-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import erf, lgamma, sqrt

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean of their ranks).

    Args:
        values: Values to rank.

    Returns:
        Float array of average ranks, aligned with ``values``.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_v = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_v[j] == sorted_v[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def mann_whitney_u(
    a: Sequence[float], b: Sequence[float], alternative: str = "greater"
) -> float:
    """One-sided Mann-Whitney U p-value (tie-corrected normal approximation, numpy).

    The unpaired rank test used for the speed-up family: the IA and blind stopping
    times are *independent* samples (measured pairing correlation ~0), so a paired test
    is invalid.

    Args:
        a: First sample.
        b: Second sample.
        alternative: ``"greater"`` (H1: ``a`` stochastically exceeds ``b``), ``"less"``,
            or ``"two-sided"``.

    Returns:
        The p-value, or ``nan`` if either sample is empty or the variance is zero.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    na, nb = x.size, y.size
    if na == 0 or nb == 0:
        return float("nan")
    combined = np.concatenate([x, y])
    ranks = _rankdata(combined)
    u_a = float(np.sum(ranks[:na])) - na * (na + 1) / 2.0
    mean_u = na * nb / 2.0
    n = na + nb
    _, counts = np.unique(combined, return_counts=True)
    tie = float(np.sum(counts**3 - counts))
    var_u = na * nb / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if var_u <= 0:
        return float("nan")
    sd = np.sqrt(var_u)
    # Continuity correction shrinks U_a toward its mean *toward the tail being tested*:
    # subtract 0.5 for the upper tail (P[U >= u]), add 0.5 for the lower (P[U <= u]).
    p_greater = 0.5 * (1.0 - erf((u_a - mean_u - 0.5) / sd / sqrt(2.0)))
    p_less = 0.5 * (1.0 + erf((u_a - mean_u + 0.5) / sd / sqrt(2.0)))
    if alternative == "greater":
        return float(min(1.0, max(0.0, p_greater)))
    if alternative == "less":
        return float(min(1.0, max(0.0, p_less)))
    return float(min(1.0, 2.0 * min(p_greater, p_less)))


def ratio_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Unpaired bootstrap 95% CI for the ratio-of-means ``mean(a) / mean(b)``.

    Resamples ``a`` and ``b`` *independently* (they are not paired) and takes the ratio
    of resampled means. The headline speed-up estimator (``a`` = blind, ``b`` = IA).

    Args:
        a: Numerator sample.
        b: Denominator sample.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed.

    Returns:
        ``(ratio, lo, hi)`` where ``ratio = mean(a)/mean(b)`` and ``[lo, hi]`` is the
        2.5/97.5 percentile interval; ``nan`` triple if either sample has fewer than
        two elements.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    na, nb = x.size, y.size
    ratios = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        num = x[rng.integers(0, na, na)].mean()
        den = y[rng.integers(0, nb, nb)].mean()
        ratios[i] = num / den if den != 0 else np.nan
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return float(x.mean() / y.mean()), float(lo), float(hi)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """``P(Binomial(n, p) <= k)`` via a log-space term sum (numpy-only, no scipy).

    Args:
        k: Success count threshold.
        n: Number of trials.
        p: Success probability.

    Returns:
        The lower-tail binomial CDF at ``k``.
    """
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = np.log(p), np.log1p(-p)
    total = 0.0
    for i in range(k + 1):
        log_term = (
            lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * lp + (n - i) * lq
        )
        total += float(np.exp(log_term))
    return total


def clopper_pearson_upper(k_errors: int, n: int, alpha: float = 0.05) -> float:
    """One-sided exact-binomial (Clopper-Pearson) upper bound on a proportion.

    Certifies δ-correctness: the true error probability is at most this bound with
    confidence ``1 - alpha``. The correct tool for a *probability* estimand at ``0/n``
    (where a bootstrap is degenerate); it is **not** part of the Holm speed-up family.

    Args:
        k_errors: Observed number of errors.
        n: Number of trials.
        alpha: One-sided error level (default 0.05 -> 95% upper bound).

    Returns:
        The upper bound in ``[0, 1]`` (``1 - alpha**(1/n)`` in closed form when
        ``k_errors == 0``; a bisection on the binomial tail otherwise).
    """
    if n <= 0 or k_errors >= n:
        return 1.0
    if k_errors == 0:
        return float(1.0 - alpha ** (1.0 / n))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _binom_cdf(k_errors, n, mid) > alpha:  # tail too heavy -> p must be larger
            lo = mid
        else:
            hi = mid
    return hi


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down correction over a pre-registered family.

    Args:
        pvalues: Mapping hypothesis name -> raw p-value (non-finite treated as 1.0).
        alpha: Family-wise error rate.

    Returns:
        Mapping hypothesis name -> whether it is rejected at family-wise ``alpha``.
    """
    items = sorted(
        ((k, p if np.isfinite(p) else 1.0) for k, p in pvalues.items()),
        key=lambda kv: kv[1],
    )
    m = len(items)
    reject: dict[str, bool] = {}
    still = True
    for rank, (name, p) in enumerate(items):
        if still and p <= alpha / (m - rank):
            reject[name] = True
        else:
            still = False
            reject[name] = False
    return reject
