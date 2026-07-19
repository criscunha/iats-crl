"""Layered causal-MDP characteristic time: composing per-stage Track-and-Stop.

Under condition **(R)** (layered-rectangular traversal, stage-local reward, branches
non-ancestral in the *unrolled* H-stage DAG), the MDP characteristic time factors into
the single-stage :func:`iats.bandit.chartime.char_time` of each stage. The
composition depends on the *budget model*:

- **Episodic** (one episode = one sample at every stage; a *shared* episode count
  ``N``): ``T*_MDP = max_h T*_h`` -- a **bottleneck**. The per-stage allocations ``w_h``
  are independent and each per-episode rate ``g_h`` depends only on ``w_h``, so
  ``sup_w min_h g_h(w_h) = min_h max_{w_h} g_h(w_h) = min_h 1/T*_h``. The separation is
  therefore **flat in H** (set by the worst stage), not amplified by depth.
- **Generative** (resettable; a *splittable* total budget): ``T*_MDP = sum_h T*_h``.
  Water-filling ``lambda_h`` over stages gives ``1/T*_MDP = 1/sum_h T*_h`` -- additive,
  ``Theta(H*k)`` for homogeneous stages.

The episodic composition being ``max_h`` rather than the additive ``sum_h`` is subtle;
:func:`brute_force_episodic_time` settles it numerically (returns ``max_h``, not
``sum_h``) as an independent cross-check.
"""

from __future__ import annotations

import numpy as np

from iats.bandit.chartime import char_time, optimal_allocation
from iats.bandit.instances import CausalBanditInstance
from iats.bandit.trackandstop import _glr


def mdp_char_time(
    stage_arms: list[list[float]], sigma: float = 1.0, budget: str = "episodic"
) -> float:
    """Layered-MDP characteristic time as a composition of per-stage ``T*_h``.

    Args:
        stage_arms: Per-stage arm-mean vectors (one list of means per stage).
        sigma: Gaussian standard deviation (shared across stages).
        budget: ``"episodic"`` (shared episode count -> ``max_h``) or ``"generative"``
            (splittable budget -> ``sum_h``).

    Returns:
        ``max_h T*_h`` if ``budget == "episodic"``, else ``sum_h T*_h``.

    Raises:
        ValueError: If ``budget`` is neither ``"episodic"`` nor ``"generative"``.
    """
    times = [char_time(arms, sigma) for arms in stage_arms]
    if budget == "episodic":
        return max(times)
    if budget == "generative":
        return float(sum(times))
    raise ValueError(f"budget must be 'episodic' or 'generative', got {budget!r}")


def mdp_separation(
    stages: list[CausalBanditInstance], sigma: float = 1.0, budget: str = "episodic"
) -> dict[str, float]:
    """Graph-aware vs structure-blind MDP characteristic times for a layered MDP.

    Each stage is a :class:`~iats.bandit.instances.CausalBanditInstance`; the
    graph-aware learner faces ``graph_arms`` per stage, the blind learner
    ``blind_arms``. The
    separation survives the lift at leading order (every branch forced binding per
    stage, Lemma 3-MDP); under the episodic budget the ratio is flat in ``H``.

    Args:
        stages: Per-stage causal-bandit instances.
        sigma: Gaussian standard deviation.
        budget: ``"episodic"`` or ``"generative"`` (see :func:`mdp_char_time`).

    Returns:
        Mapping with ``t_graph``, ``t_blind`` and their ``ratio``.
    """
    t_graph = mdp_char_time([s.graph_arms for s in stages], sigma, budget)
    t_blind = mdp_char_time([s.blind_arms for s in stages], sigma, budget)
    return {"t_graph": t_graph, "t_blind": t_blind, "ratio": t_blind / t_graph}


def _stage_rates_batch(means: list[float], w: np.ndarray, sigma: float) -> np.ndarray:
    """Per-episode GK rate ``g(w) = min_a Delta_a^2 / (2 sigma^2 (1/w_best + 1/w_a))``.

    The single-stage inner objective whose ``sup`` over the simplex is ``1/T*``,
    vectorised over a batch of allocations.

    Args:
        means: Stage arm means.
        w: Allocation batch, shape ``(n_samples, n_arms)``, each row on the simplex.
        sigma: Gaussian standard deviation.

    Returns:
        ``g(w)`` for each row, shape ``(n_samples,)``.
    """
    mu = np.asarray(means, dtype=float)
    best = int(np.argmax(mu))
    coef = (mu[best] - mu) ** 2 / (2.0 * sigma**2)  # 0 at the best arm
    inv_best = 1.0 / w[:, best]
    rates = coef[None, :] / (inv_best[:, None] + 1.0 / w)  # (n, n_arms)
    rates[:, best] = np.inf  # the best arm is not a rival
    return rates.min(axis=1)


def episodic_track_and_stop(
    stage_arms: list[list[float]],
    *,
    sigma: float = 1.0,
    delta: float = 0.01,
    rng: np.random.Generator,
    max_episodes: int = 200_000,
    realloc_every: int = 1,
) -> tuple[list[int], int]:
    """Episodic layered-MDP Track-and-Stop: one pull per stage per episode.

    A single shared episode counter ``N`` meters all stages (the episodic budget), so
    the stopping time is gated by the *slowest* stage -- the algorithm-level witness
    that the episodic characteristic time is the bottleneck ``max_h T*_h``, not the
    additive ``sum_h T*_h``. Each stage runs single-stage D-tracking toward its optimal
    allocation; the episode stops when every stage's GLR crosses ``beta(N,delta)+logH``.

    Args:
        stage_arms: Per-stage arm-mean vectors.
        sigma: Gaussian standard deviation.
        delta: Target error probability.
        rng: Random generator (drives the Gaussian feedback).
        max_episodes: Safety cap on the episode count.
        realloc_every: Per-stage allocation-cache period (1 = recompute every episode).

    Returns:
        ``(per_stage_chosen, n_episodes)`` -- the arm chosen at each stage and the
        episode count (= per-stage pulls) at stopping.
    """
    stages = [np.asarray(a, dtype=float) for a in stage_arms]
    sizes = [s.size for s in stages]
    h_count = len(stages)
    warm = max(sizes)
    counts = [np.zeros(sz) for sz in sizes]
    sums = [np.zeros(sz) for sz in sizes]
    for e in range(warm):  # warmup: every stage gets `warm` pulls, each arm >= once
        for h in range(h_count):
            i = e % sizes[h]
            sums[h][i] += rng.normal(stages[h][i], sigma)
            counts[h][i] += 1
    n = warm
    caches: list[np.ndarray | None] = [None] * h_count
    while n < max_episodes:
        beta = np.log((1.0 + np.log(n)) / delta) + np.log(h_count)
        z_min = min(
            _glr(sums[h] / counts[h], counts[h], sigma)[0] for h in range(h_count)
        )
        if z_min >= beta:  # even the cheapest single-stage flip is ruled out
            break
        for h in range(h_count):
            mu_hat = sums[h] / counts[h]
            if counts[h].min() < np.sqrt(n) - sizes[h] / 2.0:
                i = int(np.argmin(counts[h]))
            else:
                w = caches[h]
                if w is None or n % realloc_every == 0:
                    w = optimal_allocation(mu_hat, sigma)
                    caches[h] = w
                i = int(np.argmax(n * w - counts[h]))
            sums[h][i] += rng.normal(stages[h][i], sigma)
            counts[h][i] += 1
        n += 1
    chosen = [int(np.argmax(sums[h] / counts[h])) for h in range(h_count)]
    return chosen, n


def brute_force_episodic_time(
    stage_arms: list[list[float]],
    sigma: float = 1.0,
    n_samples: int = 200_000,
    seed: int = 0,
) -> float:
    """Episodic ``T*`` by direct ``sup_w min_h g_h(w_h)`` over the product polytope.

    Samples independent Dirichlet allocations per stage, pairs them as joint samples,
    takes ``min_h`` then ``max`` over samples (a lower bound on the ``sup``-``min`` that
    is tight as ``n_samples`` grows). The cross-check on **low-dimensional (2-arm)**
    stages, where it returns ``max_h T*_h``, not the additive ``sum_h T*_h``.

    Warning:
        Trustworthy only on low-dimensional stages. On multi-arm (e.g. 8-arm blind)
        stages the random sup-min under-resolves and drifts toward ``sum_h`` (measured
        103 -> 79 as ``n_samples`` grows, vs a true ``max_h`` of 55). Validate the blind
        composition with the exact per-stage ``max(char_time)`` instead.

    Args:
        stage_arms: Per-stage arm-mean vectors.
        sigma: Gaussian standard deviation.
        n_samples: Joint Dirichlet samples over the product polytope.
        seed: RNG seed.

    Returns:
        ``1 / sup_w min_h g_h(w_h)`` -- the episodic characteristic time.
    """
    rng = np.random.default_rng(seed)
    per_stage = np.stack(
        [
            _stage_rates_batch(
                arms, rng.dirichlet(np.ones(len(arms)), n_samples), sigma
            )
            for arms in stage_arms
        ]
    )  # (H, n_samples)
    sup_min = float(per_stage.min(axis=0).max())
    return 1.0 / sup_min
