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

:func:`episodic_track_and_stop` is the algorithm-level counterpart: every stage runs
single-stage D-tracking on a shared episode counter and is **frozen** at its own first
threshold crossing, with the per-stage confidence ``delta / H``. Stage ``h`` with
``K_h`` arms uses, at episode ``n``, either the analysed threshold
``log(H (K_h - 1) zeta(2) n^3 / delta)`` (delta-correct by the union bound of
:mod:`iats.bandit.trackandstop`, applied per stage and summed over the ``H`` stages) or
the calibrated one ``log((1 + log n)(K_h - 1) / delta) + log H`` (delta-correctness not
proven).
"""

from __future__ import annotations

import typing as tp

import numpy as np

from iats.bandit.chartime import char_time, optimal_allocation
from iats.bandit.instances import CausalBanditInstance
from iats.bandit.trackandstop import _glr, threshold_fn

MAX_EPISODES: tp.Final[int] = 200_000


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
    max_episodes: int = MAX_EPISODES,
    realloc_every: int = 1,
    threshold: str = "analysed",
) -> tuple[list[int], int, list[int]]:
    """Episodic layered-MDP Track-and-Stop with per-stage freezing (paper Algorithm 1).

    One episode pulls every stage once, so a single shared episode counter ``n`` meters
    all stages (the episodic budget). **Warm-up:** ``max_h K_h`` episodes in which stage
    ``h`` pulls arm ``e mod K_h`` in episode ``e`` -- every arm at least once; since the
    protocol pulls every stage once per episode, all stages necessarily receive the same
    number of warm-up episodes. **Then**, at each episode ``n``, every unfrozen stage
    ``h`` is checked: it is frozen at the first ``n`` with
    ``Z_h(n) >= beta_h(n, delta)`` and its recommendation is fixed to its empirical best
    arm at that episode. A frozen stage keeps pulling its fixed arm in every subsequent
    episode; an unfrozen stage runs single-stage D-tracking with forced exploration
    (:mod:`iats.bandit.trackandstop`). The run stops when every stage is frozen, so the
    episode count is gated by the *slowest* stage -- the algorithm-level counterpart of
    the bottleneck composition ``max_h T*_h``.

    ``beta_h(n, delta)`` is the single-stage threshold with ``K_h`` arms at the
    per-stage confidence ``delta / H``: analysed ``log(H (K_h-1) zeta(2) n^3 / delta)``
    or calibrated ``log((1 + log n)(K_h-1) / delta) + log H``.

    Args:
        stage_arms: Per-stage arm-mean vectors.
        sigma: Gaussian standard deviation.
        delta: Target error probability (split as ``delta / H`` across stages).
        rng: Random generator (drives the Gaussian feedback).
        max_episodes: Safety cap on the episode count. A capped run returns
            ``n_episodes >= max_episodes`` (the ``max_h K_h`` warm-up episodes happen
            regardless); stages still unfrozen at the cap report ``n_episodes`` as
            their stop episode and their empirical best arm.
        realloc_every: Episodes between recomputations of a stage's plug-in allocation
            (``1``, the default, is the analysed rule; larger values cache for speed).
        threshold: ``"analysed"`` (default) or ``"calibrated"``.

    Returns:
        ``(per_stage_chosen, n_episodes, per_stage_stop_episodes)`` -- the arm fixed at
        each stage, the episode count at stopping (``= max_h`` of the stop episodes,
        ``= per-stage pulls``), and the episode at which each stage was frozen.

    Raises:
        ValueError: If ``threshold`` is not one of
            :data:`iats.bandit.trackandstop.THRESHOLDS`.
    """
    beta_fn = threshold_fn(threshold)
    stages = [np.asarray(a, dtype=float) for a in stage_arms]
    sizes = [s.size for s in stages]
    h_count = len(stages)
    # per-stage confidence: beta_h(n, delta) = beta(n, delta/H)
    delta_h = delta / h_count
    warm = max(sizes)
    counts = [np.zeros(sz) for sz in sizes]
    sums = [np.zeros(sz) for sz in sizes]
    for e in range(warm):  # warm-up: every stage gets `warm` episodes, each arm >= once
        for h in range(h_count):
            i = e % sizes[h]
            sums[h][i] += rng.normal(stages[h][i], sigma)
            counts[h][i] += 1
    n = warm
    chosen = [-1] * h_count  # fixed at freeze time
    stop_at = [-1] * h_count  # freeze episode; -1 while the stage is still running
    caches: list[np.ndarray | None] = [None] * h_count
    while n < max_episodes:
        for h in range(h_count):
            if stop_at[h] < 0:
                z, best = _glr(sums[h] / counts[h], counts[h], sigma)
                if z >= beta_fn(n, delta_h, sizes[h]):
                    stop_at[h], chosen[h] = n, best
        if all(s >= 0 for s in stop_at):
            break
        for h in range(h_count):
            if stop_at[h] >= 0:  # frozen: keep pulling the fixed recommendation
                i = chosen[h]
            elif counts[h].min() < np.sqrt(n) - sizes[h] / 2.0:  # forced exploration
                i = int(np.argmin(counts[h]))
            else:  # D-tracking toward the plug-in optimal allocation
                w = caches[h]
                if w is None or n % realloc_every == 0:
                    w = optimal_allocation(sums[h] / counts[h], sigma)
                    caches[h] = w
                i = int(np.argmax(n * w - counts[h]))
            sums[h][i] += rng.normal(stages[h][i], sigma)
            counts[h][i] += 1
        n += 1
    for h in range(h_count):  # capped: unfrozen stages report the cap + current best
        if stop_at[h] < 0:
            stop_at[h], chosen[h] = n, int(np.argmax(sums[h] / counts[h]))
    return chosen, n, stop_at


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
