"""Exact characteristic time for a small finite-horizon causal MDP (Frank-Wolfe).

The layered composition (:mod:`iats.mdp.layered`) assumes condition (R): rectangular
visitation, so the occupancy set is a product of simplices and ``T*`` has a closed
form. This module drops that. It represents a small finite-horizon MDP as a DAG of
states (action means, a branch label per action, a per-action successor
distribution) and evaluates the Garivier-Kaufmann / Al-Marjani-Proutiere program

    T*^{-1} = sup_{w in Omega(M)}  min_{s : |A_s| >= 2}  g_s(w(s, .)),

directly over the state-action **occupancy (flow) polytope** ``Omega(M)``, where
``w(s, a)`` is the expected number of visits to ``(s, a)`` per episode and ``g_s`` the
single-state best-arm inner objective. The graph-aware learner drops branch rivals
(do(branch) = baseline); the blind learner keeps them.

``Omega(M)`` is a convex polytope (linear flow constraints) and each ``g_s`` is concave
(the perspective of ``x/(1+x)``), so ``sup_w min_s g_s`` is a **concave maximisation**.
We solve it exactly (to numerical tolerance) with **Frank-Wolfe**: the linear
maximisation oracle over ``Omega(M)`` is a single backward-induction pass (the vertices
of the occupancy polytope are deterministic policies), and an exact 1-D line search on
the concave objective picks the step. This is deterministic, monotone in the branch
count, and returns absolute characteristic times (no random-simplex sampler, no
seed/under-resolution caveat).

**Scope (important).** This ranks each state's actions by their *local* reward mean,
so it computes per-state best-ARM identification -- equal to best-POLICY identification
only in the **greedy-value-aligned** regime, where the local argmax is the value-optimal
action at every gating state (:func:`is_greedy_aligned`). Under gating without alignment
the true BPI alternative set has *value-coupled* cross-state rivals this single-flip
``min`` does not enumerate, so the single-flip program *understates* the true ``T*``
(:func:`single_flip_understatement`) and the general Channel-I upper bound is open. Demo
instances are aligned by construction; callers should assert :func:`is_greedy_aligned`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MDPState:
    """One state of a small finite-horizon causal MDP.

    Attributes:
        means: Per-action reward means (Gaussian, shared ``sigma``).
        is_branch: Per-action flag; ``True`` marks a reward-non-ancestor branch
            intervention (point-identified, so the graph-aware learner drops it).
        trans: Per-action successor distribution, a list of ``(next_state, prob)``
            pairs; empty is terminal. Deterministic gating ``[(next, 1.0)]`` is
            non-rectangular; a split ``[(L, 0.5), (R, 0.5)]`` on every action keeps
            visitation rectangular.
    """

    means: list[float]
    is_branch: list[bool]
    trans: list[list[tuple[int, float]]]


@dataclass(frozen=True)
class SmallMDP:
    """A small finite-horizon causal MDP as a DAG of :class:`MDPState`.

    States must be listed in topological order (root first, each successor after its
    predecessors) so occupancy is one forward pass.

    Attributes:
        states: The states, topologically ordered.
        root: Index of the initial state (default 0).
    """

    states: list[MDPState]
    root: int = 0


def _q_values(mdp: SmallMDP, s: int, value: np.ndarray) -> np.ndarray:
    """Per-action Q-values ``mu(s,a) + E V(s')`` at ``s`` given successor values."""
    st = mdp.states[s]
    return np.array(
        [
            st.means[a] + sum(p * value[ns] for ns, p in st.trans[a])
            for a in range(len(st.means))
        ]
    )


def is_greedy_aligned(mdp: SmallMDP) -> bool:
    """Is the local reward argmax the value-optimal action at every gating state?

    Backward induction gives the optimal value ``V(s) = max_a [mu(s,a) + E V(s')]``.
    Alignment holds when, at each decision state, ``argmax_a mu(s,a)`` (what this solver
    ranks by) equals ``argmax_a Q(s,a)`` (the value-optimal action). Only then does
    per-state best-arm identification coincide with best-policy identification; outside
    it, the solver and :func:`mdp_track_and_stop` compute the wrong object (Channel I).

    Args:
        mdp: The MDP (states topologically ordered).

    Returns:
        ``True`` iff every gating state is greedy-value-aligned.
    """
    n = len(mdp.states)
    value = np.zeros(n)
    for s in reversed(range(n)):  # successors before predecessors
        value[s] = float(_q_values(mdp, s, value).max()) if mdp.states[s].means else 0.0
    for s, st in enumerate(mdp.states):
        if len(st.means) >= 2 and int(np.argmax(st.means)) != int(
            np.argmax(_q_values(mdp, s, value))
        ):
            return False
    return True


def _occupancy_of(mdp: SmallMDP, policy: list[np.ndarray]) -> list[np.ndarray]:
    """State-action occupancy ``w(s,a) = rho(s) pi(a|s)`` of a per-state policy.

    One topological forward pass over the DAG (states are in topological order).

    Args:
        mdp: The MDP.
        policy: Per-state action-probability vectors.

    Returns:
        Per-state occupancy vectors ``w[s]``.
    """
    n = len(mdp.states)
    rho = np.zeros(n)
    rho[mdp.root] = 1.0
    w = [np.zeros(len(st.means)) for st in mdp.states]
    for s in range(n):
        st = mdp.states[s]
        if not st.means:
            continue
        w[s] = rho[s] * policy[s]
        for a, outs in enumerate(st.trans):
            for ns, p in outs:
                rho[ns] += w[s][a] * p
    return w


def _uniform_occupancy(mdp: SmallMDP) -> list[np.ndarray]:
    """Occupancy of the uniform-random policy -- a strictly interior FW start point.

    Args:
        mdp: The MDP (states in topological order).

    Returns:
        Per-state occupancy vectors ``w[s]`` (``w[s][a] = rho(s) / |A_s|``).
    """
    policy = [
        np.full(len(st.means), 1.0 / len(st.means)) if st.means else np.zeros(0)
        for st in mdp.states
    ]
    return _occupancy_of(mdp, policy)


def _lmo(mdp: SmallMDP, grad: list[np.ndarray]) -> list[np.ndarray]:
    """Linear-maximisation oracle over ``Omega(M)``: best deterministic-policy vertex.

    ``max_{w in Omega} <grad, w>`` is policy optimisation with per-action "reward"
    ``grad(s,a)``, solved exactly by one backward-induction pass (the vertices of the
    occupancy polytope are deterministic policies). Returns that vertex's occupancy.

    Args:
        mdp: The MDP.
        grad: Per-state supergradient vectors of the objective.

    Returns:
        The occupancy ``w[s]`` of the greedy deterministic policy for ``grad``.
    """
    n = len(mdp.states)
    value = np.zeros(n)
    best_a = np.zeros(n, dtype=int)
    for s in reversed(range(n)):
        st = mdp.states[s]
        if not st.means:
            continue
        q = np.array(
            [
                grad[s][a] + sum(p * value[ns] for ns, p in st.trans[a])
                for a in range(len(st.means))
            ]
        )
        best_a[s] = int(np.argmax(q))
        value[s] = float(q[best_a[s]])
    rho = np.zeros(n)
    rho[mdp.root] = 1.0
    w = [np.zeros(len(st.means)) for st in mdp.states]
    for s in range(n):
        st = mdp.states[s]
        if not st.means:
            continue
        w[s][best_a[s]] = rho[s]
        for ns, p in st.trans[best_a[s]]:
            rho[ns] += rho[s] * p
    return w


def _rivals(mdp: SmallMDP, graph_aware: bool) -> list[tuple[int, int, int, float]]:
    """Enumerate ``(state, best, rival, coef)`` binding terms of the inner ``min``.

    ``coef = (mu_best - mu_rival)^2 / (2 sigma^2 * sigma^2)`` is folded later; here we
    return the squared gap so callers apply ``sigma``.

    Args:
        mdp: The MDP.
        graph_aware: If ``True``, branch actions leave each state's rival set.

    Returns:
        A list of ``(s, best, a, gap_squared)`` for every rival that must be ruled out.
    """
    terms: list[tuple[int, int, int, float]] = []
    for s, st in enumerate(mdp.states):
        mu = np.asarray(st.means, dtype=float)
        if mu.size < 2:
            continue
        best = int(np.argmax(mu))
        for a in range(mu.size):
            if a == best or (graph_aware and st.is_branch[a]):
                continue
            terms.append((s, best, a, float((mu[best] - mu[a]) ** 2)))
    return terms


def _all_gs(
    mdp: SmallMDP, w: list[np.ndarray], sigma: float, graph_aware: bool
) -> np.ndarray:
    """Per-rival GK rates ``g_i(w)`` (0 if a binding rival gets zero visitation)."""
    out = []
    for s, best, a, gap2 in _rivals(mdp, graph_aware):
        if w[s][best] <= 0.0 or w[s][a] <= 0.0:
            out.append(0.0)
        else:
            denom = 2 * sigma**2 * (1.0 / w[s][best] + 1.0 / w[s][a])
            out.append(gap2 / denom)
    return np.array(out) if out else np.array([np.inf])


def _rate(mdp: SmallMDP, w: list[np.ndarray], sigma: float, graph_aware: bool) -> float:
    """Objective ``min_s g_s(w)``: cheapest single-state best-action flip under ``w``.

    Args:
        mdp: The MDP.
        w: Per-state state-action occupancy.
        sigma: Gaussian standard deviation.
        graph_aware: If ``True``, branch actions leave each state's rival set.

    Returns:
        ``min_s g_s`` (``0.0`` if some binding rival gets zero visitation).
    """
    return float(_all_gs(mdp, w, sigma, graph_aware).min())


def _smooth_supergradient(
    mdp: SmallMDP, w: list[np.ndarray], sigma: float, graph_aware: bool, alpha: float
) -> list[np.ndarray]:
    """Softmax-weighted supergradient of ``min_s g_s`` at ``w`` (temperature ``alpha``).

    Plain Frank-Wolfe stalls on the nonsmooth ``min`` of concave ``g``: a
    linear-oracle vertex zeros out a rival, so the single-active-term supergradient is
    not an ascent direction for the ``min``. We instead use the gradient of the smooth
    lower bound ``-eta log sum_i exp(-g_i / eta)`` with ``eta = alpha (max_i g_i -
    min_i g_i)`` -- a softmax over the near-active constraints, so the direction lifts
    *all* binding ``g_i`` together. As ``alpha`` is annealed to ``0`` this recovers the
    exact ``min``; the true ``min`` is used for the line search and the returned value.

    Args:
        mdp: The MDP.
        w: Per-state state-action occupancy (strictly interior).
        sigma: Gaussian standard deviation.
        graph_aware: If ``True``, branch actions leave each state's rival set.
        alpha: Relative softmax temperature (annealed toward 0).

    Returns:
        Per-state supergradient vectors of the smoothed objective.
    """
    grad = [np.zeros(len(st.means)) for st in mdp.states]
    terms = _rivals(mdp, graph_aware)
    if not terms:
        return grad
    gs = _all_gs(mdp, w, sigma, graph_aware)
    eta = _eta(gs, alpha)
    weights = np.exp(-(gs - gs.min()) / eta)
    weights /= weights.sum()
    for (s, best, a, gap2), wt in zip(terms, weights, strict=True):
        if w[s][best] <= 0.0 or w[s][a] <= 0.0:
            continue
        coef = gap2 / (2 * sigma**2)
        d = 1.0 / w[s][best] + 1.0 / w[s][a]
        grad[s][best] += wt * coef / d**2 / w[s][best] ** 2
        grad[s][a] += wt * coef / d**2 / w[s][a] ** 2
    return grad


def _eta(gs: np.ndarray, alpha: float) -> float:
    """Softmax temperature for the smoothed ``min``: ``alpha * (max g - min g)``."""
    return max(alpha * float(gs.max() - gs.min()), 1e-12)


def _smooth_value(
    mdp: SmallMDP, w: list[np.ndarray], sigma: float, graph_aware: bool, eta: float
) -> float:
    """Smooth lower bound ``-eta log mean exp(-g_i/eta)`` on ``min_i g_i`` (concave)."""
    gs = _all_gs(mdp, w, sigma, graph_aware)
    m = float(gs.min())
    return m - eta * float(np.log(np.mean(np.exp(-(gs - m) / eta))))


def _line_search(
    mdp: SmallMDP,
    w: list[np.ndarray],
    v: list[np.ndarray],
    sigma: float,
    graph_aware: bool,
    eta: float,
    iters: int = 40,
) -> float:
    """Golden-section maximiser of ``gamma -> f_eta((1-gamma)w + gamma v)``.

    Searches on the *smoothed* value ``f_eta`` (smooth and concave along the segment),
    so a step improves even when the raw ``min`` momentarily dips as one branch of a
    gating root starves -- which is exactly what a plain min line search cannot see.

    Args:
        mdp: The MDP.
        w: Current occupancy.
        v: LMO vertex occupancy.
        sigma: Gaussian standard deviation.
        graph_aware: Rival set flag.
        eta: Softmax temperature (held fixed along the segment).
        iters: Golden-section iterations.

    Returns:
        The step size ``gamma in [0, 1]``.
    """
    golden = (5**0.5 - 1) / 2
    lo, hi = 0.0, 1.0
    x1, x2 = hi - golden * (hi - lo), lo + golden * (hi - lo)
    f1 = _seg_smooth(mdp, w, v, x1, sigma, graph_aware, eta)
    f2 = _seg_smooth(mdp, w, v, x2, sigma, graph_aware, eta)
    for _ in range(iters):
        if f1 > f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - golden * (hi - lo)
            f1 = _seg_smooth(mdp, w, v, x1, sigma, graph_aware, eta)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + golden * (hi - lo)
            f2 = _seg_smooth(mdp, w, v, x2, sigma, graph_aware, eta)
    return x1 if f1 > f2 else x2


def _seg_smooth(
    mdp: SmallMDP,
    w: list[np.ndarray],
    v: list[np.ndarray],
    gamma: float,
    sigma: float,
    graph_aware: bool,
    eta: float,
) -> float:
    """Smoothed value at the segment point ``(1-gamma) w + gamma v``."""
    mix = [(1 - gamma) * w[s] + gamma * v[s] for s in range(len(w))]
    return _smooth_value(mdp, mix, sigma, graph_aware, eta)


def optimal_design(
    mdp: SmallMDP,
    sigma: float = 1.0,
    graph_aware: bool = True,
    n_iters: int = 1000,
) -> tuple[float, list[np.ndarray]]:
    """Solve ``sup_{w in Omega} min_s g_s(w)`` by Frank-Wolfe; return ``(T*, policy)``.

    Args:
        mdp: The small finite-horizon causal MDP.
        sigma: Gaussian standard deviation.
        graph_aware: Graph-aware learner (drops branch rivals) if ``True``, else blind.
        n_iters: Frank-Wolfe iterations (each is one backward-induction LMO + a 1-D
            line search); the concave objective converges globally.

    Returns:
        ``(t_star, pis)`` -- the characteristic time and the per-state exploration
        policy attaining it (``pis[s] = w(s, .) / rho(s)`` for the achievability
        witness). ``T* = inf`` if no design certifies a positive rate.
    """
    w = _uniform_occupancy(mdp)
    best_w, best_rate = w, _rate(mdp, w, sigma, graph_aware)
    for t in range(n_iters):
        alpha = 0.3 * (1e-4 / 0.3) ** (t / max(n_iters - 1, 1))  # anneal 0.3 -> 1e-4
        eta = _eta(_all_gs(mdp, w, sigma, graph_aware), alpha)
        grad = _smooth_supergradient(mdp, w, sigma, graph_aware, alpha)
        v = _lmo(mdp, grad)
        gamma = _line_search(mdp, w, v, sigma, graph_aware, eta)
        w = [(1 - gamma) * w[s] + gamma * v[s] for s in range(len(w))]
        r = _rate(mdp, w, sigma, graph_aware)
        if r > best_rate:
            best_w, best_rate = w, r
    w, rate = best_w, best_rate
    pis = [
        (w[s] / w[s].sum()) if w[s].size and w[s].sum() > 0 else w[s]
        for s in range(len(w))
    ]
    return (1.0 / rate if rate > 0 else float("inf")), pis


def char_time_polytope(
    mdp: SmallMDP,
    sigma: float = 1.0,
    graph_aware: bool = True,
    n_iters: int = 1000,
) -> float:
    """Characteristic time ``1 / sup_w min_s g_s`` over the occupancy polytope.

    Args:
        mdp: The small finite-horizon causal MDP.
        sigma: Gaussian standard deviation.
        graph_aware: Graph-aware learner (drops branch rivals) if ``True``, else blind.
        n_iters: Frank-Wolfe iterations.

    Returns:
        ``T*`` (``inf`` if no design certifies a positive rate).
    """
    return optimal_design(mdp, sigma, graph_aware, n_iters)[0]


def separation_polytope(
    mdp: SmallMDP, sigma: float = 1.0, n_iters: int = 1000
) -> dict[str, float]:
    """Graph-aware vs structure-blind ``T*`` (and their ratio) for a small causal MDP.

    Args:
        mdp: The MDP.
        sigma: Gaussian standard deviation.
        n_iters: Frank-Wolfe iterations per learner.

    Returns:
        Mapping with ``t_graph``, ``t_blind`` and ``ratio``.
    """
    t_graph = char_time_polytope(mdp, sigma, True, n_iters)
    t_blind = char_time_polytope(mdp, sigma, False, n_iters)
    return {"t_graph": t_graph, "t_blind": t_blind, "ratio": t_blind / t_graph}


def concavity_violations(
    mdp: SmallMDP,
    sigma: float = 1.0,
    graph_aware: bool = True,
    n_trials: int = 30_000,
    seed: int = 0,
) -> int:
    """Count Jensen violations of ``f(w) = min_s g_s(w)`` over random occupancy pairs.

    Samples pairs of designs (random per-state policies -> occupancy) and a random
    convex weight ``t``, and checks the concavity inequality
    ``f(t w1 + (1-t) w2) >= t f(w1) + (1-t) f(w2)`` (up to a small tolerance). The
    committed witness behind "convexity is not the obstruction" (paper Remark): the
    program stays a concave maximisation with or without rectangular visitation.

    Args:
        mdp: The MDP.
        sigma: Gaussian standard deviation.
        graph_aware: Rival set as seen by the graph-aware (``True``) / blind learner.
        n_trials: Number of random pairs tested.
        seed: RNG seed.

    Returns:
        The number of trials violating concavity (expected: ``0``).
    """
    rng = np.random.default_rng(seed)
    sizes = [len(st.means) for st in mdp.states]
    violations = 0
    for _ in range(n_trials):
        p1 = [rng.dirichlet(np.ones(k)) if k else np.zeros(0) for k in sizes]
        p2 = [rng.dirichlet(np.ones(k)) if k else np.zeros(0) for k in sizes]
        w1, w2 = _occupancy_of(mdp, p1), _occupancy_of(mdp, p2)
        t = float(rng.uniform())
        f1 = _rate(mdp, w1, sigma, graph_aware)
        f2 = _rate(mdp, w2, sigma, graph_aware)
        fmix = _rate(
            mdp,
            [t * w1[s] + (1 - t) * w2[s] for s in range(len(w1))],
            sigma,
            graph_aware,
        )
        if fmix < t * f1 + (1 - t) * f2 - 1e-9:
            violations += 1
    return violations


def value_gap_char_time(mdp: SmallMDP, sigma: float = 1.0) -> float:
    """Value-coupled BPI characteristic time for a 2-action-root gated ``H=2`` MDP.

    On a gated instance the correct BPI object must rule out the *value*-flip of the
    root's optimal action -- gap ``|Q(root, a*) - Q(root, a')|`` (backward-induction
    Q-values), not the local reward gap. This is the two-arm value game at the root, so
    its characteristic time is ``8 sigma_eff^2 / Delta_Q^2`` with per-arm value variance
    ``sigma_eff^2 = (1 + reach) sigma^2`` accumulating the root pull and the one gated
    successor pull (``reach = 1`` for a single deterministic successor).

    This is the *correct*-object baseline for :func:`single_flip_understatement`; it is
    exact only for the 2-action-root, single-successor gated family used as the
    near-value-tie witness (near-tie => large value time; the single-flip solver, blind
    to the value coupling, understates it).

    Args:
        mdp: A gated MDP whose root has two actions, each a single successor.
        sigma: Gaussian standard deviation.

    Returns:
        The value-coupled characteristic time at the root (``inf`` on an exact tie).
    """
    n = len(mdp.states)
    value = np.zeros(n)
    for s in reversed(range(n)):
        value[s] = float(_q_values(mdp, s, value).max()) if mdp.states[s].means else 0.0
    q = _q_values(mdp, mdp.root, value)
    order = np.argsort(q)[::-1]
    delta_q = float(q[order[0]] - q[order[1]])
    reach = max(len(mdp.states[mdp.root].trans[a]) for a in range(len(q)))
    sigma_eff2 = (1 + reach) * sigma**2
    return 8.0 * sigma_eff2 / delta_q**2 if delta_q > 0 else float("inf")


def single_flip_understatement(mdp: SmallMDP, sigma: float = 1.0) -> dict[str, float]:
    """Factor by which the single-flip solver understates ``T*`` on a gated instance.

    Compares the per-state single-flip characteristic time (:func:`char_time_polytope`,
    which ranks by local reward and so ignores value coupling) against the value-coupled
    root time (:func:`value_gap_char_time`). On a near-value-tie gated MDP the local
    reward gaps are large (small single-flip ``T*``) while the value gap is tiny (large
    correct ``T*``), so the single-flip program understates the truth by a large factor.

    Args:
        mdp: The gated near-value-tie MDP.
        sigma: Gaussian standard deviation.

    Returns:
        Mapping with ``t_single_flip``, ``t_value_coupled`` and ``understatement``
        (their ratio).
    """
    t_sf = char_time_polytope(mdp, sigma, graph_aware=True)
    t_vc = value_gap_char_time(mdp, sigma)
    return {
        "t_single_flip": t_sf,
        "t_value_coupled": t_vc,
        "understatement": t_vc / t_sf,
    }


def mdp_track_and_stop(
    mdp: SmallMDP,
    pis: list[np.ndarray],
    *,
    sigma: float = 1.0,
    delta: float = 0.01,
    rng: np.random.Generator,
    graph_aware: bool = True,
    max_episodes: int = 200_000,
) -> tuple[list[int], int]:
    """Navigating Track-and-Stop on a small causal MDP; witnesses the upper bound.

    Each episode follows the fixed near-optimal policy ``pis`` from root to terminal,
    observing a Gaussian reward at each visited state-action; it stops when *every*
    decision state's GLR (rivals = non-branch actions if ``graph_aware``) crosses
    ``beta(N, delta) + log(#decision states)``. Static optimal allocation + GLR attains
    ``T*.log(1/delta)(1+o(1))``, so on a non-rectangular instance this witnesses the
    Channel-I lower bound matched at leading order (finite-horizon achievability).

    Args:
        mdp: The MDP.
        pis: Per-state exploration policy (e.g. from :func:`optimal_design`).
        sigma: Gaussian standard deviation.
        delta: Target error probability.
        rng: Random generator.
        graph_aware: Drop branch rivals from each state's GLR if ``True``.
        max_episodes: Safety cap.

    Returns:
        ``(per_state_chosen, n_episodes)``.
    """
    n_states = len(mdp.states)
    counts = [np.ones(len(st.means)) for st in mdp.states]
    sums = [
        rng.normal(np.asarray(st.means), sigma) for st in mdp.states
    ]  # 1 warmup each
    decision = [s for s in range(n_states) if len(mdp.states[s].means) >= 2]
    log_d = np.log(max(len(decision), 1))
    n = 1
    while n < max_episodes:
        s = mdp.root  # roll one episode along pis
        while True:
            st = mdp.states[s]
            a = int(rng.choice(len(st.means), p=pis[s]))
            sums[s][a] += rng.normal(st.means[a], sigma)
            counts[s][a] += 1
            if not st.trans[a]:
                break
            nxt = st.trans[a]
            s = int(rng.choice([ns for ns, _ in nxt], p=[p for _, p in nxt]))
        n += 1
        beta = np.log((1.0 + np.log(n)) / delta) + log_d
        z_min = np.inf
        for s in decision:
            st = mdp.states[s]
            mu_hat = sums[s] / counts[s]
            best = _empirical_best(mu_hat, st.is_branch, graph_aware)
            for a in range(len(st.means)):
                if a == best or (graph_aware and st.is_branch[a]):
                    continue
                gap = mu_hat[best] - mu_hat[a]
                z = (
                    gap
                    * gap
                    / (2 * sigma**2 * (1.0 / counts[s][best] + 1.0 / counts[s][a]))
                )
                z_min = min(z_min, z)
        if z_min >= beta:
            break
    chosen = [
        _empirical_best(sums[s] / counts[s], mdp.states[s].is_branch, graph_aware)
        for s in range(n_states)
    ]
    return chosen, n


def _empirical_best(
    mu_hat: np.ndarray, is_branch: list[bool], graph_aware: bool
) -> int:
    """Empirical best action; the graph-aware learner excludes branch candidates.

    A branch is point-identified at the observational baseline, so the graph-aware
    learner never treats it as a best-action candidate -- otherwise a lightly sampled
    branch with a lucky draw becomes the spurious argmax and stalls the GLR.

    Args:
        mu_hat: Empirical per-action means.
        is_branch: Per-action branch flags.
        graph_aware: Exclude branch actions from the argmax if ``True``.

    Returns:
        The index of the empirical best (non-branch, when ``graph_aware``) action.
    """
    if graph_aware:
        cand = [a for a in range(len(mu_hat)) if not is_branch[a]]
        return int(cand[int(np.argmax(mu_hat[cand]))])
    return int(np.argmax(mu_hat))
