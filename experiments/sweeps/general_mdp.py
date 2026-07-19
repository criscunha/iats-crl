"""E5 deep-dive: the separation beyond the chain, via the exact flow-polytope solver.

Illustrative (non-gated, non-pre-registered) deep-dive using :mod:`iats.mdp.general`
(graph-aware and structure-blind characteristic times over the full occupancy/flow
polytope by Frank-Wolfe -- deterministic, monotone, no rectangularity assumed). It
witnesses Proposition ``prop:aligned`` (the greedy-aligned sub-case), *not*
Conjecture ``channelI``:

- **Non-chain, rectangular (within R):** a branching MDP whose stage-2 states are
  reached by a policy-independent chance split. The separation persists and *grows
  monotonically* with the per-state branch count ``k`` (exact ``T*``).
- **Non-rectangular (Channel I), H=2, greedy-aligned:** an easy root decision gates
  which stage-2 state is visited (L -> S_L, R -> S_R). The separation persists and
  tracks the single-stage ratio; here the gating decision is cheap, so navigation does
  *not* inflate ``T*`` -- branches are never free (ratio > 1) but not extra-costly. The
  navigating Track-and-Stop is delta-correct and attains ``T*`` at leading order.
- **Negative instance (near-value-tie, misaligned):** a gated MDP where the local
  reward argmax is *not* the value-optimal action and the value gap is tiny. The
  single-flip solver understates the true value-coupled ``T*`` by a large factor
  (~120x) -- the sharp reason Channel I is left open (Conjecture ``channelI``).
- **Concavity witness:** ``0`` Jensen violations over 30000 random occupancy pairs, on
  a non-rectangular instance -- convexity is not the obstruction.

Writes ``results/paper/e5_witnesses.json`` (committed artifact behind the E5 claims).

Run::

    .venv/bin/python -m experiments.sweeps.general_mdp
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from iats.analysis.stats import ratio_bootstrap_ci
from iats.mdp.general import (
    MDPState,
    SmallMDP,
    concavity_violations,
    is_greedy_aligned,
    mdp_track_and_stop,
    optimal_design,
    separation_polytope,
    single_flip_understatement,
)

RESULTS = Path("results/paper")
FW_ITERS = 2000


def rectangular(k: int) -> SmallMDP:
    """Non-chain rectangular MDP: root chance-splits 50/50 to two branchy leaves."""
    root = MDPState([0.0], [False], [[(1, 0.5), (2, 0.5)]])
    leaf = MDPState(
        [1.0, 0.5] + [0.2] * k, [False, False] + [True] * k, [[] for _ in range(2 + k)]
    )
    return SmallMDP([root, leaf, leaf])


def nonrectangular(k: int) -> SmallMDP:
    """Non-rectangular H=2 MDP: an easy root decision gates which leaf is visited."""
    root = MDPState([1.0, 0.0], [False, False], [[(1, 1.0)], [(2, 1.0)]])
    leaf = MDPState(
        [1.0, 0.5] + [0.2] * k, [False, False] + [True] * k, [[] for _ in range(2 + k)]
    )
    return SmallMDP([root, leaf, leaf])


def near_value_tie() -> SmallMDP:
    """Misaligned gated MDP: local argmax != value argmax, value gap ~0.0645 (tiny).

    Root reward argmax is action 0 (1.0 > 0.5), but Q(root,0)=1.0+0.0=1.0 while
    Q(root,1)=0.5+0.5645=1.0645, so the value-optimal action is 1 -- a near tie. The
    single-flip solver, ranking by local reward, understates the value-coupled ``T*``.
    """
    root = MDPState([1.0, 0.5], [False, False], [[(1, 1.0)], [(2, 1.0)]])
    leaf_l = MDPState([0.0], [False], [[]])
    leaf_r = MDPState([0.5645], [False], [[]])
    return SmallMDP([root, leaf_l, leaf_r])


def _sweep(name: str, build: Callable[[int], SmallMDP]) -> dict[str, list[float]]:
    """Print and collect the monotone separation sweep for one instance family."""
    aligned = all(is_greedy_aligned(build(k)) for k in (0, 2, 4, 6))
    print(f"\n=== {name}: separation vs branch count k (greedy-aligned={aligned}) ===")
    print(f"  {'k':>2} {'T*_graph':>9} {'T*_blind':>9} {'ratio':>6}")
    ks: list[float] = []
    ratios: list[float] = []
    for k in (0, 2, 4, 6):
        r = separation_polytope(build(k), n_iters=FW_ITERS)
        print(f"  {k:>2} {r['t_graph']:>9.1f} {r['t_blind']:>9.1f} {r['ratio']:>6.2f}")
        ks.append(k)
        ratios.append(r["ratio"])
    return {"ks": ks, "ratio": ratios}


def _achievability(seeds: int = 60) -> dict[str, object]:
    """Non-rectangular H=2: IA-Track-and-Stop attains T*, with a bootstrap CI ratio."""
    print("\n=== non-rectangular H=2: IA-Track-and-Stop attains T* (matching) ===")
    mdp = nonrectangular(6)
    delta = 0.01
    log_inv = np.log(1.0 / delta)
    best = [int(np.argmax(s.means)) for s in mdp.states]
    taus: dict[str, np.ndarray] = {}
    out: dict[str, object] = {}
    for label, ga in (("graph", True), ("blind", False)):
        tstar, pis = optimal_design(mdp, graph_aware=ga, n_iters=FW_ITERS)
        arr, errs = np.empty(seeds), 0
        for s in range(seeds):
            chosen, n = mdp_track_and_stop(
                mdp, pis, delta=delta, rng=np.random.default_rng(s), graph_aware=ga
            )
            arr[s] = n
            errs += int(any(chosen[i] != best[i] for i in range(len(mdp.states))))
        taus[label] = arr
        rate = float(arr.mean()) / log_inv
        print(
            f"  {label:5}: T*={tstar:6.1f}  rate={rate:6.1f}  "
            f"rate/T*={rate / tstar:.2f}  errors={errs}/{seeds}"
        )
        out[f"{label}_tstar"] = tstar
        out[f"{label}_rate_over_tstar"] = rate / tstar
        out[f"{label}_errors"] = errs
    ratio, lo, hi = ratio_bootstrap_ci(taus["blind"], taus["graph"], n_boot=2000)
    print(f"  realized speed-up (bootstrap 95% CI): {ratio:.2f} [{lo:.2f}, {hi:.2f}]")
    out |= {"realized_ratio": ratio, "ci_lo": lo, "ci_hi": hi, "seeds": seeds}
    return out


def main() -> int:
    """Run the E5 deep-dive, print the witnesses, and write the committed artifact."""
    rect = _sweep("non-chain rectangular (within R)", rectangular)
    nonrect = _sweep("non-rectangular H=2 (Channel I, greedy-aligned)", nonrectangular)
    achieve = _achievability()

    print("\n=== negative instance: near-value-tie, misaligned (single-flip fails) ===")
    tie = near_value_tie()
    us = single_flip_understatement(tie)
    print(
        f"  greedy-aligned={is_greedy_aligned(tie)}  "
        f"T*_single-flip={us['t_single_flip']:.1f}  "
        f"T*_value-coupled={us['t_value_coupled']:.1f}  "
        f"understatement={us['understatement']:.0f}x"
    )

    print("\n=== concavity witness: sup_w min_s g_s is a concave maximisation ===")
    viol = concavity_violations(
        nonrectangular(6), graph_aware=False, n_trials=30_000, seed=0
    )
    print(f"  {viol}/30000 concavity violations on the non-rectangular instance")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "e5_witnesses.json").write_text(
        json.dumps(
            {
                "fw_iters": FW_ITERS,
                "rectangular": rect,
                "nonrectangular": nonrect,
                "achievability": achieve,
                "understatement": us,
                "concavity_violations": viol,
                "concavity_trials": 30_000,
            },
            indent=2,
        )
    )
    print(f"\nwitnesses -> {RESULTS}/e5_witnesses.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
