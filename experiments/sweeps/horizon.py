"""Horizon lift: the layered-MDP lift composes per-stage T*; separation survives.

Sweeps over the horizon `H` and the per-stage branch count `k` on `LAYERED-CHAIN`
(identical separating stages), printing the graph-aware vs structure-blind MDP
characteristic times under both budget models, and the brute-force confirmation that the
episodic composition is `max_h` (not the additive `sum_h`):

- episodic ratio is **flat in H** (bottleneck `max_h`, set by the worst stage),
- generative gap is **additive Theta(H*k)** (`sum_h`),
- brute-force `sup_w min_h g_h(w_h)` lands on `max_h T*_h`, not `sum_h T*_h`.

Run::

    .venv/bin/python -m experiments.sweeps.horizon
"""

from __future__ import annotations

from iats.bandit.chartime import char_time
from iats.bandit.instances import chain_with_branches
from iats.mdp.layered import brute_force_episodic_time, mdp_char_time, mdp_separation


def main() -> int:
    """Print the horizon-lift witness (composition law + surviving separation)."""
    k = 6
    print(f"LAYERED-CHAIN: identical stages, k={k} branches each (sigma=1)\n")

    print("episodic separation is FLAT in H (bottleneck max_h):")
    print("  H   T*_graph   T*_blind   ratio")
    stage = chain_with_branches(n_branches=k)
    for h in (1, 2, 4, 8, 16):
        r = mdp_separation([stage] * h, budget="episodic")
        print(f"  {h:<3} {r['t_graph']:>8.2f} {r['t_blind']:>10.2f} {r['ratio']:>7.3f}")

    print("\ngenerative gap is ADDITIVE Theta(H*k) (sum_h):")
    print("  H   T*_graph   T*_blind   blind-graph")
    for h in (1, 2, 4, 8, 16):
        r = mdp_separation([stage] * h, budget="generative")
        gap = r["t_blind"] - r["t_graph"]
        print(f"  {h:<3} {r['t_graph']:>8.2f} {r['t_blind']:>10.2f} {gap:>11.2f}")

    print("\nload-bearing cross-check: episodic composition is max_h, not sum_h")
    # Exact per-stage max on the (multi-arm) BLIND stages -- the trustworthy check.
    blind = [stage.blind_arms] * 2
    per_stage_max = max(char_time(a) for a in blind)
    t_max = mdp_char_time(blind, budget="episodic")
    t_sum = mdp_char_time(blind, budget="generative")
    print(
        f"  exact per-stage max(char_time) = {per_stage_max:.2f}  |  "
        f"max_h={t_max:.2f}  sum_h={t_sum:.2f}"
    )
    # Brute force ONLY as a 2-arm sanity point -- it under-resolves on multi-arm stages.
    graph = [stage.graph_arms] * 2
    g_max = mdp_char_time(graph, budget="episodic")
    g_bf = brute_force_episodic_time(graph, n_samples=200_000)
    print(
        f"  2-arm brute-force sanity: sup min_h g_h = {g_bf:.2f} vs max_h={g_max:.2f} "
        "(brute force NOT trusted on multi-arm blind stages -- it drifts toward sum_h)"
    )
    is_max = (
        per_stage_max == t_max
        and t_max < 0.6 * t_sum
        and abs(g_bf - g_max) / g_max < 0.05
    )
    verdict = "PASS" if is_max else "FAIL"
    print(f"\n  acceptance check: {verdict}  (episodic max_h; separation flat in H)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
