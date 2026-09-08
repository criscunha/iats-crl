"""Separation sweep: does T*_blind / T*_graph diverge with the branch count?

The graph-aware characteristic time is flat in the number of reward-non-ancestor
branches; the blind one grows ~linearly, so the ratio diverges -- the intrinsic,
leading-order separation the paper rests on. CPU-only, seconds to run.

Run::

    .venv/bin/python -m experiments.sweeps.separation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from iats.bandit.chartime import separation
from iats.bandit.instances import chain_with_branches


def run(branch_counts: list[int]) -> list[tuple[int, float, float, float]]:
    """Compute (k, T*_graph, T*_blind, ratio) for each branch count ``k``."""
    rows: list[tuple[int, float, float, float]] = []
    for k in branch_counts:
        inst = chain_with_branches(n_branches=k)
        s = separation(inst.relevant_means, inst.branch_mean, k)
        rows.append((k, s["t_graph"], s["t_blind"], s["ratio"]))
    return rows


def main() -> int:
    """Print the separation sweep, fit the linear slope, and optionally plot it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, default=None, help="optional PNG path")
    args = parser.parse_args()

    ks = [0, 1, 2, 4, 8, 16, 32, 64]
    rows = run(ks)
    print(f"{'k':>4} {'T*_graph':>10} {'T*_blind':>10} {'ratio':>8}")
    for k, tg, tb, r in rows:
        print(f"{k:>4} {tg:>10.2f} {tb:>10.2f} {r:>8.2f}")
    karr = np.array([r[0] for r in rows], dtype=float)
    ratios = np.array([r[3] for r in rows], dtype=float)
    slope = float(np.polyfit(karr, ratios, 1)[0])
    print(f"\nratio ~ linear in k, slope = {slope:.3f} per branch")
    verdict = "PASS (separation diverges)" if ratios[-1] > 3 * ratios[1] else "FAIL"
    print(f"ACCEPTANCE CHECK: {verdict}")

    if args.figure is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(karr, ratios, marker="o")
        ax.set_xlabel("number of reward-non-ancestor branches k")
        ax.set_ylabel(r"$T^*_{\mathrm{blind}} / T^*_{\mathrm{graph}}$")
        ax.set_title("Graph-aware vs structure-blind separation")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print(f"figure -> {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
