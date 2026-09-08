"""Achievability: does IA-Track-and-Stop attain T*_G and beat the blind learner?

Runs Track-and-Stop over many seeds for the intervention-aware learner (ancestor arms
only) and the structure-blind learner (ancestors + k branches), reporting:
- empirical error rate (compared with delta) and the number of capped runs,
- mean stopping time tau and tau/log(1/delta) vs the characteristic time T* (the rate),
- the realized speed-up tau_blind / tau_IA -- the separation, in practice.

Run::

    .venv/bin/python -m experiments.sweeps.achievability --seeds 300
"""

from __future__ import annotations

import argparse

import numpy as np

from iats.bandit.chartime import char_time
from iats.bandit.instances import chain_with_branches
from iats.bandit.trackandstop import MAX_STEPS, THRESHOLDS, track_and_stop


def run_trials(
    arms: list[float],
    sigma: float,
    delta: float,
    seeds: int,
    realloc_every: int = 1,
    threshold: str = "analysed",
) -> dict[str, object]:
    """Run Track-and-Stop over ``seeds`` seeds; return error rate and stopping stats.

    Args:
        arms: True arm means for this learner.
        sigma: Gaussian standard deviation.
        delta: Target error probability.
        seeds: Number of seeds (``default_rng(0..seeds-1)``).
        realloc_every: D-tracking allocation-cache period (1 = recompute every step,
            the analysed rule; larger values cache the allocation for speed).
        threshold: Stopping threshold, ``"analysed"`` or ``"calibrated"``
            (:mod:`iats.bandit.trackandstop`).

    Returns:
        Mapping with ``error_rate``, ``mean_tau``, ``tstar``, ``errors`` (count),
        ``capped`` (count of runs that hit the step cap ``MAX_STEPS``), and the per-seed
        arrays ``taus``, ``forced`` and ``wrong`` (bool, wrong recommendation).
    """
    best = int(np.argmax(arms))
    taus = np.empty(seeds)
    forced = np.empty(seeds)
    wrong = np.zeros(seeds, dtype=bool)
    for s in range(seeds):
        rng = np.random.default_rng(s)
        chosen, tau, fp = track_and_stop(
            arms,
            sigma=sigma,
            delta=delta,
            rng=rng,
            realloc_every=realloc_every,
            threshold=threshold,
        )
        taus[s] = tau
        forced[s] = fp
        wrong[s] = chosen != best
    return {
        "error_rate": float(wrong.mean()),
        "mean_tau": float(taus.mean()),
        "tstar": char_time(arms, sigma),
        "errors": int(wrong.sum()),
        "capped": int((taus >= MAX_STEPS).sum()),
        "taus": taus,
        "forced": forced,
        "wrong": wrong,
    }


def main() -> int:
    """Print the IA vs structure-blind achievability comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=300)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--threshold", choices=THRESHOLDS, default="analysed")
    args = parser.parse_args()

    inst = chain_with_branches(n_branches=args.branches)
    ia_arms = inst.graph_arms
    blind_arms = inst.blind_arms
    print(
        f"instance: ancestors={ia_arms}  +{args.branches} branches @ {inst.branch_mean}"
        f"  threshold={args.threshold}"
    )
    for delta in (0.1, 0.01):
        log_inv = np.log(1.0 / delta)
        ia = run_trials(
            ia_arms, args.sigma, delta, args.seeds, threshold=args.threshold
        )
        bl = run_trials(
            blind_arms, args.sigma, delta, args.seeds, threshold=args.threshold
        )
        print(f"\n--- delta = {delta} (log(1/delta) = {log_inv:.2f}) ---")
        for name, r in (("IA  ", ia), ("blind", bl)):
            err, mean_tau = float(r["error_rate"]), float(r["mean_tau"])
            tstar = float(r["tstar"])
            print(
                f"  {name}: err={err:.3f}  cap={r['capped']}  mean_tau={mean_tau:.0f}  "
                f"T*={tstar:.1f}  tau/log(1/d)={mean_tau / log_inv:.1f}  "
                f"(ratio to T*: {mean_tau / log_inv / tstar:.2f})"
            )
        speedup = float(bl["mean_tau"]) / float(ia["mean_tau"])
        print(f"  speed-up tau_blind/tau_IA = {speedup:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
