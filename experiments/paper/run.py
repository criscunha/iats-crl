"""Driver: run E1-E4, apply confirmatory stats, print the acceptance-check table.

One command reproduces the paper's experiments section:

- **E1** (theory-only): the separation ``T*_blind/T*_G`` diverges ~linearly in the
  branch count ``k`` while ``T*_G`` is flat -- intrinsic and leading-order.
- **E2** (300 seeds): IA-Track-and-Stop is delta-correct (Clopper-Pearson upper bound
  <= delta, certified where n allows, delta >= 0.01) and its rate ``tau/log(1/delta)``
  decreases toward ``T*`` as delta -> 0 (the ``1 + o(1)`` finite-delta transient).
- **E3** (300 seeds, headline): the realized speed-up ``tau_blind/tau_IA``
  (ratio-of-means, unpaired bootstrap CI) tracks theory, significant (Holm-Bonferroni).
- **E4** (theory + episodic sim): the episodic layered-MDP composition lands on
  ``max_h`` (not ``sum_h``), the separation is flat in H, the generative gap is linear.

Writes per-seed arrays to ``results/paper/arrays.npz`` and the scalar/gate summary to
``results/paper/summary.json`` (both gitignored); :mod:`experiments.paper.figures`
renders the figures from that cache without re-simulating.

Run::

    .venv/bin/python -m experiments.paper.run --seeds 300  # powered run (~30-45 min)
    .venv/bin/python -m experiments.paper.run --quick      # smoke -> results/scratch/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.sweeps.achievability import run_trials
from iats.analysis.stats import (
    clopper_pearson_upper,
    holm_bonferroni,
    mann_whitney_u,
    ratio_bootstrap_ci,
)
from iats.bandit.chartime import separation
from iats.mdp.layered import episodic_track_and_stop, mdp_char_time

SIGMA = 1.0
RELEVANT = [1.0, 0.5]
BRANCH_MEAN = 0.2
DELTA_0 = 1.0 - BRANCH_MEAN  # branch gap (0.8)
DELTA_HEAD = 0.01  # headline delta for E3/E4
REALLOC = 20  # D-tracking allocation-cache period (artifact-free, ~20x cheaper)
RESULTS = Path("results/paper")


def _blind_arms(k: int) -> list[float]:
    """Structure-blind arms: the relevant arms plus ``k`` non-ancestor branches."""
    return [*RELEVANT, *([BRANCH_MEAN] * k)]


def _sim(
    arms: list[float],
    delta: float,
    seeds: int,
    cache: dict[tuple[tuple[float, ...], float], dict[str, object]],
) -> dict[str, object]:
    """Run (or fetch from ``cache``) Track-and-Stop over seeds at ``REALLOC``."""
    key = (tuple(arms), delta)
    if key not in cache:
        cache[key] = run_trials(arms, SIGMA, delta, seeds, realloc_every=REALLOC)
    return cache[key]


def _gate(
    name: str, observed: float, threshold: str, passed: bool
) -> dict[str, object]:
    """One row of the pre-registered acceptance-check table."""
    return {"name": name, "observed": observed, "threshold": threshold, "pass": passed}


def _e1_separation() -> dict[str, object]:
    """E1: deterministic separation sweep -- flat ``T*_G``, diverging ratio, floor."""
    ks = [0, 1, 2, 4, 8, 16, 32, 64]
    tg, tb, ratio = [], [], []
    for k in ks:
        s = separation(RELEVANT, BRANCH_MEAN, k, SIGMA)
        tg.append(s["t_graph"])
        tb.append(s["t_blind"])
        ratio.append(s["ratio"])
    floor = 2 * SIGMA**2 / DELTA_0**2  # per-branch characteristic-time floor
    floor_ok = all(b >= floor * k - 1e-9 for k, b in zip(ks, tb, strict=True))
    slope = float(np.polyfit(np.array(ks, float), np.array(ratio, float), 1)[0])
    return {
        "ks": ks,
        "t_graph": tg,
        "t_blind": tb,
        "ratio": ratio,
        "floor": floor,
        "gates": [
            _gate("E1_tG_flat", float(np.ptp(tg)), "< 1e-9", float(np.ptp(tg)) < 1e-9),
            _gate(
                "E1_floor", 1.0 if floor_ok else 0.0, "T*_blind >= floor*k", floor_ok
            ),
            _gate("E1_diverges", ratio[-1] / ratio[1], "> 3", ratio[-1] > 3 * ratio[1]),
            _gate("E1_slope_pos", slope, "> 0", slope > 0),
        ],
    }


def _e2_correctness(seeds: int, deltas: list[float]) -> dict[str, object]:
    """E2: delta-correctness (Clopper-Pearson) and rate stabilisation at k=8."""
    cache: dict[tuple[tuple[float, ...], float], dict[str, object]] = {}
    ia_arms, bl_arms = RELEVANT, _blind_arms(8)
    rows: dict[str, dict[str, object]] = {}
    for delta in deltas:
        log_inv = np.log(1.0 / delta)
        for label, arms in (("ia", ia_arms), ("blind", bl_arms)):
            r = _sim(arms, delta, seeds, cache)
            errors, taus = int(r["errors"]), np.asarray(r["taus"])
            cp = clopper_pearson_upper(errors, seeds)
            rows[f"{label}_d{delta:g}"] = {
                "delta": delta,
                "errors": errors,
                "cp_upper": cp,
                "rate": float(taus.mean()) / log_inv,
                "rate_over_tstar": (float(taus.mean()) / log_inv) / float(r["tstar"]),
                "forced_frac": float(np.asarray(r["forced"]).mean() / taus.mean()),
                "taus": taus,
            }
    gates = []
    # delta-correctness. The exact 95% CP bound only certifies down to cp_floor =
    # cp(0, n); below that, n cannot prove err <= delta, so we assert the weaker
    # (and true) "0 errors observed" and flag the certification as seed-limited.
    cp_floor = clopper_pearson_upper(0, seeds)
    for key, row in rows.items():
        d = float(row["delta"])
        if d >= cp_floor:  # certifiable at this seed count
            gates.append(
                _gate(
                    f"E2_cp_{key}",
                    float(row["cp_upper"]),
                    f"cp <= {d:g}",
                    row["cp_upper"] <= d,
                )
            )
        else:  # deeper than n can certify: require 0 observed errors instead
            errs = int(row["errors"])
            gates.append(
                _gate(
                    f"E2_cp_{key}",
                    errs,
                    f"0 err (cp {cp_floor:.4f} > {d:g}, n-limited)",
                    errs == 0,
                )
            )
    # rate flatness for IA between the two deepest deltas (NOT convergence to 1).
    ds = sorted(deltas)
    if len(ds) >= 2:
        a = float(rows[f"ia_d{ds[0]:g}"]["rate_over_tstar"])
        b = float(rows[f"ia_d{ds[1]:g}"]["rate_over_tstar"])
        gates.append(_gate("E2_rate_flat", abs(a - b), "< 0.1", abs(a - b) < 0.1))
    # forced-exploration is o(tau): the speed-up is the T* gap, not an arm-count tax.
    ff = (
        float(rows[f"blind_d{DELTA_HEAD:g}"]["forced_frac"])
        if f"blind_d{DELTA_HEAD:g}" in rows
        else 1.0
    )
    gates.append(_gate("E2_forced_small", ff, "< 0.2", ff < 0.2))
    return {"rows": rows, "gates": gates}


def _e3_headline(seeds: int) -> dict[str, object]:
    """E3: realized speed-up vs theory, unpaired ratio CI + Holm Mann-Whitney."""
    cache: dict[tuple[tuple[float, ...], float], dict[str, object]] = {}
    ks = [0, 2, 4, 8, 16, 32]
    ia_taus = np.asarray(_sim(RELEVANT, DELTA_HEAD, seeds, cache)["taus"])
    per_k: dict[int, dict[str, object]] = {}
    pvals: dict[str, float] = {}
    for k in ks:
        bl_taus = np.asarray(_sim(_blind_arms(k), DELTA_HEAD, seeds, cache)["taus"])
        ratio, lo, hi = ratio_bootstrap_ci(bl_taus, ia_taus, n_boot=2000)
        theory = separation(RELEVANT, BRANCH_MEAN, k, SIGMA)["ratio"]
        per_k[k] = {
            "ratio_of_means": ratio,
            "ci_lo": lo,
            "ci_hi": hi,
            "median": float(np.median(bl_taus) / np.median(ia_taus)),
            "theory": theory,
            "bl_taus": bl_taus,
        }
        # Mann-Whitney U is rank-based, so invariant to the monotone log transform --
        # we pass the raw stopping times directly.
        if k in (2, 8, 16, 32):
            pvals[f"H{[2, 8, 16, 32].index(k) + 1}_speedup_k{k}"] = mann_whitney_u(
                bl_taus, ia_taus, "greater"
            )
    # H5: realized growth in k (blind tau at k=32 stochastically exceeds k=2).
    pvals["H5_speedup_growth"] = mann_whitney_u(
        np.asarray(per_k[32]["bl_taus"]),
        np.asarray(per_k[2]["bl_taus"]),
        "greater",
    )
    holm = holm_bonferroni(pvals)
    # Permuted-arm control: same two arms reversed -> speed-up ~1, MW not significant.
    ctrl_taus = np.asarray(_sim([0.5, 1.0], DELTA_HEAD, seeds, cache)["taus"])
    ctrl_ratio, _, _ = ratio_bootstrap_ci(ctrl_taus, ia_taus, n_boot=2000)
    ctrl_p = mann_whitney_u(ctrl_taus, ia_taus, "two-sided")

    gates = []
    for k in (2, 8, 16, 32):
        ci_ok = float(per_k[k]["ci_lo"]) > 1.0
        gates.append(
            _gate(f"E3_ci_excl1_k{k}", float(per_k[k]["ci_lo"]), "CI_lo > 1", ci_ok)
        )
        env = float(per_k[k]["ratio_of_means"]) / float(per_k[k]["theory"])
        gates.append(
            _gate(f"E3_tracks_theory_k{k}", env, "in [0.7, 1.6]", 0.7 <= env <= 1.6)
        )
    for name, rejected in holm.items():
        gates.append(
            _gate(f"E3_holm_{name}", 1.0 if rejected else 0.0, "rejected", rejected)
        )
    gates.append(
        _gate("E3_control_near1", ctrl_ratio, "in [0.9, 1.1]", 0.9 <= ctrl_ratio <= 1.1)
    )
    gates.append(_gate("E3_control_ns", ctrl_p, "p > 0.05 (n.s.)", ctrl_p > 0.05))
    return {
        "ks": ks,
        "per_k": per_k,
        "pvals": pvals,
        "holm": holm,
        "ia_taus": ia_taus,
        "control": {"ratio": ctrl_ratio, "p": ctrl_p},
        "gates": gates,
    }


def _e4_horizon(seeds: int, horizons_sim: list[int]) -> dict[str, object]:
    """E4: episodic composition (``max_h``) witnessed by sub-linear ``N``-scaling.

    The theory identities (episodic ratio flat in H; generative gap linear) are exact.
    The non-trivial witness is that the episodic rollout's *episode count* ``N(H)``
    grows **sub-linearly** in H (order-statistic of the slowest stage), not linearly
    (``sum_h``): summing identical stages would force ``N(H)/N(1) = H`` exactly, while
    ``max_h`` gives only slow order-statistic growth. Run on cheap 2-arm IA stages so H
    can be pushed far enough that ``max_h`` and ``sum_h`` diverge unambiguously.
    """
    hs = [1, 2, 4, 8, 16]
    stage_g, stage_b = RELEVANT, _blind_arms(6)
    epi_ratio, gen_gap = [], []
    for h in hs:
        epi_ratio.append(
            mdp_char_time([stage_b] * h, budget="episodic")
            / mdp_char_time([stage_g] * h, budget="episodic")
        )
        gen_gap.append(
            mdp_char_time([stage_b] * h, budget="generative")
            - mdp_char_time([stage_g] * h, budget="generative")
        )
    # Episodic scaling on IA stages (cheap); N(H)/N(1) sub-linear -> max_h, not sum_h.
    log_inv = np.log(1.0 / DELTA_HEAD)
    scaling: dict[str, float] = {}
    ns_by_h: dict[int, np.ndarray] = {}
    for h in horizons_sim:
        ns = np.array(
            [
                episodic_track_and_stop(
                    [stage_g] * h,
                    delta=DELTA_HEAD,
                    rng=np.random.default_rng(s),
                    realloc_every=50,
                )[1]
                for s in range(seeds)
            ],
            dtype=float,
        )
        ns_by_h[h] = ns
        scaling[str(h)] = float(ns.mean()) / log_inv
    h_max = max(horizons_sim)
    h_min = min(horizons_sim)
    g_max = scaling[str(h_max)] / scaling[str(h_min)]
    # Bootstrap 95% CI on the growth ratio N(h_max)/N(h_min): the whole interval sits
    # below the additive line H, so max_h (not sum_h) is not a point-estimate artefact.
    _, g_lo, g_hi = ratio_bootstrap_ci(ns_by_h[h_max], ns_by_h[h_min], n_boot=2000)
    gates = [
        _gate(
            "E4_epi_ratio_flat",
            float(np.ptp(epi_ratio)),
            "< 1e-9",
            float(np.ptp(epi_ratio)) < 1e-9,
        ),
        _gate(
            "E4_gen_linear_r2",
            _linfit_r2(hs, gen_gap),
            "> 0.99",
            _linfit_r2(hs, gen_gap) > 0.99,
        ),
        _gate(
            f"E4_sim_sublinear_H{h_max}",
            g_max,
            f"< 0.5*H (={0.5 * h_max:g}); ={h_max} if additive",
            g_max < 0.5 * h_max,
        ),
        _gate(
            f"E4_growth_ci_sublinear_H{h_max}",
            g_hi,
            f"CI_hi < {h_max} (additive line)",
            g_hi < h_max,
        ),
    ]
    return {
        "hs": hs,
        "epi_ratio": epi_ratio,
        "gen_gap": gen_gap,
        "sim_scaling": scaling,
        "sim_growth": g_max,
        "sim_growth_ci": [g_lo, g_hi],
        "gates": gates,
    }


def _linfit_r2(x: list[int], y: list[float]) -> float:
    """R^2 of a linear fit ``y ~ x`` (for the 'generative gap is linear in H' gate)."""
    xa, ya = np.array(x, float), np.array(y, float)
    slope, intercept = np.polyfit(xa, ya, 1)
    pred = slope * xa + intercept
    ss_res = float(np.sum((ya - pred) ** 2))
    ss_tot = float(np.sum((ya - ya.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def _jsonable(obj: object) -> object:
    """Recursively drop ndarray fields and coerce numpy scalars for ``summary.json``."""
    if isinstance(obj, dict):
        return {
            k: _jsonable(v) for k, v in obj.items() if not isinstance(v, np.ndarray)
        }
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(
        obj, np.generic
    ):  # np.bool_, np.floating, np.integer -> Python scalar
        return obj.item()
    return obj


def main(argv: list[str] | None = None) -> int:
    """Run E1-E4, write the cache + summary, and print the acceptance-check table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=300)
    parser.add_argument(
        "--quick", action="store_true", help="~30 seeds, 2 deltas (smoke)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output dir (default: results/paper; --quick uses results/scratch so it "
        "never clobbers the committed 300-seed cache)",
    )
    args = parser.parse_args(argv)
    seeds = 30 if args.quick else args.seeds
    deltas = [0.1, 0.01] if args.quick else [0.1, 0.01, 1e-3, 1e-4]
    # --quick writes to a scratch dir so a smoke run cannot overwrite the committed
    # 300-seed results/paper/{summary.json,arrays.npz} that the figures/paper depend on.
    out_dir = args.out or (Path("results/scratch") if args.quick else RESULTS)
    # Episodic scaling needs H reaching 8 so max_h and sum_h diverge unambiguously
    # (g(4)=2.05 borders the 0.5*H line; g(8)=2.46 clears it with margin).
    horizons_sim = [1, 8] if args.quick else [1, 2, 4, 8]
    sim_seeds = 15 if args.quick else 40

    print(f"run: seeds={seeds}  deltas={deltas}  (quick={args.quick})")
    e1 = _e1_separation()
    print("  E1 done (theory)")
    e2 = _e2_correctness(seeds, deltas)
    print("  E2 done (delta-correctness + rate)")
    e3 = _e3_headline(seeds)
    print("  E3 done (headline speed-up)")
    e4 = _e4_horizon(sim_seeds, horizons_sim)
    print("  E4 done (horizon lift)")

    gates = e1["gates"] + e2["gates"] + e3["gates"] + e4["gates"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "arrays.npz",
        e1_ks=np.array(e1["ks"]),
        e1_ratio=np.array(e1["ratio"]),
        e1_tblind=np.array(e1["t_blind"]),
        e3_ks=np.array(e3["ks"]),
        e3_ratio=np.array([e3["per_k"][k]["ratio_of_means"] for k in e3["ks"]]),
        e3_lo=np.array([e3["per_k"][k]["ci_lo"] for k in e3["ks"]]),
        e3_hi=np.array([e3["per_k"][k]["ci_hi"] for k in e3["ks"]]),
        e3_theory=np.array([e3["per_k"][k]["theory"] for k in e3["ks"]]),
        e4_hs=np.array(e4["hs"]),
        e4_epi_ratio=np.array(e4["epi_ratio"]),
        e4_gen_gap=np.array(e4["gen_gap"]),
        e4_sim_hs=np.array([int(h) for h in e4["sim_scaling"]]),
        e4_sim_rate=np.array(list(e4["sim_scaling"].values())),
    )
    summary = {
        "config": {
            "seeds": seeds,
            "deltas": deltas,
            "sigma": SIGMA,
            "quick": args.quick,
        },
        "e1": _jsonable(e1),
        "e2": _jsonable(e2),
        "e3": _jsonable(e3),
        "e4": _jsonable(e4),
        "gates": _jsonable(gates),
        "all_pass": all(bool(g["pass"]) for g in gates),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'check':<28}{'observed':>14}  {'threshold':<22}{'result'}")
    print("-" * 78)
    for g in gates:
        obs = g["observed"]
        obs_s = f"{obs:.4g}" if isinstance(obs, (int, float)) else str(obs)
        print(
            f"{g['name']:<28}{obs_s:>14}  {g['threshold']:<22}"
            f"{'PASS' if g['pass'] else 'FAIL'}"
        )
    n_pass = sum(bool(g["pass"]) for g in gates)
    print(f"\n{n_pass}/{len(gates)} acceptance checks pass -> {out_dir}/summary.json")
    if args.quick and not summary["all_pass"]:
        print(
            "(--quick: deep-delta Clopper-Pearson checks need --seeds 300 to certify)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
