"""Driver: run E1-E4, apply confirmatory stats, print the acceptance-check table.

One command reproduces the paper's experiments section:

- **E1** (theory-only): the separation ``T*_blind/T*_G`` diverges ~linearly in the
  branch count ``k`` while ``T*_G`` is flat.
- **E2** (300 seeds): the empirical error rate of IA- and blind Track-and-Stop
  (Clopper-Pearson 95% upper bound compared with ``delta``; where ``n`` cannot resolve
  ``delta`` the check is "0 observed errors") and the rate ``tau/log(1/delta)`` compared
  with ``T*`` across ``delta``. Run with the primary threshold (``--threshold``, default
  ``analysed``) and repeated with the other threshold, stored under ``e2_<other>`` so
  the finite-delta rates of the two thresholds can be compared (comparison only, not
  part of the gate family). The pre-specified ``E2_rate_flat`` gate compares
  ``tau/(T* log(1/delta))`` between the two deepest deltas; under the analysed
  threshold that ratio tracks ``beta_an(tau, delta)/log(1/delta)`` and is not flat, so
  the gate is expected to fail and each row also carries the non-gated diagnostic
  ``rate_over_tstar_beta = tau / (T* beta(tau, delta))`` (near 1 when the rate is
  attained).
- **E3** (300 seeds, headline): the realized speed-up ``tau_blind/tau_IA`` (ratio of
  means, unpaired bootstrap CI) is compared with the characteristic-time ratio
  (pre-specified envelope ``[0.7, 1.6]``); the CI excludes 1 for ``k >= 2``;
  Holm-Bonferroni over H1-H5; a permuted-arm control sits near 1.
- **E4** (theory + episodic sim): the episodic layered-MDP composition lands on
  ``max_h`` (not ``sum_h``); the per-stage-freezing episodic Track-and-Stop's episode
  count grows sub-linearly in ``H``.

Every simulated cell reports how many runs hit the step/episode cap (``capped``); the
summary records the threshold, allocation-cache periods, package version and git commit.
Gates are pre-specified and reported whether they pass or fail.

Writes per-seed arrays to ``results/paper/arrays.npz`` and the scalar/gate summary to
``results/paper/summary.json`` (both gitignored); :mod:`experiments.paper.figures`
renders the figures from that cache without re-simulating.

Run::

    .venv/bin/python -m experiments.paper.run --seeds 300  # powered run
    .venv/bin/python -m experiments.paper.run --quick      # smoke -> results/scratch/
"""

from __future__ import annotations

import argparse
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
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
from iats.bandit.trackandstop import MAX_STEPS, THRESHOLDS, threshold_fn
from iats.mdp.layered import MAX_EPISODES, episodic_track_and_stop, mdp_char_time

SIGMA = 1.0
RELEVANT = [1.0, 0.5]
BRANCH_MEAN = 0.2
DELTA_0 = 1.0 - BRANCH_MEAN  # branch gap (0.8)
DELTA_HEAD = 0.01  # headline delta for E3/E4
# Allocation-cache periods of the paper runs (the library default is 1 = recompute the
# plug-in allocation every round, the analysed rule). One plug-in solve costs ~7 ms, so
# per-round recomputation would turn the powered run into hours; the periods used are
# recorded in the summary metadata and must be stated wherever the runs are described.
REALLOC = 20
E4_REALLOC = 50
RESULTS = Path("results/paper")


def _blind_arms(k: int) -> list[float]:
    """Structure-blind arms: the relevant arms plus ``k`` non-ancestor branches."""
    return [*RELEVANT, *([BRANCH_MEAN] * k)]


def _sim(
    arms: list[float],
    delta: float,
    seeds: int,
    cache: dict[tuple[tuple[float, ...], float], dict[str, object]],
    threshold: str,
) -> dict[str, object]:
    """Run (or fetch from ``cache``) Track-and-Stop over seeds at ``REALLOC``."""
    key = (tuple(arms), delta)
    if key not in cache:
        cache[key] = run_trials(
            arms, SIGMA, delta, seeds, realloc_every=REALLOC, threshold=threshold
        )
    return cache[key]


def _gate(
    name: str, observed: float, threshold: str, passed: bool
) -> dict[str, object]:
    """One row of the pre-specified acceptance-check table."""
    return {"name": name, "observed": observed, "threshold": threshold, "pass": passed}


def _code_version() -> dict[str, str]:
    """Package version and ``git describe --always --dirty`` (summary metadata).

    Returns:
        ``{"iats_version": ..., "git_commit": ...}``; ``"unknown"`` where unavailable.
    """
    try:
        pkg = version("iats")
    except PackageNotFoundError:
        pkg = "unknown"
    try:
        commit = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {"iats_version": pkg, "git_commit": commit}


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


def _e2_correctness(
    seeds: int, deltas: list[float], threshold: str = "analysed"
) -> dict[str, object]:
    """E2: error rate (Clopper-Pearson) and rate vs ``T*`` at k=8 for one threshold."""
    cache: dict[tuple[tuple[float, ...], float], dict[str, object]] = {}
    beta_fn = threshold_fn(threshold)
    ia_arms, bl_arms = RELEVANT, _blind_arms(8)
    rows: dict[str, dict[str, object]] = {}
    for delta in deltas:
        log_inv = np.log(1.0 / delta)
        for label, arms in (("ia", ia_arms), ("blind", bl_arms)):
            r = _sim(arms, delta, seeds, cache, threshold)
            errors, taus = int(r["errors"]), np.asarray(r["taus"])
            cp = clopper_pearson_upper(errors, seeds)
            mean_tau, tstar = float(taus.mean()), float(r["tstar"])
            rows[f"{label}_d{delta:g}"] = {
                "delta": delta,
                "errors": errors,
                "capped": int(r["capped"]),
                "cp_upper": cp,
                "rate": mean_tau / log_inv,
                "rate_over_tstar": (mean_tau / log_inv) / tstar,
                # Diagnostic (not gated): normalised by the threshold actually crossed.
                "rate_over_tstar_beta": mean_tau
                / beta_fn(mean_tau, delta, len(arms))
                / tstar,
                "forced_frac": float(np.asarray(r["forced"]).mean() / taus.mean()),
                "taus": taus,
                "wrong": np.asarray(r["wrong"]),
            }
    gates = []
    # Error rate. The exact 95% CP bound only resolves down to cp_floor = cp(0, n);
    # below that, n cannot show err <= delta, so we check the weaker "0 errors observed"
    # and flag the check as seed-limited.
    cp_floor = clopper_pearson_upper(0, seeds)
    for key, row in rows.items():
        d = float(row["delta"])
        if d >= cp_floor:  # resolvable at this seed count
            gates.append(
                _gate(
                    f"E2_cp_{key}",
                    float(row["cp_upper"]),
                    f"cp <= {d:g}",
                    row["cp_upper"] <= d,
                )
            )
        else:  # deeper than n can resolve: require 0 observed errors instead
            errs = int(row["errors"])
            gates.append(
                _gate(
                    f"E2_cp_{key}",
                    errs,
                    f"0 err (cp {cp_floor:.4f} > {d:g}, n-limited)",
                    errs == 0,
                )
            )
    # rate flatness for IA between the two deepest deltas (NOT convergence to 1). Under
    # the analysed threshold the ratio tracks beta_an(tau)/log(1/delta) and is not flat,
    # so this pre-specified gate is expected to FAIL; see rate_over_tstar_beta above.
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
    return {"threshold": threshold, "rows": rows, "gates": gates}


def _e3_headline(seeds: int, threshold: str = "analysed") -> dict[str, object]:
    """E3: realized speed-up vs theory, unpaired ratio CI + Holm Mann-Whitney."""
    cache: dict[tuple[tuple[float, ...], float], dict[str, object]] = {}
    ks = [0, 2, 4, 8, 16, 32]
    ia = _sim(RELEVANT, DELTA_HEAD, seeds, cache, threshold)
    ia_taus = np.asarray(ia["taus"])
    per_k: dict[int, dict[str, object]] = {}
    pvals: dict[str, float] = {}
    for k in ks:
        bl = _sim(_blind_arms(k), DELTA_HEAD, seeds, cache, threshold)
        bl_taus = np.asarray(bl["taus"])
        ratio, lo, hi = ratio_bootstrap_ci(bl_taus, ia_taus, n_boot=2000)
        theory = separation(RELEVANT, BRANCH_MEAN, k, SIGMA)["ratio"]
        per_k[k] = {
            "ratio_of_means": ratio,
            "ci_lo": lo,
            "ci_hi": hi,
            "median": float(np.median(bl_taus) / np.median(ia_taus)),
            "theory": theory,
            "capped": int(bl["capped"]),
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
    ctrl = _sim([0.5, 1.0], DELTA_HEAD, seeds, cache, threshold)
    ctrl_taus = np.asarray(ctrl["taus"])
    ctrl_ratio, _, _ = ratio_bootstrap_ci(ctrl_taus, ia_taus, n_boot=2000)
    ctrl_p = mann_whitney_u(ctrl_taus, ia_taus, "two-sided")

    gates = []
    for k in (2, 8, 16, 32):
        ci_ok = float(per_k[k]["ci_lo"]) > 1.0
        gates.append(
            _gate(f"E3_ci_excl1_k{k}", float(per_k[k]["ci_lo"]), "CI_lo > 1", ci_ok)
        )
        # Pre-specified envelope on realized / characteristic-time ratio at delta=0.01.
        rel = float(per_k[k]["ratio_of_means"]) / float(per_k[k]["theory"])
        gates.append(
            _gate(f"E3_tracks_theory_k{k}", rel, "in [0.7, 1.6]", 0.7 <= rel <= 1.6)
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
        "threshold": threshold,
        "ks": ks,
        "per_k": per_k,
        "pvals": pvals,
        "holm": holm,
        "ia_taus": ia_taus,
        "ia_capped": int(ia["capped"]),
        "ctrl_taus": ctrl_taus,
        "ctrl_capped": int(ctrl["capped"]),
        "control": {"ratio": ctrl_ratio, "p": ctrl_p},
        "gates": gates,
    }


def _e4_horizon(
    seeds: int, horizons_sim: list[int], threshold: str = "analysed"
) -> dict[str, object]:
    """E4: episodic composition (``max_h``) witnessed by sub-linear ``N``-scaling.

    The theory identities (episodic ratio flat in H; generative gap linear) are exact.
    The non-trivial witness is that the episodic rollout's *episode count* ``N(H)``
    grows **sub-linearly** in H (order-statistic of the slowest stage), not linearly
    (``sum_h``): summing identical stages would force ``N(H)/N(1) = H`` exactly, while
    ``max_h`` gives only slow order-statistic growth. Run on cheap 2-arm IA stages so H
    can be pushed far enough that ``max_h`` and ``sum_h`` diverge unambiguously. The
    per-seed episode counts and per-stage stop episodes are returned as ``ns_H{h}`` and
    ``stops_H{h}`` arrays.
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
    arrays: dict[str, np.ndarray] = {}
    capped = 0
    for h in horizons_sim:
        runs = [
            episodic_track_and_stop(
                [stage_g] * h,
                delta=DELTA_HEAD,
                rng=np.random.default_rng(s),
                realloc_every=E4_REALLOC,
                threshold=threshold,
            )
            for s in range(seeds)
        ]
        ns = np.array([r[1] for r in runs], dtype=float)
        arrays[f"ns_H{h}"] = ns
        arrays[f"stops_H{h}"] = np.array([r[2] for r in runs], dtype=float)
        capped += int((ns >= MAX_EPISODES).sum())
        scaling[str(h)] = float(ns.mean()) / log_inv
    h_max = max(horizons_sim)
    h_min = min(horizons_sim)
    g_max = scaling[str(h_max)] / scaling[str(h_min)]
    # Bootstrap 95% CI on the growth ratio N(h_max)/N(h_min): the whole interval sits
    # below the additive line H, so max_h (not sum_h) is not a point-estimate artefact.
    _, g_lo, g_hi = ratio_bootstrap_ci(
        arrays[f"ns_H{h_max}"], arrays[f"ns_H{h_min}"], n_boot=2000
    )
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
        "threshold": threshold,
        "hs": hs,
        "epi_ratio": epi_ratio,
        "gen_gap": gen_gap,
        "sim_scaling": scaling,
        "sim_growth": g_max,
        "sim_growth_ci": [g_lo, g_hi],
        "sim_capped": capped,
        "gates": gates,
        **arrays,
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
        "never clobbers the powered-run cache)",
    )
    parser.add_argument(
        "--threshold",
        choices=THRESHOLDS,
        default="analysed",
        help="stopping threshold for every simulation (E2-E4): 'analysed' = "
        "log((K-1) zeta(2) t^3/delta), delta-correct by a union bound; 'calibrated' = "
        "log((1+log t)(K-1)/delta), the Garivier-Kaufmann experimental threshold whose "
        "delta-correctness is not proven. E2 is also run with the other threshold "
        "(summary key e2_<other>) for comparison.",
    )
    args = parser.parse_args(argv)
    seeds = 30 if args.quick else args.seeds
    deltas = [0.1, 0.01] if args.quick else [0.1, 0.01, 1e-3, 1e-4]
    # --quick writes to a scratch dir so a smoke run cannot overwrite the powered-run
    # results/paper/{summary.json,arrays.npz} that the figures/paper depend on.
    out_dir = args.out or (Path("results/scratch") if args.quick else RESULTS)
    # Episodic scaling needs H reaching 8 so max_h and sum_h diverge unambiguously.
    horizons_sim = [1, 8] if args.quick else [1, 2, 4, 8]
    sim_seeds = 15 if args.quick else 40
    other = next(t for t in THRESHOLDS if t != args.threshold)

    print(
        f"run: seeds={seeds}  deltas={deltas}  threshold={args.threshold}  "
        f"(quick={args.quick})"
    )
    e1 = _e1_separation()
    print("  E1 done (theory)")
    e2 = _e2_correctness(seeds, deltas, args.threshold)
    print(f"  E2 done (error rate + rate, {args.threshold})")
    e2_other = _e2_correctness(seeds, deltas, other)  # comparison only, not gated
    print(f"  E2 done ({other} threshold, comparison)")
    e3 = _e3_headline(seeds, args.threshold)
    print("  E3 done (headline speed-up)")
    e4 = _e4_horizon(sim_seeds, horizons_sim, args.threshold)
    print("  E4 done (horizon lift)")

    gates = e1["gates"] + e2["gates"] + e3["gates"] + e4["gates"]
    per_seed: dict[str, np.ndarray] = {}
    for exp in (e2, e2_other):
        for key, row in exp["rows"].items():
            per_seed[f"e2_{exp['threshold']}_{key}_taus"] = row["taus"]
            per_seed[f"e2_{exp['threshold']}_{key}_wrong"] = row["wrong"]
    per_seed["e3_ia_taus"] = e3["ia_taus"]
    per_seed["e3_ctrl_taus"] = e3["ctrl_taus"]
    for k in e3["ks"]:
        per_seed[f"e3_bl_taus_k{k}"] = e3["per_k"][k]["bl_taus"]
    for h in horizons_sim:
        per_seed[f"e4_ns_H{h}"] = e4[f"ns_H{h}"]
        per_seed[f"e4_stops_H{h}"] = e4[f"stops_H{h}"]
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
        **per_seed,
    )
    capped = {
        "e2": sum(int(r["capped"]) for r in e2["rows"].values()),
        f"e2_{other}": sum(int(r["capped"]) for r in e2_other["rows"].values()),
        "e3": int(e3["ia_capped"])
        + int(e3["ctrl_capped"])
        + sum(int(v["capped"]) for v in e3["per_k"].values()),
        "e4": int(e4["sim_capped"]),
    }
    summary = {
        "config": {
            "seeds": seeds,
            "deltas": deltas,
            "sigma": SIGMA,
            "quick": args.quick,
            "threshold": args.threshold,
            "e2_comparison_threshold": other,
            "realloc_every": REALLOC,
            "realloc_every_e4": E4_REALLOC,
            "max_steps": MAX_STEPS,
            "max_episodes": MAX_EPISODES,
            **_code_version(),
        },
        "e1": _jsonable(e1),
        "e2": _jsonable(e2),
        f"e2_{other}": _jsonable(e2_other),
        "e3": _jsonable(e3),
        "e4": _jsonable(e4),
        "gates": _jsonable(gates),
        "capped_runs": capped,
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
    print(f"capped runs (hit the step/episode cap): {capped}")
    if args.quick and not summary["all_pass"]:
        print(
            "(--quick: deep-delta Clopper-Pearson checks need --seeds 300 to resolve)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
