"""Render the paper figures from the ``results/paper`` cache (no re-simulation).

Reads ``results/paper/arrays.npz`` and ``results/paper/summary.json`` (produced by
:mod:`experiments.paper.run`) and writes the three figures into per-section
subdirectories of ``figures/`` (``separation/fig1_headline_separation.png``,
``correctness/fig2_rate_convergence.png``, ``horizon/fig3_horizon_lift.png``).
matplotlib ``Agg`` backend, deterministic -- iterate on figures without re-running the
powered simulation. ``make paper`` copies them flat into ``.paper/figures/``.

Run::

    .venv/bin/python -m experiments.paper.figures
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

RESULTS = Path("results/paper")
FIGDIR = Path("figures")


def fig1_headline(npz: dict[str, np.ndarray], plt: Any) -> None:
    """The money shot: exact theory ratio + realized speed-up with a CI band."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(
        npz["e1_ks"],
        npz["e1_ratio"],
        "-",
        color="C0",
        label=r"theory $T^*_{blind}/T^*_G$",
    )
    ax.plot(
        npz["e3_ks"],
        npz["e3_ratio"],
        "o-",
        color="C3",
        label=r"realized $\tau_{blind}/\tau_{IA}$",
    )
    ax.fill_between(npz["e3_ks"], npz["e3_lo"], npz["e3_hi"], color="C3", alpha=0.2)
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("number of reward-non-ancestor branches $k$")
    ax.set_ylabel("separation ratio")
    ax.set_title("Intrinsic, leading-order separation (theory) and its realization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "separation" / "fig1_headline_separation.png", dpi=150)
    plt.close(fig)


def fig2_rate_correctness(summary: dict[str, Any], plt: Any) -> None:
    """Left: rate/T* decreases toward 1 as delta->0. Right: Clopper-Pearson bound."""
    rows = summary["e2"]["rows"]
    deltas = sorted({r["delta"] for r in rows.values()})
    xs = [np.log(1.0 / d) for d in deltas]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for label, color in (("ia", "C0"), ("blind", "C3")):
        rate = [rows[f"{label}_d{d:g}"]["rate_over_tstar"] for d in deltas]
        axl.plot(xs, rate, "o-", color=color, label=label.upper())
    axl.axhline(1.0, color="gray", lw=0.8, ls=":", label=r"$T^*$ (asymptote)")
    axl.set_xlabel(r"$\log(1/\delta)$")
    axl.set_ylabel(r"rate $/\ T^*$")
    axl.set_title(r"Rate decreases toward $T^*$ as $\delta\to0$ ($1+o(1)$)")
    axl.legend()
    axl.grid(True, alpha=0.3)
    width = 0.35
    idx = np.arange(len(deltas))
    for j, (label, color) in enumerate((("ia", "C0"), ("blind", "C3"))):
        cp = [rows[f"{label}_d{d:g}"]["cp_upper"] for d in deltas]
        axr.bar(
            idx + j * width, cp, width, color=color, label=f"{label.upper()} CP upper"
        )
    for j, d in enumerate(deltas):
        axr.plot([idx[j] - 0.2, idx[j] + 0.55], [d, d], "k--", lw=1.0)
    axr.set_yscale("log")
    axr.set_xticks(idx + width / 2)
    axr.set_xticklabels([f"{d:g}" for d in deltas])
    axr.set_xlabel(r"$\delta$")
    axr.set_ylabel("error upper bound (95% Clopper-Pearson)")
    axr.set_title(r"$\delta$-correctness certified (bound $\leq \delta$, dashed)")
    axr.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "correctness" / "fig2_rate_convergence.png", dpi=150)
    plt.close(fig)


def fig3_horizon(npz: dict[str, np.ndarray], summary: dict[str, Any], plt: Any) -> None:
    """A: episodic ratio flat in H + realized N-scaling. B: generative gap linear."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axa.plot(
        npz["e4_hs"],
        npz["e4_epi_ratio"],
        "s-",
        color="C0",
        label="episodic ratio (theory)",
    )
    axa.set_xlabel("horizon $H$")
    axa.set_ylabel(r"$T^*_{blind}/T^*_G$ (episodic)")
    axa.set_title("Episodic separation: flat in $H$ (worst-stage $\\max_h$)")
    axa.set_ylim(0, max(npz["e4_epi_ratio"]) * 1.5)
    axa.grid(True, alpha=0.3)
    # Realized episodic N-scaling: sublinear in H (rules out summing) on a twin axis.
    scale = summary["e4"].get("sim_scaling")
    if scale:
        hs = sorted(int(h) for h in scale)
        g = [scale[str(h)] / scale[str(hs[0])] for h in hs]
        axt = axa.twinx()
        axt.plot(hs, g, "o--", color="C3", label=r"realized $N(H)/N(1)$")
        axt.plot(hs, hs, ":", color="gray", label=r"$H$ (if additive)")
        axt.set_ylabel(r"episode-count growth $N(H)/N(1)$")
        axt.legend(loc="upper left")
    axa.legend(loc="lower right")
    axb.plot(
        npz["e4_hs"],
        npz["e4_gen_gap"],
        "^-",
        color="C2",
        label="generative gap (theory)",
    )
    axb.set_xlabel("horizon $H$")
    axb.set_ylabel(r"$T^*_{blind}-T^*_G$ (generative)")
    axb.set_title(r"Generative gap: additive $\Theta(Hk)$")
    axb.legend()
    axb.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "horizon" / "fig3_horizon_lift.png", dpi=150)
    plt.close(fig)


def main() -> int:
    """Load the cache and render the three figures into ``figures/{section}/``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npz = dict(np.load(RESULTS / "arrays.npz"))
    summary = json.loads((RESULTS / "summary.json").read_text())
    for section in ("separation", "correctness", "horizon"):
        (FIGDIR / section).mkdir(parents=True, exist_ok=True)
    fig1_headline(npz, plt)
    fig2_rate_correctness(summary, plt)
    fig3_horizon(npz, summary, plt)
    print(f"figures -> {FIGDIR}/{{separation,correctness,horizon}}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
