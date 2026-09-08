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
import typing as tp
from pathlib import Path

import numpy as np

RESULTS = Path("results/paper")
FIGDIR = Path("figures")


def fig1_headline(npz: dict[str, np.ndarray], plt: tp.Any) -> None:
    """Characteristic-time ratio (theory) and realized ratio of mean stopping times."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(
        npz["e1_ks"],
        npz["e1_ratio"],
        "-",
        color="C0",
        label=r"characteristic-time ratio $T^*_{blind}/T^*_G$",
    )
    ax.plot(
        npz["e3_ks"],
        npz["e3_ratio"],
        "o-",
        color="C3",
        label=r"realized ratio of mean stopping times $\bar\tau_{blind}/\bar\tau_{IA}$",
    )
    ax.fill_between(
        npz["e3_ks"],
        npz["e3_lo"],
        npz["e3_hi"],
        color="C3",
        alpha=0.2,
        label="95% bootstrap CI",
    )
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("number of reward-non-ancestor branches $k$")
    ax.set_ylabel("ratio (blind / graph-aware)")
    ax.set_title(r"Characteristic-time ratio and realized stopping-time ratio vs $k$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "separation" / "fig1_headline_separation.png", dpi=150)
    plt.close(fig)


def fig2_rate_correctness(summary: dict[str, tp.Any], plt: tp.Any) -> None:
    """Left: mean tau / (T* log(1/delta)) per delta. Right: Clopper-Pearson bound."""
    rows = summary["e2"]["rows"]
    deltas = sorted({r["delta"] for r in rows.values()})
    xs = [np.log(1.0 / d) for d in deltas]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for label, color in (("ia", "C0"), ("blind", "C3")):
        rate = [rows[f"{label}_d{d:g}"]["rate_over_tstar"] for d in deltas]
        axl.plot(xs, rate, "o-", color=color, label=label.upper())
    axl.axhline(1.0, color="gray", lw=0.8, ls=":", label=r"$T^*\log(1/\delta)$")
    axl.set_xlabel(r"$\log(1/\delta)$")
    axl.set_ylabel(r"$\bar\tau\ /\ (T^*\log(1/\delta))$")
    axl.set_title(r"Mean stopping time relative to $T^*\log(1/\delta)$")
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
        label = r"target $\delta$" if j == 0 else None
        axr.plot([idx[j] - 0.2, idx[j] + 0.55], [d, d], "k--", lw=1.0, label=label)
    axr.set_yscale("log")
    axr.set_xticks(idx + width / 2)
    axr.set_xticklabels([f"{d:g}" for d in deltas])
    axr.set_xlabel(r"$\delta$")
    axr.set_ylabel("error-probability upper bound (95% Clopper-Pearson)")
    axr.set_title("Error probability: 95% Clopper-Pearson upper bound")
    axr.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "correctness" / "fig2_rate_convergence.png", dpi=150)
    plt.close(fig)


def fig3_horizon(
    npz: dict[str, np.ndarray], summary: dict[str, tp.Any], plt: tp.Any
) -> None:
    """A: episodic ratio vs H with realized episode-count growth. B: generative gap."""
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
    axa.set_title(r"Episodic ratio $T^*_{blind}/T^*_G$ and realized $N(H)/N(1)$")
    axa.set_ylim(0, max(npz["e4_epi_ratio"]) * 1.5)
    axa.grid(True, alpha=0.3)
    # Realized N-scaling on a twin axis; the additive line H is the reference.
    scale = summary["e4"].get("sim_scaling")
    if scale:
        hs = sorted(int(h) for h in scale)
        g = [scale[str(h)] / scale[str(hs[0])] for h in hs]
        axt = axa.twinx()
        axt.plot(hs, g, "o--", color="C3", label=r"realized mean $N(H)/N(1)$")
        axt.plot(hs, hs, ":", color="gray", label=r"$H$ (additive reference)")
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
    axb.set_title(r"Generative gap $T^*_{blind}-T^*_G$ vs $H$")
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
    if "threshold" not in summary["config"]:  # cache from before the threshold change
        raise SystemExit(
            f"{RESULTS}/summary.json predates the analysed threshold (no "
            "config.threshold); re-run `make experiments` before rendering figures"
        )
    for section in ("separation", "correctness", "horizon"):
        (FIGDIR / section).mkdir(parents=True, exist_ok=True)
    fig1_headline(npz, plt)
    fig2_rate_correctness(summary, plt)
    fig3_horizon(npz, summary, plt)
    print(f"figures -> {FIGDIR}/{{separation,correctness,horizon}}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
