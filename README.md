# Instance-Optimal Best-Policy Identification in Decoupled Causal MDPs with a Known Graph

Code accompanying the paper *"Instance-Optimal Best-Policy Identification in Decoupled Causal MDPs with a Known Graph"*.

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Contributions](#contributions)
- [Setting up](#setting-up)
- [Repository structure](#repository-structure)
- [Using the code](#using-the-code)
- [Reproducing the paper](#reproducing-the-paper)
- [Outputs: results and figures](#outputs-results-and-figures)
- [Dependencies](#dependencies)
- [Licensing](#licensing)
- [Citation](#citation)

## Overview

Given a known causal graph, a learner can drop reward-non-ancestor interventions from a
best-policy identification problem: intervening on a non-ancestor leaves the reward at
its (already identified) observational baseline, so those interventions never need to be
ruled out. This shrinks the alternative set and, with it, the Garivier–Kaufmann
characteristic time that governs the sample complexity of δ-correct identification.

This package computes those characteristic times, quantifies the **graph-aware vs.
structure-blind separation**, implements the Track-and-Stop identification algorithm for
both the single-stage causal bandit and the decoupled (layered) causal MDP, and provides the
statistical machinery and experiment drivers behind the paper's empirical claims. It is
CPU-only — no GPU, no model training, no clusters — built on `numpy`.

## Features

- **Characteristic-time solvers.** Gaussian best-arm characteristic time `T*` and the
  optimal sampling allocation via a golden-section inner solve
  (`iats.bandit.chartime`), plus the layered- and general-MDP lifts
  (`iats.mdp.layered`, `iats.mdp.general`).
- **Separation quantification.** The graph-aware `T*_G` vs. structure-blind `T*_blind`
  ratio, shown to diverge with the number of reward-non-ancestor branches.
- **Track-and-Stop.** D-tracking sampling with forced exploration and a Gaussian GLR
  stopping rule, for bandits (`iats.bandit.trackandstop`) and layered episodic MDPs
  (`iats.mdp.layered`); these are the analysed algorithms. `iats.mdp.general` is the
  illustrative local-certification strategy over the full visitation polytope used in the
  paper's E5 study (calibrated threshold, non-confirmatory).
- **Numpy-only statistics.** Unpaired ratio-of-means bootstrap CIs, Mann–Whitney U,
  exact Clopper–Pearson bounds, and Holm–Bonferroni correction (`iats.analysis.stats`).
- **One-command reproduction.** `make reproduce` runs the powered experiments and renders
  the paper figures.

## Contributions

- A **two-sided characteristic-time theorem** for best-policy identification in causal
  MDPs with a known graph, exposing an **intrinsic, leading-order separation** from
  structure-blind reinforcement learning.
- A characterization of how the separation **composes across a horizon**: it is flat in
  `H` under the episodic budget (a worst-stage bottleneck, `max_h`) and additive
  `Θ(H·k)` under the generative budget (`sum_h`).
- A reference **Track-and-Stop implementation** with an explicit stopping threshold
  `log((K-1) ζ(2) t³/δ)` that is δ-correct by a union bound (see
  `iats.bandit.trackandstop`), plus the Garivier–Kaufmann calibrated threshold
  `log((1+log t)(K-1)/δ)` for empirical comparison (its δ-correctness is not proven);
  the characteristic-time solvers are validated against closed-form and brute-force
  solvers.
- A compact, dependency-light **experimental testbed and statistical protocol** for
  pure-exploration separation results, reproducible on a laptop.

## Setting up

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --group dev                 # install the package + dev tooling (or: make install)
uv sync --group dev --extra paper   # also install matplotlib (needed to render figures)
```

`uv` creates and manages a local `.venv`; prefix commands with `uv run` (as below) to use
it, or activate `.venv` yourself.

## Repository structure

```
src/iats/                # the installable package (import name: iats)
  bandit/                # single-stage causal bandit
    chartime.py          #   characteristic time, optimal allocation, separation
    trackandstop.py      #   Track-and-Stop identification
    instances.py         #   causal best-policy-identification testbed instances
  mdp/                   # multi-stage causal MDP
    layered.py           #   layered/episodic characteristic time + Track-and-Stop
    general.py           #   general small-MDP solvers over the visitation polytope
  analysis/              # statistical inference for the experiments
    stats.py             #   bootstrap / rank tests / Clopper–Pearson / Holm–Bonferroni

experiments/             # runnable experiment drivers (no code at the package root)
  sweeps/                # standalone demonstrations (one script per phenomenon)
    separation.py        #   T*_blind / T*_graph diverges with the branch count
    achievability.py     #   Track-and-Stop error rate and stopping time vs T*
    horizon.py           #   episodic composition is max_h; separation flat in H
    general_mdp.py       #   separation persists on non-chain / non-rectangular MDPs
  paper/                 # the paper's experiments
    run.py               #   one-command driver (experiments E1–E4) -> results/paper/
    figures.py           #   renders the paper figures -> figures/

tests/                   # pytest suite, mirroring the package layout
  bandit/  mdp/  analysis/  paper/

scripts/pre-commit/      # validate_conventions.py + check_gitflow.py (pre-commit hooks)
results/paper/           # experiment outputs (arrays.npz, summary.json) -- gitignored; regenerate with `make reproduce`
figures/                 # paper figures, by section -- gitignored; regenerate with `make reproduce`
  separation/            #   fig1_headline_separation.png
  correctness/           #   fig2_rate_convergence.png
  horizon/               #   fig3_horizon_lift.png
```

## Using the code

Compute a characteristic time and the graph-aware separation:

```python
from iats.bandit.chartime import char_time, separation

char_time([1.0, 0.5])                      # Gaussian best-arm characteristic time
separation(relevant_means=[1.0, 0.5], branch_mean=0.2, n_branches=8)
# -> {"t_graph": ..., "t_blind": ..., "ratio": ...}
```

Run Track-and-Stop on a set of arm means:

```python
import numpy as np
from iats.bandit.trackandstop import track_and_stop

chosen, tau, forced = track_and_stop([1.0, 0.5], sigma=1.0, delta=0.01,
                                     rng=np.random.default_rng(0))
```

Run the standalone demonstrations (each prints a table and a PASS/FAIL acceptance summary):

```bash
uv run python -m experiments.sweeps.separation
uv run python -m experiments.sweeps.achievability --seeds 300
uv run python -m experiments.sweeps.horizon
uv run python -m experiments.sweeps.general_mdp
```

Common development tasks are wrapped in the `Makefile`:

```bash
make check     # lint + format-check + type-check + conventions + tests (the CI gate)
make test      # run the test suite with coverage
```

## Reproducing the paper

Results and figures are **not** stored in the repository — a clean clone contains no
`results/` or `figures/`. One command regenerates them from scratch:

```bash
make reproduce     # experiments -> results/paper/ -> figures/{section}/
```

It runs these stages, each also available on its own:

```bash
make experiments   # powered run (--seeds 300; roughly 1 h, scaled from the ~4-5 min --quick run) -> results/paper/
make figures       # render figures from results/paper/ -> figures/{section}/ (needs --extra paper)
```

Everything is deterministic (fixed seeds), so a run regenerates `results/paper/` and
`figures/` and reproduces the paper's numbers exactly. Every simulation uses the analysed
stopping threshold by default; pass `--threshold calibrated` to `experiments.paper.run`
to use the Garivier–Kaufmann calibrated threshold instead (E2 is always run with both, for
comparison). The paper runs recompute the plug-in D-tracking allocation every 20 rounds
(E2/E3) and every 50 episodes (E4) — `REALLOC` / `E4_REALLOC` in
`experiments/paper/run.py`, recorded in `summary.json` under `config.realloc_every*`; the
library default `realloc_every=1` recomputes it every round, which is the analysed rule
but makes the powered run take many hours. For a fast smoke check (fewer seeds; the
deepest error-rate gates need the full run), which writes to a **scratch** dir so it never
overwrites your full-run outputs:

```bash
uv run --extra paper python -m experiments.paper.run --quick   # -> results/scratch/
```

`make figures` (and the figure step of `make reproduce`) needs matplotlib, so install the
`paper` extra first (`uv sync --group dev --extra paper`) or the Makefile targets already
prefix `uv run --extra paper`.

## Outputs: results and figures

`make reproduce` produces two directories (both gitignored — regenerate them any time with
`make reproduce`) that map directly onto the paper:

| Path | Contents | Used by |
| --- | --- | --- |
| `results/paper/summary.json` | Experiment configuration, the full pre-registered acceptance-check table, and every statistic behind the paper's claims (bootstrap CIs, Mann–Whitney p-values, Holm–Bonferroni decisions, Clopper–Pearson bounds). | The paper's claims and numbers. |
| `results/paper/arrays.npz` | Per-seed arrays and the theory/realized curves cached from the run. | `experiments.paper.figures` (re-render figures without re-simulating). |
| `figures/separation/fig1_headline_separation.png` | Theory `T*_blind/T*_G` and the realized speed-up vs. the branch count `k`. | Figure 1 (from experiments E1 + E3). |
| `figures/correctness/fig2_rate_convergence.png` | Mean stopping time relative to `T* log(1/δ)`, and the Clopper–Pearson upper bounds on the error probability. | Figure 2 (E2). |
| `figures/horizon/fig3_horizon_lift.png` | Episodic separation flat in `H`, and the additive generative gap. | Figure 3 (E4). |

Everything else under `results/` is scratch and is gitignored. To re-render the figures
from an existing `results/paper/` cache without re-running the simulation:

```bash
uv run --extra paper python -m experiments.paper.figures
```

## Dependencies

- **Runtime:** Python ≥ 3.12, `numpy`.
- **Figures (`paper` extra):** `matplotlib`.
- **Development (`dev` group):** `pytest`, `pytest-cov`, `ruff`, `mypy`.

Exact versions are pinned in `pyproject.toml` and `uv.lock`.

## Licensing

Released under the MIT License. See [LICENSE](LICENSE).

## Citation

If you use this code, please cite the accompanying paper. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff); a BibTeX stub:

```bibtex
@article{iats,
  title   = {Instance-Optimal Best-Policy Identification in Decoupled Causal MDPs with a Known Graph},
  author  = {Anonymous},
  year    = {2026},
  note    = {Under review}
}
```
