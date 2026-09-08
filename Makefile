# IATS (Intervention-Aware Track-and-Stop) -- one-command workflows.
#
# Quick start:
#   make install    # install dev dependencies (uv)
#   make check      # lint + format-check + type-check + conventions + tests (CI gate)
#   make test       # run the test suite with coverage
#   make experiments  # run the powered experiments -> results/paper/
#   make figures      # render the paper figures -> figures/{section}/
#   make reproduce    # experiments -> results -> figures (clean-clone reproduction)
#   make paper        # AUTHOR-ONLY: assemble the submission bundle (needs local .paper/)

.DEFAULT_GOAL := help
.PHONY: help install lint format format-check typecheck conventions test check \
        experiments figures witnesses paper reproduce clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install dev dependencies.
	uv sync --group dev

lint: ## Lint with ruff.
	uv run ruff check .

format: ## Auto-format with ruff.
	uv run ruff format .

format-check: ## Check formatting without modifying files.
	uv run ruff format --check .

typecheck: ## Type-check the package with mypy.
	uv run mypy src

conventions: ## Run the project convention validator over src + experiments + tests.
	uv run python scripts/pre-commit/validate_conventions.py \
		$$(find src experiments tests -name '*.py') conftest.py

test: ## Run the test suite with coverage.
	uv run pytest

check: lint format-check typecheck conventions test ## Full CI gate.

experiments: ## Run the powered experiments (roughly 1 h, scaled from --quick) -> results/paper/.
	uv run --extra paper python -m experiments.paper.run --seeds 300

figures: ## Render the paper figures from results/paper/ -> figures/{section}/.
	uv run --extra paper python -m experiments.paper.figures

witnesses: ## Run the E5 deep-dive -> results/paper/e5_witnesses.json.
	uv run python -m experiments.sweeps.general_mdp

reproduce: experiments witnesses figures ## Reproduce experiments + E5 witnesses + figures (clean clone).

paper: figures ## AUTHOR-ONLY: copy figures into local .paper/ and rebuild the submission zip.
	@test -f .paper/main.tex || { echo "error: .paper/ manuscript sources are author-only and not in the public repo (make paper is not part of 'make reproduce')"; exit 1; }
	rm -rf .paper/figures && mkdir -p .paper/figures
	cp figures/separation/fig1_headline_separation.png .paper/figures/
	cp figures/correctness/fig2_rate_convergence.png   .paper/figures/
	cp figures/horizon/fig3_horizon_lift.png           .paper/figures/
	cd .paper && rm -f dcc-tmlr-submission.zip && \
		zip -r dcc-tmlr-submission.zip main.tex tmlr.sty tmlr.bst fancyhdr.sty references.bib figures
	@echo "Submission bundle -> .paper/dcc-tmlr-submission.zip"

clean: ## Remove caches and coverage artefacts.
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
