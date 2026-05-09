# The Budget That Learns

**Bayesian Forecasting for Periodic Budget Revisions — from Prior Beliefs to Posterior Certainty**

Companion repository for a portfolio-grade technical article that models
periodic FYF (Forecast Year-end Financial) budget revisions as sequential
Bayesian updating. The traditional approach treats each forecast revision
as a fresh estimate; the Bayesian approach recognises that each revision
is a refinement — the prior (last forecast) is updated with new evidence
(actuals) to produce a posterior (revised forecast) that is sharper, more
calibrated, and mathematically justified.

This is article 4 of a four-part series on probabilistic methods for
budget analytics:

| # | Article            | Math branch              |
|---|--------------------|--------------------------|
| 1 | Monte Carlo budget | Simulation theory        |
| 2 | Distributions      | Statistical inference    |
| 3 | Headcount          | Stochastic processes     |
| **4** | **This article**   | **Bayesian inference**   |

## What is here

```
bayesian-fyf-article/
├── article/      # Final article — bayesian-fyf.md (EN canonical) + ptbr translation
├── docs/         # Foundational documents — thesis, FYF model, article outline
├── src/          # Library code: priors, conjugate updaters, predictive, FYF model
├── scripts/      # Reproducible experiment runners
├── notebooks/    # Exploratory notebooks per phase
├── notes/        # Phase-by-phase derivations (pre-article)
├── exercises/    # Paper exercises, one set per phase
├── til/          # "Today I Learned" — short publishable insights
├── tests/        # pytest unit tests
├── figures/      # Publication-quality figures (300 DPI, fixed seeds)
└── .github/setup # bash + gh scripts to bootstrap labels, milestones, issues
```

The roadmap that drives every phase is `roadmap-bayesian-fyf-v1.md`.

## Local setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Unix
source .venv/bin/activate

pip install -e .[dev,notebook]
```

There is intentionally no `requirements.txt`; dependencies live under
`[project.dependencies]` in `pyproject.toml`.

Verify the install:

```bash
pytest tests/
ruff check .
```

## GitHub bootstrap

Once the remote is in place, create labels, milestones, and the full
issue list with:

```bash
bash .github/setup/labels.sh    brunoramosmartins/bayesian-fyf-article
bash .github/setup/milestones.sh brunoramosmartins/bayesian-fyf-article
bash .github/setup/issues.sh    brunoramosmartins/bayesian-fyf-article
```

These scripts use the `gh` CLI; they are idempotent for labels (`--force`)
but not for milestones or issues — run them once.

## Project status

| Phase | Title                                  | Status   | Tag                      |
|-------|----------------------------------------|----------|--------------------------|
| 0     | Foundation                             | active   | `v0.1-foundation`        |
| 1     | Bayesian Foundations                   | planned  | `v0.2-bayes-foundations` |
| 2     | Conjugate Families                     | planned  | `v0.3-conjugate-families`|
| 3     | Sequential Updating                    | planned  | `v0.4-sequential-updating`|
| 4     | Predictive Inference                   | planned  | `v0.5-predictive-inference`|
| 5     | Applied FYF Model                      | planned  | `v0.6-fyf-model`         |
| 6     | Experiments & Visualisations           | planned  | `v0.7-experiments`       |
| 7     | Article Writing                        | planned  | `v0.8-article-draft`     |
| 8     | Review & Publish                       | planned  | `v1.0.0`                 |

Public releases are cut at Phase 6 (pre-release), Phase 7 (pre-release),
and Phase 8 (stable).

## Reading the article

The canonical article is `article/bayesian-fyf.md` (English). The
Portuguese translation lives at `article/bayesian-fyf-ptbr.md`. Both are
plain Markdown with `$$...$$` LaTeX blocks; the canonical English file is
processed by the author's external `github.io` MD → HTML pipeline.

## Author

Bruno Ramos Martins — Analytics Engineer transitioning to Data Science /
Machine Learning. Background in Mathematics; current work in IT
headcount budgeting.

## License

MIT — see `LICENSE`.
