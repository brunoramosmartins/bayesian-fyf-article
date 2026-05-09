# Article Outline

**Status:** v0.1 (Phase 0 draft). Word counts are budgets, not contracts;
final lengths are reconciled in Phase 7 after the experiments are run.

Total target: **≈ 7,600 words** for the canonical English file
`article/bayesian-fyf.md`. The Portuguese translation
`article/bayesian-fyf-ptbr.md` is a faithful translation, not a rewrite,
and inherits the structure.

---

## Table of Contents (with provenance and word budget)

| #  | Section                                              | Source phase | Words |
|----|------------------------------------------------------|--------------|-------|
| 1  | Introduction: the forecast that learns               | —            | 700   |
| 2  | Bayes' theorem for budget analysts                   | Phase 1      | 700   |
| 3  | Conjugate families: closed-form updating             | Phase 2      | 900   |
| 4  | Sequential updating and shrinkage                    | Phase 3      | 800   |
| 5  | Posterior predictive: forecasting, not estimating    | Phase 4      | 700   |
| 6  | The FYF model: a complete annual cycle               | Phase 5      | 1,000 |
| 7  | Experiments and results                              | Phase 6      | 1,200 |
| 8  | Diagnostics: is the model working?                   | Phase 5      | 500   |
| 9  | Connection to the series (Articles 1–3)              | —            | 400   |
| 10 | Practical framework for budget analysts              | —            | 400   |
| 11 | Conclusion                                           | —            | 300   |
|    | **Total**                                            |              | **≈ 7,600** |

---

## Section Briefs

### 1. Introduction: the forecast that learns (~700 w)

Hook with the FYF cycle as practitioners experience it: the analyst
arrives at each quarterly review carrying months of actuals, last
quarter's forecast, and the original budget plan, and is asked to produce
a "new" number. State the article's central claim — every revision is a
Bayesian update — and preview the precision-weighted mean as the central
identity. Close with the road map of the article.

**Must contain:** the diagram from the roadmap (prior + likelihood →
posterior, posterior recycled as next prior); a one-paragraph statement
of scope and anti-scope.

### 2. Bayes' theorem for budget analysts (~700 w)

State Bayes' theorem in its continuous form, derive
$\pi(\theta\mid x) \propto f(x\mid\theta)\pi(\theta)$ from the joint
density, identify the marginal likelihood $p(x)$ as the normaliser,
and prove that the posterior is a proper density. Define MAP and
posterior mean and note when they coincide. Define equal-tailed and HPD
credible intervals and contrast them with frequentist confidence
intervals. Close with prior elicitation: budget plan ⟶ Normal prior.

**Source:** `notes/phase1-bayes-foundations.md`.
**Figures:** `figures/posterior_proper.png` (illustrating proportionality
and the normaliser).

### 3. Conjugate families: closed-form updating (~900 w)

Define conjugacy. Derive each of the four pairs in turn, in order of
relevance:

1. Normal–Normal (known $\sigma^2$): full derivation, leading to the
   precision-weighted mean and the precision-additivity identity.
2. Normal–Inverse-Gamma (unknown $\mu, \sigma^2$): state the prior, give
   the four updated hyperparameters, sketch the derivation.
3. Gamma–Poisson (incident counts).
4. Beta–Binomial (overtime proportion).

End each subsection with the FYF interpretation: pseudo-observations,
pseudo-counts, and the budget-plan-as-prior identity.

**Source:** `notes/phase2-conjugate-families.md`.
**Figures:** `figures/normal_normal_update.png`,
`figures/conjugate_update_grid.png`.

### 4. Sequential updating and shrinkage (~800 w)

State and prove the sequential = batch theorem for the Normal–Normal pair
and remark on its generalisation to exponential families. Derive the
shrinkage weight $w_0 = \tau_0 / (\tau_0 + n\tau)$, prove $w_0 \to 0$ and
$\sigma_n \to 0$, plot the shrinkage trajectory for the FYF reference
parameters. Discuss prior–data conflict: what to do when $\bar x$ is
many standard deviations from $\mu_0$ and the model is therefore wrong.

**Source:** `notes/phase3-sequential-updating.md`.
**Figures:** `figures/shrinkage_trajectory.png`,
`figures/prior_sensitivity.png`.

### 5. Posterior predictive: forecasting, not estimating (~700 w)

Argue that the parameter estimate $\theta$ is a means to an end; the
business needs the predictive distribution of next month's cost
$X_{t+1}$ and of the year-end total $\sum_{t=1}^{12} X_t$. Derive the
predictive for Normal–Normal in closed form, state the analogous
results for Gamma–Poisson and Beta–Binomial. Apply: P(over budget) at
each FYF.

**Source:** `notes/phase4-predictive-inference.md`.
**Figures:** `figures/predictive_density.png`,
`figures/yearend_forecast.png`.

### 6. The FYF model: a complete annual cycle (~1,000 w)

Walk through the model defined in `docs/model-design.md`: revision
calendar, cost decomposition, conjugate pairs, default parameters.
Carry the worked Q1 example fully — prior + 3 actuals → posterior — with
arithmetic in the body. Then run a 12-month simulation under the
on-target scenario and tabulate $\mu_n, \sigma_n,$ 95 % credible interval
for $n = 1, \ldots, 12$.

**Source:** `notes/phase5-fyf-model.md`.
**Figures:** `figures/fyf_yearly_cycle.png`,
`figures/fyf_credible_intervals.png`.

### 7. Experiments and results (~1,200 w)

Five scenarios from Phase 5 plus three diagnostic experiments from Phase
6, each with one figure and a paragraph of interpretation:

- A. on-target,
- B. optimistic,
- C. shock (mid-year cost jump),
- D. seasonal pattern,
- E. prior sensitivity (informative vs weakly informative),
- F. credible vs confidence interval comparison,
- G. Bayes factor vs AIC for model comparison,
- H. posterior predictive check on residuals.

**Source:** Phase 6 scripts in `scripts/exp_*.py`.
**Figures:** one per experiment, all saved at 300 DPI with fixed seeds,
plus one animated GIF of posterior evolution across the year.

### 8. Diagnostics: is the model working? (~500 w)

Posterior predictive checks: simulate from the predictive at month $n$
and compare with the actual $x_{n+1}$. Calibration plot. Surprise
detection rule (e.g. flag months where $x_{n+1}$ falls outside the 95 %
predictive interval). Close with a checklist for the practising analyst.

**Source:** `notes/phase5-fyf-model.md` (model checking subsection),
`src/diagnostics.py`.

### 9. Connection to the series (Articles 1–3) (~400 w)

Explicit mapping: how the LogNormal/Normal distributions from Article 2
become priors here, how the headcount Markov chain from Article 3 feeds
$n_t$ into the cost decomposition, how Monte Carlo from Article 1 is
reused for posterior predictive sampling. One paragraph per article.

### 10. Practical framework for budget analysts (~400 w)

A short, practitioner-oriented checklist: how to translate a budget
plan into a prior, how to score each FYF on shrinkage and surprise, when
to consult the predictive vs the posterior of $\theta$, when the
prior–data conflict warrants stepping outside the model.

### 11. Conclusion (~300 w)

Restate the thesis. List the three operational gains: precision-weighted
mean replaces ad hoc reweighting, monotone shrinkage replaces vague
"the forecast got tighter", posterior predictive answers
P(over budget) directly. Close with one forward-looking sentence on
hierarchical models as the natural next article.

---

## Notation Conventions

These are fixed at Phase 0 to avoid drift across phases.

- $\theta$ — parameter of interest (mean monthly cost in the base model).
- $\mu_0, \sigma_0$ — prior mean and standard deviation.
- $\sigma$ — observation standard deviation (known in the base model).
- $\tau \equiv 1/\sigma^2$ — observation precision.
- $n$ — number of observed months.
- $\bar x_n$ — sample mean of the first $n$ observations.
- $\mu_n, \sigma_n$ — posterior mean and standard deviation after $n$
  observations.
- $\pi(\theta), \pi(\theta\mid x)$ — prior and posterior densities.
- $f(x\mid\theta)$ — sampling density (likelihood when viewed as a
  function of $\theta$).
- $p(x) = \int f(x\mid\theta) \pi(\theta)\,\mathrm d\theta$ — marginal
  likelihood (evidence).
- LaTeX: display math `$$ ... $$`, inline math `$ ... $`. No `\(` `\)`.

## Open Questions for Phase 1

These are resolved in Phase 1 and back-propagated to this outline at the
end of Phase 5:

1. Does the article use natural-log notation $\log$ or base-10 $\log_{10}$
   anywhere? (Default: natural log throughout, declared in §2.)
2. Do credible intervals default to equal-tailed or HPD in the
   experiments? (Default: equal-tailed, with HPD shown once for
   comparison in §7.)
3. Does the introduction in §1 include the ASCII update-cycle diagram or
   only the rendered figure? (Default: rendered figure only;
   ASCII version stays in `docs/`.)
