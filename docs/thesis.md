# Thesis — The Budget That Learns

**Status:** v0.1 (draft, Phase 0). Will be revised at the end of Phase 5
once the FYF model is fully simulated, and re-confirmed at the end of
Phase 7 against the final article text.

---

## Central Claim

A budget forecast that ignores its own history is a memoryless estimator —
it discards the information accumulated in previous revisions and observed
actuals. Bayesian updating provides the mathematically optimal way to
combine prior beliefs (the original budget) with incoming evidence
(monthly actuals): the posterior forecast is always at least as good as
the prior, it sharpens monotonically as data arrives, and it automatically
balances confidence in the plan against confidence in the data through
the **precision-weighted mean**. **Each FYF cycle is not a new forecast —
it is a Bayesian update.**

The value the article delivers is not a new statistical result. It is the
explicit, rigorous identification of a process that budget analysts
already perform — periodic forecast revision — with the canonical
machinery of Bayesian inference. Once the identification is made,
shrinkage, calibration, prior–data conflict, and posterior predictive
forecasting cease to be abstract concepts and become operational tools
for the FYF cycle.

## Central Axis (one-sentence operationalisation)

Every forecast revision is Bayes' theorem in disguise.

```
Prior (budget plan)  +  Likelihood (observed actuals)  →  Posterior (revised forecast)
         ↑                                                          │
         └────────── becomes the new prior ◄────────────────────────┘
```

## Audience

Primary: a reader with a solid undergraduate background in calculus and
probability (integration by substitution, completing the square, basic
exponential families, the Normal/LogNormal/Poisson PDFs, the laws of
total expectation and total variance) who has seen Bayes' theorem at the
discrete level and wants the continuous version derived in full and
applied to a finance problem.

Secondary: practitioners in FP&A, finance partnership, or analytics
engineering who run periodic budget reviews and recognise the FYF cycle
but have never had the underlying logic made explicit.

The article is **self-contained**: every result that is not the central
contribution is either derived from the assumed background or stated as a
classical black-box result with a citation.

## Scope

The article covers, in full derivation:

- Bayes' theorem for continuous parameters: posterior, marginal
  likelihood, MAP vs posterior mean, credible intervals (equal-tailed and
  HPD), and prior elicitation strategies.
- Four conjugate prior–likelihood pairs:
  Normal-Normal (known variance),
  Normal–Inverse-Gamma (unknown mean and variance),
  Gamma–Poisson, and Beta–Binomial.
- Sequential updating: equivalence of sequential and batch updating in
  exponential families; the shrinkage weight
  $w_0 = \tau_0 / (\tau_0 + n\tau)$ and its asymptotics.
- Posterior predictive distributions for forecasting future months and
  the year-end total; the probability of ending the year over budget.
- Prior sensitivity analysis: how much the prior choice matters at $n=1$,
  at one quarter, and at year-end.
- Model checking: posterior predictive checks, calibration, and
  surprise detection.
- The applied FYF model: the revision calendar, the cost decomposition
  $X_t = n_t \bar S_t \beta + C_{\text{ot},t} + C_{\text{inc},t}$, and a
  full annual simulation across five scenarios (on-target, optimistic,
  shock, seasonal, prior-sensitivity).

## Anti-Scope

The article deliberately does **not** cover:

- MCMC methods (Gibbs sampler, Metropolis–Hastings, HMC, NUTS).
- Hierarchical/multilevel models (random-effects across cost centres).
- Nonparametric Bayesian methods (Dirichlet process, Gaussian process
  regression).
- Variational inference and Bayesian neural networks.
- Real, identifiable company data: every numerical example uses
  fabricated parameters chosen to match plausible orders of magnitude
  for a 50-person IT team.

The justification for the anti-scope: the article's thesis is that
**closed-form Bayesian updating is enough** to explain and improve the
FYF cycle. Adding numerical inference would dilute that claim, distract
the reader, and add complexity that the budget problem does not require.

## Abstract (one paragraph)

Periodic budget revisions — the Forecast Year-end Financial (FYF) cycle —
are usually treated as a sequence of independent re-estimates: the analyst
walks into each review with a clean slate, looks at year-to-date actuals,
and produces a new forecast. This article argues that the FYF cycle is in
fact sequential Bayesian updating, and shows that recognising it as such
turns three pain points into closed-form results. (i) "How do I weight
the budget plan against the data I now have?" is answered by the
precision-weighted mean of the Normal–Normal posterior; the weights are
not a matter of judgement but a function of stated uncertainties. (ii)
"How tight is my forecast?" is answered by the monotonic shrinkage of
the posterior variance: $\sigma_n^2 \to 0$ as $n \to \infty$, with an
explicit rate. (iii) "What is the probability of ending the year over
budget?" is answered by the posterior predictive distribution applied to
the remaining months. The article derives Bayes' theorem for continuous
parameters, develops the four conjugate families that cover the FYF cost
model (Normal–Normal, Normal–Inverse-Gamma, Gamma–Poisson, Beta–Binomial),
proves the sequential equivalence theorem, and applies the resulting
machinery to a full annual cycle of a 50-person IT headcount budget. The
result is a budget that learns: each month of actuals tightens the
forecast, each quarterly review is a posterior, and the precision-weighted
mean replaces ad hoc reweighting.

## Connection to the Series

This is article 4 of a four-part series on probabilistic methods for
budget analytics. Articles 1–3 deferred FYF as motivation only; this
article develops it fully, using the distributions from article 2 as
priors, the headcount dynamics from article 3 to model evolving team
size, and the Monte Carlo machinery from article 1 for posterior
predictive sampling.
