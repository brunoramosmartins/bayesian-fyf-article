# FYF Model Design

**Status:** v0.1 (Phase 0 specification). Mathematical derivations and
proofs live in `notes/phaseN-*.md`; this document fixes the **model
choices** that those derivations specialise.

---

## 1. The Revision Cycle

A typical IT headcount budget follows this calendar:

| Month               | Event                          | Bayesian analogue                    |
|---------------------|--------------------------------|--------------------------------------|
| December (year N−1) | Budget plan approved           | Prior $\pi(\theta)$                  |
| January–March       | Q1 actuals arrive              | Likelihood $L(\theta\mid x_{1:3})$   |
| **April**           | **FYF #1 (Q1 review)**         | Posterior #1 $\pi(\theta\mid x_{1:3})$ |
| April–June          | Q2 actuals arrive              | New likelihood                       |
| **July**            | **FYF #2 (mid-year review)**   | Posterior #2 $\pi(\theta\mid x_{1:6})$ |
| July–September      | Q3 actuals arrive              | New likelihood                       |
| **October**         | **FYF #3 (Q3 review)**         | Posterior #3 $\pi(\theta\mid x_{1:9})$ |
| October–December    | Q4 actuals arrive              | Final data                           |
| January (year N+1)  | Year-end close                 | Posterior #4 $\pi(\theta\mid x_{1:12})$ |

Each FYF is a posterior. The previous posterior becomes the next prior;
this is what makes the cycle a sequential Bayesian update rather than a
sequence of fresh estimates.

## 2. Cost Decomposition

The monthly cost $X_t$ for month $t$ is modelled as the sum of three
components:

$$
X_t \;=\; \underbrace{n_t \cdot \bar S_t \cdot \beta}_{\text{salary + benefits}}
       \;+\; \underbrace{C_{\text{ot},t}}_{\text{overtime}}
       \;+\; \underbrace{C_{\text{inc},t}}_{\text{incident-related}} .
$$

Definitions and units (all monetary values in BRL):

- $n_t$: headcount at month $t$ (integer; 50 by default for the
  reference scenario).
- $\bar S_t$: average gross monthly salary at month $t$.
- $\beta$: benefit-and-charge multiplier (≈ 1.75; covers payroll taxes,
  social charges, benefits).
- $C_{\text{ot},t}$: overtime cost in month $t$.
- $C_{\text{inc},t}$: incident-related cost in month $t$ (on-call,
  emergency hires, vendor escalations).

For the **base Bayesian model** developed in Phases 1–4 the unknown of
interest is the mean monthly cost $\theta$, and we model

$$
X_t \mid \theta \sim N(\theta, \sigma^2_{\text{known}}).
$$

This is intentionally simple: it isolates the Bayesian machinery so the
reader sees the precision-weighted mean and the shrinkage rate before
extra moving parts are introduced. Extensions to unknown $\sigma^2$,
Poisson incident counts, and Binomial overtime proportions reuse the
same machinery and are introduced in Phase 5.

## 3. Conjugate Pairs Used

The article works with exactly four conjugate pairs. Each is fully
derived in Phase 2 (`notes/phase2-conjugate-families.md`). They cover
every component of the cost model.

| Component                          | Likelihood                       | Prior                                    | Posterior                                |
|------------------------------------|----------------------------------|------------------------------------------|------------------------------------------|
| Mean monthly cost ($\sigma^2$ known) | $N(\theta, \sigma^2/n)$          | $N(\mu_0, \sigma_0^2)$                   | $N(\mu_n, \sigma_n^2)$                   |
| Mean monthly cost ($\sigma^2$ unknown) | $N(\theta, \sigma^2)$           | $N\text{-}IG(\mu_0,\kappa_0,\alpha_0,\beta_0)$ | $N\text{-}IG(\mu_n,\kappa_n,\alpha_n,\beta_n)$ |
| Incident count per month           | $\text{Poisson}(\lambda)$        | $\text{Gamma}(\alpha_0, \beta_0)$        | $\text{Gamma}(\alpha_n, \beta_n)$        |
| Overtime proportion                | $\text{Binomial}(n, p)$          | $\text{Beta}(\alpha_0, \beta_0)$         | $\text{Beta}(\alpha_n, \beta_n)$         |

The central closed-form result, used throughout, is the
**Normal–Normal precision-weighted mean** for the case
$\sigma^2$ known with $n$ observations $\bar x$:

$$
\mu_n \;=\; \frac{\sigma^2 \,\mu_0 \;+\; n\sigma_0^2\,\bar x}{\sigma^2 + n\sigma_0^2},
\qquad
\sigma_n^2 \;=\; \frac{\sigma^2 \sigma_0^2}{\sigma^2 + n\sigma_0^2}.
$$

Equivalently, in **precision form** with $\tau \equiv 1/\sigma^2$ and
$\tau_0 \equiv 1/\sigma_0^2$:

$$
\tau_n \;=\; \tau_0 + n\tau,
\qquad
\mu_n \;=\; \frac{\tau_0 \mu_0 + n\tau \bar x}{\tau_n}.
$$

The posterior precision is the sum of prior precision and data
precision. The posterior mean is the precision-weighted average of the
budget plan and the empirical mean. This identity is the article's
recurring motif.

## 4. Default Parameters

The reference scenario is a 50-person IT team. All parameters are
fabricated (not real company data) and chosen for plausible
order-of-magnitude.

| Parameter                       | Value                  | Rationale                                                         |
|---------------------------------|------------------------|-------------------------------------------------------------------|
| Budget plan (prior mean) $\mu_0$ | R$ 1{,}050{,}000 / month | $50 \times \text{R\$ 12{,}000} \times 1.75$.                       |
| Prior uncertainty $\sigma_0$     | R$ 150{,}000           | Planner is ≈ 90 % sure the truth lies within ±15 % of the plan.    |
| Observation noise $\sigma$       | R$ 80{,}000            | Month-to-month variability observed in actuals.                   |
| Incident-rate prior              | $\text{Gamma}(3, 1)$   | Prior mean 3 incidents/month, prior s.d. $\sqrt{3}\approx 1.7$.   |
| Overtime-proportion prior        | $\text{Beta}(2, 8)$    | Prior mean 0.20 (20 % of the team does overtime in a given month).|

Translating "planner is 90 % sure the truth is within ±15 %" into
$\sigma_0$:

$$
\mu_0 \pm 1.645\,\sigma_0 \;=\; \mu_0 \,(1 \pm 0.15)
\;\Longrightarrow\;
\sigma_0 \;=\; \frac{0.15 \,\mu_0}{1.645} \;\approx\; \text{R\$ 95{,}744}.
$$

The roadmap rounds this to R$ 150{,}000 for round numbers and a slightly
weaker prior; both choices are valid and the prior-sensitivity experiment
in Phase 6 will quantify the difference. We adopt **R$ 150,000** as the
default to match the roadmap.

## 5. Worked Example — Prior + 3 Months → Posterior

Setup: $\mu_0 = 1{,}050{,}000$, $\sigma_0 = 150{,}000$,
$\sigma = 80{,}000$. Q1 actuals are

$$
x_1 = 1{,}120{,}000, \quad
x_2 = 1{,}080{,}000, \quad
x_3 = 1{,}095{,}000.
$$

Sample size $n = 3$, sample mean $\bar x = 1{,}098{,}333.\overline{3}$.

**Posterior variance** (precision form):

$$
\tau_0 = 1/150{,}000^2 \approx 4.444\times 10^{-11},
\quad
\tau   = 1/80{,}000^2  \approx 1.5625\times 10^{-10},
$$

$$
\tau_3 = \tau_0 + 3\tau \approx 5.131\times 10^{-10},
\qquad
\sigma_3 = 1/\sqrt{\tau_3} \approx \text{R\$ 44{,}131}.
$$

**Posterior mean**:

$$
\mu_3 \;=\;
\frac{\tau_0 \mu_0 + 3\tau\,\bar x}{\tau_3}
\;\approx\;
\frac{4.667\times 10^{-5} + 5.149\times 10^{-4}}{5.131\times 10^{-10}}
\;\approx\;
\text{R\$ 1{,}094{,}163}.
$$

**Numerical check (closed form)**:

$$
\mu_3 \;=\;
\frac{\sigma^2 \mu_0 + 3\sigma_0^2 \bar x}{\sigma^2 + 3\sigma_0^2}
\;=\;
\frac{6.4\times 10^9 \cdot 1.05\times 10^6 \;+\; 3 \cdot 2.25\times 10^{10} \cdot 1.0983\overline{3}\times 10^6}
     {6.4\times 10^9 + 6.75\times 10^{10}}
\;\approx\; \text{R\$ 1{,}094{,}163},
$$

$$
\sigma_3^2 \;=\;
\frac{\sigma^2\sigma_0^2}{\sigma^2 + 3\sigma_0^2}
\;=\;
\frac{6.4\times 10^9 \cdot 2.25\times 10^{10}}{7.39\times 10^{10}}
\;\approx\; 1.948\times 10^9
\;\Longrightarrow\;
\sigma_3 \approx \text{R\$ 44{,}131}.
$$

Interpretation:

- Prior mean R$ 1.050M, prior s.d. R$ 150K.
- After Q1 (3 actuals slightly above plan), posterior mean shifts to
  **R$ 1.094M** (+R$ 44K, roughly half-way to $\bar x$ because the prior
  is not negligible).
- Posterior s.d. tightens from **R$ 150K → R$ 44K** — a
  ≈ 71 % reduction in uncertainty after just one quarter of data.

The shrinkage weight on the prior is

$$
w_0 \;=\; \frac{\tau_0}{\tau_0 + 3\tau}
\;=\;\frac{4.444\times10^{-11}}{5.131\times10^{-10}}
\;\approx\; 0.087,
$$

so the data carries ≈ 91 % of the weight after Q1.

## 6. Key Questions the Article Answers

The article uses the FYF model to answer five business-relevant
questions. Each maps cleanly onto a Bayesian quantity.

1. **Shrinkage.** How much does each month of actuals pull the forecast
   away from the plan? — Answered by the trajectory of the posterior
   mean $\mu_n$ as $n$ grows (Phase 3).
2. **Precision.** How tight is the 95 % credible interval at each FYF? —
   Answered by $\sigma_n$ at $n=3, 6, 9, 12$ (Phase 3).
3. **Surprise detection.** When should we override the Bayesian update
   and question the model? — Answered by posterior predictive checks
   (Phase 5).
4. **Prior sensitivity.** How much does the prior choice matter after 6
   months of data? — Answered by overlaying $\mu_n, \sigma_n$ for two
   priors (Phase 3 and Phase 6).
5. **Predictive accuracy.** What is the probability of ending the year
   over budget? — Answered by the year-end posterior predictive
   distribution (Phase 4).

## 7. Pointers to Subsequent Phases

- **Phase 1** — `notes/phase1-bayes-foundations.md`: derivation of
  Bayes' theorem for continuous $\theta$, MAP vs posterior mean,
  credible intervals, prior elicitation.
- **Phase 2** — `notes/phase2-conjugate-families.md`: full derivation of
  the four conjugate pairs.
- **Phase 3** — `notes/phase3-sequential-updating.md`: sequential = batch
  proof, shrinkage rate, prior–data conflict.
- **Phase 4** — `notes/phase4-predictive-inference.md`: posterior
  predictive, year-end forecast, Bayes factors.
- **Phase 5** — `notes/phase5-fyf-model.md`: full annual simulation
  across the five scenarios.
