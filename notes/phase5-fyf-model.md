# Phase 5 — Applied FYF Model and Scenarios

> Working notes for the article. Phases 1–4 built the machine; Phase 5
> drives it. We assemble the conjugate updaters (Phase 2), the
> sequential engine (Phase 3), and the predictive distributions
> (Phase 4) into a complete annual FYF cycle, simulate five canonical
> scenarios, and add the diagnostic layer that decides when the model
> is and is not to be trusted. Notation follows `docs/outline.md`.

---

## 1. The FYF Model — One Object, Three Capabilities

### 1.1 Specification

The `FYFModel` couples three components into a single stateful object:

1. **Sequential Bayesian engine.** A `SequentialUpdater` wrapping a
   `NormalNormalUpdater` with hyperparameters $(\mu_0, \sigma_0^2, \sigma^2)$.
   This is the inference layer derived in Phases 2–3.
2. **Year-end predictive forecast.** Given the current posterior at
   month $m$ and the cumulative observed total, returns the predictive
   distribution of the annual total $T = S_m + \tilde S$ via
   `year_end_predictive_total` (Phase 4 §5).
3. **Surprise diagnostic.** Before consuming a new monthly actual,
   computes the predictive z-score
   $z_t = (x_t - \mu_{t-1})/\sqrt{\sigma_{t-1}^2 + \sigma^2}$. After
   the update it logs the new posterior alongside $z_t$.

The state is a list of `MonthlyReview` records: month, actual,
posterior after the update, surprise z, year-end forecast, and
$P(T > B)$ if a budget ceiling $B$ is set.

### 1.2 Three operations the article uses

- **`process_month(actual)`** — feed one observation, return the
  monthly review.
- **`fyf_review(quarter)`** — at the end of quarters 1–4 (months 3,
  6, 9, 12), produce a quarterly summary tying the posterior, the
  year-end forecast, and an actionable interpretation of $P(T > B)$.
- **`annual_cycle(monthly_actuals)`** — bulk-feed 12 months and
  return the full trajectory. Used by the scenario simulators.

The model deliberately keeps the cost decomposition simple — the
Normal-Normal layer with known $\sigma$. The Gamma-Poisson incidents
and Beta-Binomial overtime extensions appear in `docs/model-design.md`
and are wired in conceptually but not used in the canonical scenarios.

---

## 2. Five Canonical Scenarios

Each scenario shares the reference parameters
$(\mu_0, \sigma_0, \sigma, B) = (1{,}050{,}000;\;150{,}000;\;80{,}000;\;13{,}200{,}000)$
and differs only in how the 12 monthly actuals are generated. All
random draws use a fixed seed (one per scenario) so the figures and
tests are reproducible.

### S1 — On-target

$x_t \sim N(\mu_0, \sigma^2)$ for $t = 1,\ldots,12$. The plan is
correct in expectation; the posterior should converge to $\mu_0$ with
shrinking $\sigma_n$ and $P(T > B)$ should sit around 50 %.

### S2 — Optimistic plan

$x_t \sim N(\mu_0 + \delta, \sigma^2)$ with $\delta = +\text{R\$ 50{,}000}$.
The plan systematically underestimates costs. The posterior corrects
upward; by Q2 the year-end forecast should comfortably exceed $\mu_0 \cdot 12$.
$P(T > B)$ rises sharply between Q1 and Q2.

### S3 — Budget shock

$x_t \sim N(\mu_0, \sigma^2)$ for $t = 1, \ldots, 4$; $x_5 = \text{R\$ 1{,}400{,}000}$
(a +4σ event); $x_t \sim N(\mu_0 + 50{,}000, \sigma^2)$ for $t = 6, \ldots, 12$
(a permanent step up after the shock). The point of S3 is the
**diagnostic**: the surprise $z_5$ should clear $|z| > 3$, alerting the
analyst that the conjugate update may be mechanical-but-wrong (the
data-generating process changed).

### S4 — Seasonal variation

$x_t = \mu_0 + s_t + \varepsilon_t$ with a fixed seasonal pattern
$s_t$ (Q3 below baseline, Q4 above) and iid $N(0, \sigma^2)$ noise.
The Normal-Normal model assumes a single $\theta$, so it cannot
separate the seasonal signal from the noise. The posterior settles
near the *annual* mean and $\sigma_n$ shrinks, but the surprise
sequence shows a periodic pattern — a hint that a more elaborate
model (Phase 8 future work) would do better.

### S5 — Prior sensitivity

Two analysts, identical data: Analyst A holds
$\sigma_0^{(A)} = 100{,}000$ (very confident plan), Analyst B
$\sigma_0^{(B)} = 300{,}000$ (uncertain plan). Both consume the same
S1-style actuals. The Phase 3 §3 result predicts the gap shrinks at
rate $w_0(n)$ tied to the analyst's own $\sigma_0$; we plot both
trajectories and the gap.

---

## 3. The Diagnostic Layer

### 3.1 Surprise score

$$
z_t \;=\; \frac{x_t - \mu_{t-1}}{\sqrt{\sigma_{t-1}^2 + \sigma^2}}.
$$

This is the **standardised innovation** under the posterior predictive
$\tilde x_t \mid x_{1:t-1} \sim N(\mu_{t-1}, \sigma_{t-1}^2 + \sigma^2)$.
Under a correctly specified model
$z_t \mid x_{1:t-1} \sim N(0, 1)$, so the heuristic
$|z_t| > 2$ is "uncomfortable" and $|z_t| > 3$ is "investigate". The
surprise scores are NOT iid across $t$ in finite samples (the
posterior at $t-1$ is itself random), but they are exchangeable under
correct specification; the diagnostic is robust enough for routine
practice.

### 3.2 Posterior predictive p-value

A two-sided p-value:

$$
p_t \;=\; 2 \cdot \min\!\Big(\Phi(z_t),\; 1 - \Phi(z_t)\Big).
$$

Aggregated over the year, the empirical distribution of
$\{p_t\}_{t=1}^{12}$ should be approximately Uniform on $[0, 1]$. A
qq-plot vs Uniform is the standard visual.

### 3.3 Calibration

A 95 % equal-tailed predictive interval should contain the next
actual approximately 95 % of the time. **Calibration score** over $T$
months and $K$ years:

$$
\widehat C \;=\; \frac{1}{TK}\sum_{i,t}\,\mathbb 1\{x_{i,t} \in \text{PI}_{0.95}(x_{i,t-1})\}.
$$

If the model is well calibrated $\widehat C \approx 0.95$. Departures
flag systematic over- or under-confidence. A simple binomial test
quantifies whether the gap is significant.

### 3.4 Cumulative surprise

$$
S_n \;=\; \sum_{t=1}^n z_t.
$$

A random walk with mean zero and variance $\approx n$ under the
model. A persistent drift (e.g. $S_n / \sqrt n$ leaving $[-2, 2]$)
flags **structural drift** — the analyst's sampling model has gone
stale.

### 3.5 When the diagnostic fires, what to do

The article distinguishes three regimes:

| Diagnostic                              | Action                                                           |
|-----------------------------------------|------------------------------------------------------------------|
| Single $\lvert z_t\rvert > 3$           | One-off shock. Trust the update; flag for narrative.             |
| Repeated $\lvert z_t\rvert > 2$ same sign | Drift: re-elicit prior or change model.                          |
| Calibration $\widehat C \le 0.85$       | Underconfident model: $\sigma$ too small or model too rigid.    |
| Calibration $\widehat C \ge 0.99$       | Over-cautious: $\sigma$ too large; intervals are uninformative. |

---

## 4. Quarterly FYF Reviews

At months 3, 6, 9, 12 the model emits a `QuarterlyReview` with:

- **Posterior** $N(\mu_n, \sigma_n^2)$ (the parameter inference).
- **Year-end forecast**: predictive distribution of $T$, with mean
  $S_n + (12-n)\mu_n$ and standard deviation
  $\sqrt{(12-n)^2 \sigma_n^2 + (12-n)\sigma^2}$ (Phase 4 §5).
- **$P(T > B)$**: probability of exceeding the budget ceiling.
- **Surprise summary**: $\max_t |z_t|$ in the quarter, count of
  surprises with $|z| > 2$.
- **Recommendation**: a one-line action ("hold", "re-elicit",
  "investigate shock", "request budget revision").

The recommendations are heuristic, not part of the formal model; they
exist to make the article concrete for the practising analyst.

---

## 5. What Phase 5 Does Not Address

The Normal-Normal model treats every monthly cost as
$x_t = \theta + \varepsilon_t$ with $\theta$ fixed. This is correct
under structural stability and clearly wrong otherwise:

- **Trend / drift**: a slow upward trend in salaries.
- **Regime change**: a re-org changes the team structure mid-year.
- **Seasonality**: S4 in §2 above; structurally periodic
  $\theta_t$.

The article's stance is honest: the simple model is enough to cover
80 % of practice, the diagnostic layer flags the other 20 %, and a
hierarchical or state-space generalisation is the natural next
article. We *cite* state-space models (Kalman filtering, Phase 3
recursion above) as the next step but do not build one here.

---

## 6. Summary

- The `FYFModel` is the sequential Bayesian engine plus the year-end
  predictive plus a surprise diagnostic, packaged as a single
  stateful object.
- Five canonical scenarios — on-target, optimistic, shock, seasonal,
  prior sensitivity — exercise the model end-to-end and produce the
  article's centrepiece figures.
- Diagnostics: surprise z-scores, posterior predictive p-values,
  calibration coverage, cumulative drift. They turn the model from
  "always answers" to "answers, and tells you when its answer should
  not be trusted".
- Out-of-scope structural extensions (trend, regime change,
  hierarchical pooling, state space) are noted as future work.

Phase 6 runs the experiments end-to-end and produces all
publication figures. Phase 7 writes the article.

## References

- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed., Ch. 6
  (model checking) and Ch. 7 (model comparison).
- West, M. & Harrison, J. (1997). *Bayesian Forecasting and Dynamic
  Models*. Springer (state-space generalisations cited as future
  work).
- Box, G. E. P. (1980). "Sampling and Bayes' inference in scientific
  modelling and robustness". *J. R. Statist. Soc. A* 143, 383–430
  (the foundational "all models are wrong" diagnostic paper).
