# Exercises — Phase 5 (Applied FYF Model)

> Pencil-and-paper exercises. Numerical answers can be cross-checked
> against `src/fyf_model.py` and `src/diagnostics.py` once the modules
> are in place.

---

## Computations

### 1. Walking through Q1 month by month

Reference parameters: $\mu_0 = 1{,}050{,}000$, $\sigma_0 = 150{,}000$,
$\sigma = 80{,}000$. Q1 actuals
$\{\text{R\$ 1{,}120K},\;\text{R\$ 1{,}095K},\;\text{R\$ 1{,}080K}\}$.

For each month $n = 1, 2, 3$, compute and tabulate:

| $n$ | $\mu_n$ | $\sigma_n$ | 95 % CI for $\theta$ | $w_0(n)$ | surprise $z_n$ |
|----:|--------:|-----------:|----------------------|---------:|----------------|
|  1  |         |            |                      |          |                |
|  2  |         |            |                      |          |                |
|  3  |         |            |                      |          |                |

The surprise z-score uses the predictive at *the previous step*:
$z_t = (x_t - \mu_{t-1})/\sqrt{\sigma_{t-1}^2 + \sigma^2}$. Use
$\mu_0, \sigma_0$ for $t = 1$.

*Approximate expected values (verify):*
$\mu_3 \approx \text{R\$ 1{,}094{,}163}$,
$\sigma_3 \approx \text{R\$ 44{,}131}$,
$w_0(3) \approx 0.087$. All three $|z_t| < 1$, so no surprise.

---

### 2. Q1 FYF year-end forecast

Continuing from exercise 1: at the Q1 FYF (end of month 3) the
posterior is $N(\mu_3, \sigma_3^2)$ from above. Observed cumulative
total $S_3 = \text{R\$ 3{,}295{,}000}$. Annual budget
$B = \text{R\$ 13{,}200{,}000}$.

(a) Compute the year-end predictive total
$T \mid x_{1:3} \sim N(\mu_T, \sigma_T^2)$ with the **correct**
quadratic-h variance from Phase 4 §5.
(b) Compute $P(T > B)$.
(c) Compare with the naïve iid formula and report both probabilities
side by side. By how many percentage points does the naïve formula
under-estimate the over-budget risk?

*Expected (correct):* $\mu_T \approx \text{R\$ 13{,}142{,}467}$,
$\sigma_T \approx \text{R\$ 401{,}480}$, $P(T > B) \approx 0.443$.
*Expected (naïve):* $\sigma_T \approx \text{R\$ 247{,}725}$,
$P(T > B) \approx 0.405$. Naïve under-estimates by ≈ 4 percentage
points.

---

### 3. Budget shock detection

Same prior. Months 1–4 actuals
$\{1{,}055\text{K},\;1{,}042\text{K},\;1{,}068\text{K},\;1{,}049\text{K}\}$.
Month 5 actual: $x_5 = \text{R\$ 1{,}400{,}000}$.

(a) Compute the posterior $(\mu_4, \sigma_4)$ at the end of month 4.
(b) Compute the surprise z-score $z_5$ at the start of month 5,
*before* consuming $x_5$.
(c) Apply the diagnostic from `notes/phase5-fyf-model.md` §3.5: does
$|z_5| > 3$ trigger? What does the rule say to do?
(d) Compute the posterior $(\mu_5, \sigma_5)$ that the conjugate
update would produce *if accepted*. Comment on whether trusting the
update is appropriate when the diagnostic has fired.

*Expected:* $z_5 \gg 3$ (the shock is many predictive σ above the
forecast), the diagnostic triggers, and the analyst should
investigate the cause before mechanically accepting the update — see
the operational rule in §3.3 of `notes/phase3-sequential-updating.md`.

---

### 4. Prior sensitivity at month 6

Two analysts, different prior confidence:
- Analyst A: $\sigma_0^{(A)} = 100{,}000$ (very confident).
- Analyst B: $\sigma_0^{(B)} = 300{,}000$ (uncertain).

Both share $\mu_0 = 1{,}050{,}000$ and $\sigma = 80{,}000$. Six months
of identical actuals with $\bar x_6 = 1{,}085{,}000$.

(a) Compute the posterior $\mu_6$ for each analyst.
(b) Compute the gap $|\mu_6^{(A)} - \mu_6^{(B)}|$.
(c) Find the smallest $n$ for which the gap drops below R$ 10{,}000.
*Hint*: from Phase 3 §3.1, the gap between two posterior means with
*different* $\sigma_0$ values does **not** factor as cleanly as the
same-$\sigma_0$ case; you have to compute each posterior mean
explicitly and take the difference.

*Approximate:* at $n=6$ the gap is on the order of R$ 10{,}000–15{,}000$;
work it out exactly.

---

### 5. Calibration test from historical FYFs

Over the last 5 years, the team has produced 60 monthly forecasts
(12 months × 5 years). Of these, 51 actuals fell inside the 95 %
posterior predictive interval.

(a) Under the null hypothesis "the model is well calibrated", what is
the distribution of the count?
(b) Compute the expected count and the binomial probability
$\Pr(X \le 51 \mid X \sim \text{Binomial}(60, 0.95))$.
(c) Conclude whether the model is well-calibrated, under-confident,
or over-confident. *Hint:* the expected count is 57; observing only
51 suggests the predictive intervals are too narrow.
(d) What fix would you suggest? *Two options to consider:* (i) raise
$\sigma$ (more sampling-noise allowance); (ii) raise $\sigma_0$ (more
prior uncertainty). Which one helps when the credibility-interval
under-coverage persists into the steady-state, large-$n$ regime?

*Expected:* $\Pr(X \le 51) \approx 0.024$; the model is moderately
under-confident at the 5 % level. Because under-coverage persists at
large $n$ (when the prior contribution $\sigma_n^2 \to 0$), the fix
must be on $\sigma$, not $\sigma_0$ — the sampling model is too
narrow.
