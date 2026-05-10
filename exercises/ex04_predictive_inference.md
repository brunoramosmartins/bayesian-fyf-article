# Exercises — Phase 4 (Predictive Inference)

> Pencil-and-paper exercises. Numerical answers can be cross-checked
> against `src/predictive.py` once the module is in place.

---

## A. Proofs

### 1. Normal-Normal posterior predictive

Show that, for the Normal-Normal model with posterior
$\theta \mid x \sim N(\mu_n, \sigma_n^2)$ and likelihood
$\tilde x \mid \theta \sim N(\theta, \sigma^2)$, the posterior
predictive is

$$
\tilde x \mid x \;\sim\; N\!\big(\mu_n,\; \sigma_n^2 + \sigma^2\big).
$$

(a) Decompose $\tilde x = \theta + \varepsilon$ with
$\varepsilon \sim N(0, \sigma^2)$ independent of $\theta$ given $x$;
use additivity of independent Normals.
(b) **Or** carry out the convolution
$\int N(\tilde x \mid \theta, \sigma^2)\,N(\theta \mid \mu_n, \sigma_n^2)\,\mathrm d\theta$
by completing the square in $\theta$. Show every step.

---

### 2. Law of total variance for the predictive

Prove the decomposition

$$
\mathrm{Var}(\tilde x \mid x)
\;=\;
\underbrace{\mathbb E\!\big[\mathrm{Var}(\tilde x \mid \theta) \,\big|\, x\big]}_{\text{expected sampling noise}}
\;+\;
\underbrace{\mathrm{Var}\!\big(\mathbb E[\tilde x \mid \theta] \,\big|\, x\big)}_{\text{parameter uncertainty}}.
$$

(a) Start from $\mathrm{Var}(\tilde x \mid x)
= \mathbb E[(\tilde x - \mathbb E[\tilde x \mid x])^2 \mid x]$.
(b) Add and subtract $\mathbb E[\tilde x \mid \theta]$ inside the
square; expand.
(c) Note that the cross term vanishes (use the tower property
$\mathbb E[\,\cdot \mid x\,] = \mathbb E[\,\mathbb E[\cdot \mid \theta]\mid x\,]$).
(d) Specialise to Normal-Normal: confirm the first term is $\sigma^2$
(constant) and the second is $\sigma_n^2$ (shrinks with $n$).

---

### 3. Gamma-Poisson predictive is Negative Binomial

Compute

$$
p(\tilde x = k \mid x)
\;=\;
\int_0^\infty \frac{\lambda^k e^{-\lambda}}{k!}
\cdot
\frac{\beta_n^{\alpha_n}}{\Gamma(\alpha_n)}\,\lambda^{\alpha_n - 1}\,e^{-\beta_n \lambda}\,
\mathrm d\lambda.
$$

(a) Group powers of $\lambda$ and exponentials; identify the
remaining integral as a Gamma normalising constant.
(b) Conclude
$\tilde x \mid x \sim \text{NegBin}\big(\alpha_n,\; \beta_n/(\beta_n+1)\big)$
(shape–success parameterisation).
(c) Compute the mean ($\alpha_n/\beta_n$) and variance
($\alpha_n(\beta_n+1)/\beta_n^2$). Verify the variance decomposition
from exercise 2 explicitly:
$\mathrm{Var}(\tilde x \mid x) = \mathbb E[\lambda \mid x] + \mathrm{Var}(\lambda \mid x)$.

---

### 4. Year-end predictive total — beware of independence

After observing $m$ months with cumulative sum $S_m$, the remaining
total $\tilde S = \sum_{t=m+1}^{12} \tilde x_t$ is not the sum of iid
predictives — the future months share $\theta$.

(a) Write $\tilde x_t = \theta + \varepsilon_t$ with
$\varepsilon_t$ iid $N(0, \sigma^2)$ independent of $\theta$. Sum to
get
$\tilde S = (12-m)\theta + \sum_{t=m+1}^{12}\varepsilon_t$.
(b) Use independence of the two summands given $x_{1:m}$ to derive

$$
\tilde S \mid x_{1:m}
\;\sim\;
N\!\big((12-m)\,\mu_m,\;\; (12-m)^2\,\sigma_m^2 + (12-m)\,\sigma^2\big).
$$

(c) Compare with the **incorrect** "iid future months" formula
$N\big((12-m)\mu_m,\;(12-m)(\sigma_m^2 + \sigma^2)\big)$. Show that
the parameter-uncertainty term is under-estimated by a factor of
$(12-m)$. Explain why this matters operationally for FYF: in early
months the horizon is long and the under-estimate is large.

(d) Derive the closed form for $P(T > B \mid x_{1:m})$ where
$T = S_m + \tilde S$ is the annual total.

---

### 5. Bayes factor for two Normal priors

Two competing Normal-Normal models share the sampling density
$x_i \mid \theta \sim N(\theta, \sigma^2)$ but disagree on the prior:
$M_A : \theta \sim N(\mu_A, \sigma_A^2)$ and
$M_B : \theta \sim N(\mu_B, \sigma_B^2)$.

(a) Compute the marginal likelihood
$p(x_{1:n} \mid M) = \int L(\theta) \pi(\theta;\mu, \sigma^2)\,\mathrm d\theta$
for each model in closed form. *Hint: exploit conjugacy — the integral
is just the ratio of normalising constants between prior and posterior.*
(b) Form $BF_{AB} = p(x \mid M_A)/p(x \mid M_B)$.
(c) State the Jeffreys scale (anecdotal / substantial / strong / very
strong / decisive) and place the BF you computed on it for the
exercise-7 numbers.

---

## B. Computations

### 6. Predictive interval after 6 months

Posterior $\theta \mid x_{1:6} \sim N(1{,}085{,}000,\; 32{,}000^2)$.
Sampling noise $\sigma = 80{,}000$.

(a) Compute the posterior predictive distribution for $\tilde x_7$.
(b) Compute the 95 % equal-tailed predictive interval.
(c) Compute the 95 % credible interval for $\theta$ from the
posterior. What is the ratio of the predictive width to the credible
width? *Expected: ≈ 2.7×.*

*Expected:*
predictive $\sim N(1{,}085{,}000, \;7.424\!\times\!10^9)$,
$\sigma_{\text{pred}} \approx \text{R\$ 86{,}162}$,
predictive 95 % interval $\approx [\,916{,}123,\;1{,}253{,}877\,]$,
credible 95 % interval $\approx [\,1{,}022{,}280,\;1{,}147{,}720\,]$.

---

### 7. Probability of exceeding annual budget

Same posterior as exercise 6. Months 1–6 observed total
$S_6 = \text{R\$ 6{,}510{,}000}$. Annual budget
$B = \text{R\$ 13{,}200{,}000}$.

(a) Compute the predictive distribution of the year-end total
$T = S_6 + \tilde S$ using the **correct** formula from exercise 4.
(b) Compute $P(T > B)$.
(c) Repeat with the **incorrect** iid formula. By how many percentage
points does the probability change?

*Expected (correct):*
$T \sim N(13{,}020{,}000,\;7.526\!\times\!10^{10})$,
$\sigma_T \approx \text{R\$ 274{,}343}$,
$z \approx 0.656$, $P(T > B) \approx 0.256$.

*Expected (naïve):*
$\sigma_T \approx \text{R\$ 211{,}055}$,
$z \approx 0.853$, $P(T > B) \approx 0.197$.

The naïve formula understates the over-budget risk by ≈ 6 percentage
points.

---

### 8. (Optional) Negative Binomial predictive for incidents

Posterior $\lambda \mid x \sim \text{Gamma}(20, 7)$ (from Phase 2
exercise 8).

(a) Identify the predictive distribution for next month's incident
count $\tilde x$.
(b) Compute the predictive mean, variance, and 95 % equal-tailed
predictive interval (smallest $[a, b]$ such that $\Pr(a \le \tilde x \le b) \ge 0.95$).
(c) Compare the predictive variance with the Poisson-only variance
that would result from plugging in the posterior mean as a fixed
$\lambda$. The difference is the **overdispersion** induced by
parameter uncertainty.
