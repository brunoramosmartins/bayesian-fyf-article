# Phase 4 — Posterior Predictive Inference

> Working notes for the article. The posterior $\pi(\theta \mid x)$ is a
> means; the posterior **predictive** $p(\tilde x \mid x)$ is the end —
> it is what answers "what will next month / the year-end total cost
> be?". We derive the predictive in closed form for the three single-
> parameter conjugate pairs, work out the year-end total carefully
> (including the dependence between future months that share $\theta$),
> connect to Article 1 Monte Carlo, and define Bayes factors for model
> comparison. Notation follows `docs/outline.md`.

---

## 1. Definition and Two Reasons It Is the Right Object

### 1.1 Definition

The **posterior predictive distribution** of a future observation
$\tilde x$ given past data $x = x_{1:n}$ is

$$
\boxed{\quad
p(\tilde x \mid x)
\;=\;
\int_\Theta f(\tilde x \mid \theta)\,\pi(\theta \mid x)\,\mathrm d\theta.
\quad}
$$

The integral averages the sampling density $f(\tilde x \mid \theta)$
over the posterior. Equivalently, $\tilde x$ has the joint density
$f(\tilde x \mid \theta)\pi(\theta \mid x)$ and we marginalise out
$\theta$.

### 1.2 Two reasons this is what the business actually wants

**(a) The CFO does not bet on $\theta$ — they bet on observations.**
A budget committee asks "will we exceed R$ 13.2M?". That is a
statement about the year-end total, an observable, not about the
parameter. The posterior $\pi(\theta \mid x)$ has no answer; only the
predictive $p(\tilde x \mid x)$ does.

**(b) The predictive captures both kinds of uncertainty.** The
posterior credible interval reflects only **parameter uncertainty**.
Real-world prediction has to add **irreducible noise** — the
month-to-month variability captured by $\sigma$. The predictive
combines them via the **law of total variance**:

$$
\mathrm{Var}(\tilde x \mid x)
\;=\;
\underbrace{\mathbb E\!\big[\mathrm{Var}(\tilde x \mid \theta) \,\big|\, x\big]}_{\text{expected sampling noise}}
\;+\;
\underbrace{\mathrm{Var}\!\big(\mathbb E[\tilde x \mid \theta] \,\big|\, x\big)}_{\text{parameter uncertainty}}.
$$

For the FYF Normal-Normal model with $\sigma$ known, the first term
equals $\sigma^2$ exactly (a constant — the parameter-uncertainty share
shrinks with $n$, the noise share does not).

---

## 2. Normal-Normal Predictive

### 2.1 Closed form

**Setting.** Posterior $\theta \mid x \sim N(\mu_n, \sigma_n^2)$
(Phase 2). Sampling model $\tilde x \mid \theta \sim N(\theta, \sigma^2)$
with $\sigma^2$ known.

**Decompose** $\tilde x = \theta + \varepsilon$ with
$\varepsilon \sim N(0, \sigma^2)$ independent of $\theta$ given the data.
Then $\tilde x \mid x$ is the sum of two independent Normals:
$N(\mu_n, \sigma_n^2)$ and $N(0, \sigma^2)$. Sums of independent
Normals are Normal, with means and variances adding:

$$
\boxed{\quad
\tilde x \mid x_{1:n} \;\sim\; N\big(\mu_n,\; \sigma_n^2 + \sigma^2\big).
\quad}
$$

A direct integration is just as quick:
$\int N(\tilde x \mid \theta, \sigma^2)\, N(\theta \mid \mu_n, \sigma_n^2)\,\mathrm d\theta$
is a convolution of Normals, evaluated by completing the square —
yielding the same $N(\mu_n, \sigma_n^2 + \sigma^2)$.

### 2.2 Variance decomposition

$\mathrm{Var}(\tilde x \mid x) = \sigma_n^2 + \sigma^2$. The two terms
play different roles:

- $\sigma_n^2$ — *parameter uncertainty*. Decreases at rate $1/n$
  (Phase 3). Goes to $0$ as $n \to \infty$.
- $\sigma^2$ — *irreducible sampling noise*. Constant in $n$. **Does
  not vanish** with more data.

This is the formal sense in which "you cannot eliminate uncertainty,
only reduce the parameter share". A predictive 95 % interval is always
**wider** than the credible interval for $\theta$, and the gap closes
asymptotically toward $\pm 1.96\sigma$ — never toward zero.

### 2.3 FYF reading

Reference posterior after 6 months (from Phase 3):
$\theta \mid x_{1:6} \sim N(1{,}085{,}000, \approx 32{,}000^2)$,
$\sigma = 80{,}000$.

Predictive standard deviation:
$\sqrt{32{,}000^2 + 80{,}000^2} \approx \text{R\$ 86{,}162}$.

| Quantity                                        | 95 % interval (R$)             |
|-------------------------------------------------|--------------------------------|
| Credible interval for $\theta$ (given the data) | $[\,1{,}022{,}280,\; 1{,}147{,}720\,]$  |
| Predictive interval for $\tilde x_7$ (next month) | $[\,916{,}123,\; 1{,}253{,}877\,]$ |

The predictive band is ≈ 2.7× wider than the credible band. The
parameter is well-estimated; the future is still noisy.

---

## 3. Gamma-Poisson Predictive (Negative Binomial)

### 3.1 Closed form

**Setting.** Posterior $\lambda \mid x \sim \text{Gamma}(\alpha_n, \beta_n)$
in the rate parameterisation. Sampling model
$\tilde x \mid \lambda \sim \text{Poisson}(\lambda)$.

Compute the predictive PMF for $\tilde x = k \in \{0, 1, 2, \ldots\}$:

$$
p(\tilde x = k \mid x)
\;=\;
\int_0^\infty \frac{\lambda^k e^{-\lambda}}{k!}
\cdot
\frac{\beta_n^{\alpha_n}}{\Gamma(\alpha_n)}\,\lambda^{\alpha_n - 1}\,e^{-\beta_n \lambda}\,
\mathrm d\lambda.
$$

Pull out the constants and group powers of $\lambda$:

$$
=
\frac{\beta_n^{\alpha_n}}{\Gamma(\alpha_n)\,k!}
\int_0^\infty
\lambda^{\alpha_n + k - 1}\, e^{-(\beta_n + 1)\lambda}\,\mathrm d\lambda.
$$

The integrand is the kernel of $\text{Gamma}(\alpha_n + k, \beta_n + 1)$;
the integral evaluates to $\Gamma(\alpha_n + k)/(\beta_n + 1)^{\alpha_n + k}$.
Therefore

$$
p(\tilde x = k \mid x)
\;=\;
\frac{\Gamma(\alpha_n + k)}{\Gamma(\alpha_n)\,k!}
\Big(\frac{\beta_n}{\beta_n + 1}\Big)^{\alpha_n}
\Big(\frac{1}{\beta_n + 1}\Big)^{k}.
$$

This is the PMF of the **Negative Binomial distribution** in the
"shape $r$, success probability $p$" parameterisation, with
$r = \alpha_n$ and $p = \beta_n/(\beta_n + 1)$:

$$
\boxed{\quad
\tilde x \mid x \;\sim\; \text{NegBin}\!\big(r = \alpha_n,\; p = \beta_n/(\beta_n+1)\big).
\quad}
$$

### 3.2 Mean, variance, and overdispersion

For this parameterisation
$\mathbb E[\tilde x] = r(1-p)/p$ and $\mathrm{Var}(\tilde x) = r(1-p)/p^2$:

$$
\mathbb E[\tilde x \mid x] = \frac{\alpha_n}{\beta_n}, \qquad
\mathrm{Var}(\tilde x \mid x) = \frac{\alpha_n (\beta_n + 1)}{\beta_n^2}
\;>\;
\frac{\alpha_n}{\beta_n}.
$$

The predictive variance exceeds the mean — **overdispersion**, in
contrast to the Poisson-only model (where Var = mean). The extra
dispersion is exactly the parameter-uncertainty contribution from
$\lambda$ via the law of total variance:

$$
\mathrm{Var}(\tilde x \mid x)
= \underbrace{\mathbb E[\mathrm{Var}(\tilde x \mid \lambda)\mid x]}_{=\;\mathbb E[\lambda \mid x]\;=\;\alpha_n/\beta_n}
+ \underbrace{\mathrm{Var}(\mathbb E[\tilde x \mid \lambda]\mid x)}_{=\;\mathrm{Var}(\lambda\mid x)\;=\;\alpha_n/\beta_n^2}
= \frac{\alpha_n}{\beta_n} + \frac{\alpha_n}{\beta_n^2}
= \frac{\alpha_n(\beta_n+1)}{\beta_n^2}.
$$

---

## 4. Beta-Binomial Predictive

### 4.1 Closed form (single trial, then $m$ trials)

**Setting.** Posterior $p \mid x \sim \text{Beta}(\alpha_n, \beta_n)$.
Sampling model: $\tilde x$ counts successes in a future batch of size
$m$, $\tilde x \mid p \sim \text{Binomial}(m, p)$.

For $m = 1$ (a single Bernoulli), the predictive is Bernoulli with
success probability
$\Pr(\tilde x = 1 \mid x) = \mathbb E[p \mid x] = \alpha_n/(\alpha_n + \beta_n)$.

For general $m$, the predictive is the **Beta-Binomial distribution**:

$$
\boxed{\quad
\Pr(\tilde x = k \mid x)
=
\binom{m}{k}\,
\frac{B(\alpha_n + k,\; \beta_n + m - k)}{B(\alpha_n, \beta_n)},
\quad k = 0, 1, \ldots, m.
\quad}
$$

The derivation mirrors §3: write the integral
$\int \binom{m}{k} p^k(1-p)^{m-k} \cdot \frac{p^{\alpha_n - 1}(1-p)^{\beta_n - 1}}{B(\alpha_n, \beta_n)}\,\mathrm dp$,
recognise the Beta kernel and use the Beta-function normalising
constant.

Mean and variance:
$\mathbb E[\tilde x] = m \cdot \alpha_n/(\alpha_n + \beta_n)$ and

$$
\mathrm{Var}(\tilde x \mid x)
=
m \cdot \frac{\alpha_n \beta_n (\alpha_n + \beta_n + m)}{(\alpha_n + \beta_n)^2(\alpha_n + \beta_n + 1)},
$$

again exceeding the Binomial-only variance $m \cdot p(1-p)$ — the
extra term is parameter uncertainty.

---

## 5. The Year-End Predictive Total

### 5.1 Naïve formula and the trap

After observing months $1, \ldots, m$ with cumulative actuals
$S_m = \sum_{t=1}^m x_t$, define the remaining total
$\tilde S = \sum_{t=m+1}^{12} \tilde x_t$. We want the distribution of
the **annual total** $T = S_m + \tilde S$.

A naïve calculation says: each $\tilde x_t \mid x_{1:m} \sim N(\mu_m,
\sigma_m^2 + \sigma^2)$ from §2; "they are iid" so the variance of
$\tilde S$ is $(12-m)(\sigma_m^2 + \sigma^2)$.

**This is wrong.** The future months $\tilde x_t$ are *only*
conditionally iid **given $\theta$**; marginally over the posterior
they share $\theta$ and are therefore correlated. The naïve formula
under-estimates the year-end variance.

### 5.2 Correct derivation

Write $\tilde x_t = \theta + \varepsilon_t$ where the $\varepsilon_t$
are iid $N(0, \sigma^2)$, independent of $\theta$. Sum:

$$
\tilde S
\;=\;
(12 - m)\,\theta \;+\; \sum_{t=m+1}^{12} \varepsilon_t.
$$

The two summands are independent given $x_{1:m}$. The first has
distribution $(12-m) \cdot N(\mu_m, \sigma_m^2)$, i.e.
$N\big((12-m)\mu_m,\;(12-m)^2 \sigma_m^2\big)$. The second is
$N\big(0, (12-m)\sigma^2\big)$. Their sum is Normal:

$$
\boxed{\quad
\tilde S \mid x_{1:m}
\;\sim\;
N\big((12-m)\mu_m,\;\; (12-m)^2\,\sigma_m^2 \;+\; (12-m)\,\sigma^2\big).
\quad}
$$

The annual total is

$$
T \mid x_{1:m}
\;\sim\;
N\big(S_m + (12-m)\mu_m,\;\; (12-m)^2 \sigma_m^2 + (12-m)\sigma^2\big).
$$

Compare with the naïve formula: the parameter-uncertainty term is
$(12-m)^2 \sigma_m^2$, **quadratic** in the horizon, not linear. The
noise term is linear, $(12-m)\sigma^2$, as in iid additive aggregation.

### 5.3 Probability of exceeding budget

Given a budget ceiling $B$,

$$
P(T > B \mid x_{1:m})
\;=\;
1 - \Phi\!\Big(\frac{B - \big(S_m + (12-m)\mu_m\big)}{\sqrt{(12-m)^2 \sigma_m^2 + (12-m)\sigma^2}}\Big),
$$

where $\Phi$ is the standard Normal CDF.

### 5.4 FYF instance

After 6 months: $\mu_6 = \text{R\$ 1{,}085{,}000}$,
$\sigma_6 = \text{R\$ 32{,}000}$, $\sigma = \text{R\$ 80{,}000}$,
$S_6 = \text{R\$ 6{,}510{,}000}$, $B = \text{R\$ 13{,}200{,}000}$,
$12 - m = 6$.

- Mean $T$: $6{,}510{,}000 + 6 \cdot 1{,}085{,}000 = \text{R\$ 13{,}020{,}000}$.
- Variance $T$:
  $36 \cdot 32{,}000^2 + 6 \cdot 80{,}000^2
   = 3.6864 \times 10^{10} + 3.84 \times 10^{10}
   = 7.5264 \times 10^{10}$.
- Standard deviation: $\sigma_T \approx \text{R\$ 274{,}343}$.
- $z = (13{,}200{,}000 - 13{,}020{,}000)/274{,}343 \approx 0.656$.
- $P(T > B) \approx 1 - \Phi(0.656) \approx 0.256$.

So about a **26 %** chance of exceeding the annual budget at mid-year.
Using the *naïve* iid formula would give variance
$6 \cdot (32{,}000^2 + 80{,}000^2) = 4.4544 \times 10^{10}$,
$\sigma_T \approx \text{R\$ 211{,}055}$, $z \approx 0.853$, and
$P(T > B) \approx 0.197$ — nearly seven percentage points too low.
**The dependence between future months is not a rounding error.**

---

## 6. Connection to Article 1 — Monte Carlo Posterior Predictive

The closed-form predictive is convenient when conjugacy is available,
but the *Monte Carlo* recipe works in any setting and is conceptually
the cleanest connection to Article 1 of the series.

**Recipe.** To draw a posterior predictive sample $\tilde x$:

1. Draw $\theta^{(s)} \sim \pi(\theta \mid x)$.
2. Draw $\tilde x^{(s)} \sim f(\,\cdot \mid \theta^{(s)})$.

Repeat $S$ times. The empirical distribution of $\{\tilde x^{(s)}\}$
approximates $p(\tilde x \mid x)$ as $S \to \infty$ (Strong Law of
Large Numbers).

This is exactly the simulation strategy from Article 1 — Monte Carlo
sampling — but with the **posterior** as the input distribution
instead of the prior. Article 1 asked "given a prior over the unknown
parameters, what does the budget look like?". Phase 4 of this article
asks "given the *posterior* — i.e. the prior updated with observed
months — what does the *remaining year* look like?". Same machine,
better-informed input.

For multi-period forecasts (e.g. the year-end total), preserve the
correlation by drawing $\theta^{(s)}$ once per replication and then
sampling all future months conditional on that $\theta^{(s)}$:

```
for s = 1 .. S:
    θ_s   ← posterior sample
    x_s   ← (θ_s + N(0, σ²)) for each remaining month
    T_s   ← S_observed + sum(x_s)
return empirical distribution of {T_s}
```

Naïvely sampling $\tilde x_t$ from the marginal predictive
$N(\mu_m, \sigma_m^2 + \sigma^2)$ independently across $t$ would
reproduce the *wrong* variance from §5.1.

---

## 7. Bayes Factors (brief)

For two competing models $M_1, M_2$ with parameter-prior pairs
$\big(\theta_1, \pi_1\big)$ and $\big(\theta_2, \pi_2\big)$, the
**Bayes factor** in favour of $M_1$ is

$$
BF_{12}
\;=\;
\frac{p(x \mid M_1)}{p(x \mid M_2)}
\;=\;
\frac{\int f_1(x \mid \theta_1)\,\pi_1(\theta_1)\,\mathrm d\theta_1}
     {\int f_2(x \mid \theta_2)\,\pi_2(\theta_2)\,\mathrm d\theta_2}.
$$

The numerator and denominator are the **marginal likelihoods**
(evidences) under each model — the Bayes-factor concept already
appeared in Phase 1 §2 as the normalising constant of the posterior.

**Interpretation.** $BF_{12} = 5$ reads "the data are 5 times more
likely under $M_1$ than under $M_2$"; combined with prior model
probabilities $\pi(M_i)$ via
$\pi(M_1 \mid x)/\pi(M_2 \mid x) = BF_{12} \cdot \pi(M_1)/\pi(M_2)$
it gives posterior odds.

**Jeffreys' scale** (informal; see Kass & Raftery, 1995):

| $\log_{10} BF_{12}$ | $BF_{12}$       | Evidence for $M_1$         |
|---------------------|------------------|----------------------------|
| $0$ to $0.5$        | $1$ to $\sim 3$  | Anecdotal                  |
| $0.5$ to $1$        | $\sim 3$ to $10$ | Substantial                |
| $1$ to $1.5$        | $10$ to $\sim 30$ | Strong                    |
| $1.5$ to $2$        | $\sim 30$ to $100$| Very strong              |
| $> 2$               | $> 100$          | Decisive                   |

**Comparison with Article 2's AIC/BIC.** AIC/BIC are penalised log-
likelihoods evaluated at the MLE; they approximate model comparison
without integrating. In large samples
$-\tfrac{1}{2}\,\text{BIC} \approx \log p(x \mid M)$, so
$\log BF_{12} \approx -\tfrac{1}{2}(\text{BIC}_1 - \text{BIC}_2)$.
AIC penalises differently (by $-k$ rather than $-\tfrac{k}{2}\log n$)
and corresponds to a different limit (predictive accuracy on a held-
out sample) — see Burnham & Anderson (2002).

For Normal-Normal with two priors, the marginal likelihood has a
closed form (see exercise 5); for other models we usually estimate it
by Monte Carlo or thermodynamic integration. We do not develop those
estimators here — the article uses Bayes factors only as a
conceptually clean comparator to AIC/BIC.

---

## 8. Summary

- **Posterior predictive**:
  $p(\tilde x \mid x) = \int f(\tilde x \mid \theta)\pi(\theta \mid x)\mathrm d\theta$.
  Always wider than the credible interval because it adds sampling
  noise to parameter uncertainty.
- **Closed forms**: Normal-Normal → $N(\mu_n, \sigma_n^2 + \sigma^2)$;
  Gamma-Poisson → $\text{NegBin}(\alpha_n, \beta_n/(\beta_n+1))$;
  Beta-Binomial → Beta-Binomial distribution.
- **Year-end total**: Normal with mean $S_m + (12-m)\mu_m$ and
  variance $(12-m)^2 \sigma_m^2 + (12-m)\sigma^2$ —
  **quadratic in the horizon** for the parameter share, linear for the
  noise share. The naïve "iid future months" formula
  understates the variance.
- **Monte Carlo recipe**: draw $\theta$ from the posterior, then
  sample observations conditional on that $\theta$. Preserve
  correlation across future periods by reusing the same $\theta$
  draw for all of them.
- **Bayes factors**: ratio of marginal likelihoods; informally
  $\log BF_{12} \approx -\tfrac{1}{2}(\text{BIC}_1 - \text{BIC}_2)$
  in large samples.

Phase 5 brings the FYF model end-to-end: the conjugate updaters from
Phase 2, sequential engine from Phase 3, and the predictive machinery
above are wired into a complete annual cycle, simulated across five
scenarios, and validated with posterior predictive checks.

## References

- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed., Ch. 2.5
  (predictive), Ch. 7 (model comparison and Bayes factors).
- Hoff, P. (2009). *A First Course in Bayesian Statistical Methods*,
  Ch. 4 (predictive inference).
- Kass, R. E. & Raftery, A. E. (1995). "Bayes factors". *J. Amer.
  Statist. Assoc.* 90, 773–795.
- Burnham, K. P. & Anderson, D. R. (2002). *Model Selection and
  Multimodel Inference*. Springer.
