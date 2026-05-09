# Phase 2 — Conjugate Families

> Working notes for the article. Every result is derived in full algebra
> ("complete the square" is shown step by step at least once). The goal
> is the four conjugate pairs the FYF model uses, plus the
> precision-weighted mean as the article's recurring identity.
> Notation follows `docs/outline.md` §"Notation Conventions".

---

## 1. Conjugacy and Why It Matters

Let $\mathcal F$ be a family of prior densities indexed by a
finite-dimensional hyperparameter $\xi$. The family $\mathcal F$ is
**conjugate** to a sampling model $f(x \mid \theta)$ if, for every
prior $\pi(\theta;\xi) \in \mathcal F$ and every observed dataset $x$,
the posterior

$$
\pi(\theta \mid x) \;\propto\; f(x \mid \theta)\,\pi(\theta;\xi)
$$

is itself a member of $\mathcal F$ — say with hyperparameter $\xi'$.
The map $\xi \mapsto \xi'$ defined by Bayes' theorem is the
**update rule** of the conjugate family.

Three reasons conjugacy is the right starting point for the article:

1. **Closed form.** The marginal likelihood
   $\int f(x \mid \theta)\pi(\theta)\,\mathrm d\theta$ is implied by the
   normalising constant of $\mathcal F$; we never compute it directly.
2. **Sequential updating is trivial.** If $\xi'$ is the posterior
   hyperparameter after dataset $x_{1:n}$, then feeding $x_{n+1}$ alone
   into the same update rule starting from $\xi'$ gives the right
   posterior for $x_{1:n+1}$. We prove this rigorously in Phase 3.
3. **Interpretability.** The hyperparameters of conjugate posteriors
   admit a *pseudo-observation* reading: the prior contributes
   "imaginary data" of a known size and content, and the actual data
   are added on top. This makes prior elicitation operational.

> **Brief sketch — why exponential families work.** Sampling models in
> the canonical exponential family
> $f(x \mid \theta) = h(x) g(\theta) \exp\!\big(\eta(\theta)^\top T(x)\big)$
> always admit a conjugate prior of the form
> $\pi(\theta) \propto g(\theta)^{\nu_0}
> \exp\!\big(\eta(\theta)^\top \tau_0\big)$;
> the posterior just bumps $\nu_0 \to \nu_0 + n$ and
> $\tau_0 \to \tau_0 + \sum_i T(x_i)$. We do not prove this
> characterisation here — Robert (2007), Ch. 3 has the proof — but it
> explains why every pair below has the same kernel-matching shape.

---

## 2. Normal-Normal (Known Variance) — the Article's Anchor

### 2.1 Set-up

Let $\theta$ be the unknown mean of a Normal sampling model with
**known** variance $\sigma^2 > 0$. Observations are conditionally iid:

$$
x_1, \ldots, x_n \mid \theta \;\overset{\text{iid}}{\sim}\; N(\theta, \sigma^2),
\qquad
\theta \sim N(\mu_0, \sigma_0^2).
$$

The likelihood, viewed as a function of $\theta$, is

$$
L(\theta)
\;=\;
\prod_{i=1}^n \frac{1}{\sigma\sqrt{2\pi}}
\exp\!\Big(-\tfrac{1}{2\sigma^2}(x_i - \theta)^2\Big)
\;\propto\;
\exp\!\Big(-\tfrac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \theta)^2\Big).
$$

Drop the $\theta$-independent factors $1/(\sigma\sqrt{2\pi})^n$.

### 2.2 First simplification: $\bar x$ is a sufficient statistic

Expand the squared deviations using the identity
$\sum (x_i - \theta)^2 = \sum (x_i - \bar x)^2 + n(\bar x - \theta)^2$,
where $\bar x = \tfrac{1}{n}\sum x_i$:

$$
\sum_{i=1}^n (x_i - \theta)^2 \;=\; \sum_{i=1}^n (x_i - \bar x)^2 \;+\; n\,(\bar x - \theta)^2.
$$

The first term does not involve $\theta$, so it is absorbed into the
proportionality constant. The likelihood simplifies to

$$
L(\theta) \;\propto\;
\exp\!\Big(-\tfrac{n}{2\sigma^2}(\bar x - \theta)^2\Big).
$$

This is exactly the kernel of $N(\bar x, \sigma^2/n)$ in $\theta$, so
$\bar x$ together with $n$ is a sufficient statistic for $\theta$.

### 2.3 Multiply prior × likelihood and complete the square

The prior is

$$
\pi(\theta) \;\propto\;
\exp\!\Big(-\tfrac{1}{2\sigma_0^2}(\theta - \mu_0)^2\Big).
$$

The unnormalised posterior:

$$
\pi(\theta \mid x_{1:n})
\;\propto\;
\exp\!\bigg(\;
   -\tfrac{1}{2\sigma_0^2}(\theta - \mu_0)^2
   \;-\;\tfrac{n}{2\sigma^2}(\theta - \bar x)^2
\;\bigg).
$$

Set $\tau_0 \equiv 1/\sigma_0^2$ and $\tau \equiv 1/\sigma^2$ (precisions).
Expand the exponent and group powers of $\theta$:

$$
-\tfrac{1}{2}\Big[
   \tau_0 (\theta - \mu_0)^2
   \;+\; n\tau (\theta - \bar x)^2
\Big]
\;=\;
-\tfrac{1}{2}\Big[
   (\tau_0 + n\tau)\,\theta^2
   \;-\; 2(\tau_0 \mu_0 + n\tau \bar x)\,\theta
   \;+\; (\tau_0 \mu_0^2 + n\tau \bar x^2)
\Big].
$$

The constant in $\theta$ is again absorbed into the proportionality
factor. Define

$$
\tau_n \;\equiv\; \tau_0 + n\tau,
\qquad
\mu_n \;\equiv\; \frac{\tau_0 \mu_0 + n\tau\,\bar x}{\tau_n}.
$$

The exponent becomes

$$
-\tfrac{\tau_n}{2}\Big[\theta^2 - 2\mu_n \theta\Big]
\;=\;
-\tfrac{\tau_n}{2}(\theta - \mu_n)^2 \;+\; \text{const},
$$

after completing the square ($\theta^2 - 2\mu_n \theta =
(\theta - \mu_n)^2 - \mu_n^2$). The constant $-\tfrac{\tau_n}{2}\mu_n^2$
is independent of $\theta$ and absorbed.

### 2.4 The Normal-Normal posterior

The kernel is that of a Normal density in $\theta$:

$$
\boxed{\quad
\theta \mid x_{1:n} \;\sim\; N\!\big(\mu_n,\; \sigma_n^2\big),
\qquad
\sigma_n^2 \;=\; 1/\tau_n,
\quad}
$$

with the **precision-weighted mean** identity

$$
\mu_n \;=\; \frac{\tau_0 \mu_0 + n\tau\,\bar x}{\tau_0 + n\tau},
\qquad
\tau_n \;=\; \tau_0 + n\tau.
$$

In variance form (multiply numerator and denominator by
$\sigma_0^2 \sigma^2$):

$$
\mu_n \;=\; \frac{\sigma^2 \mu_0 + n\sigma_0^2 \bar x}{\sigma^2 + n\sigma_0^2},
\qquad
\sigma_n^2 \;=\; \frac{\sigma^2 \sigma_0^2}{\sigma^2 + n\sigma_0^2}.
$$

### 2.5 Three readings of the same result

**(a) Precision additivity.** $\tau_n = \tau_0 + n\tau$. The posterior
precision is the sum of prior precision and the precision contributed
by $n$ observations. Each observation adds exactly $\tau$ units of
precision.

**(b) Weighted average of means.** Let
$w_0 = \tau_0 / (\tau_0 + n\tau)$ and $w_d = n\tau / (\tau_0 + n\tau) = 1 - w_0$.
Then
$\mu_n = w_0 \mu_0 + w_d \bar x$. The weights sum to one and are
proportional to the *information content* of each source: prior
information contributes $\tau_0$ units, data contributes $n\tau$ units,
and $\mu_n$ is the convex combination weighted by share of total
information. This is the **CFO sentence** the TIL points to: "the
forecast is a weighted average of the plan and the actuals, where the
weights are how confident we are in each."

**(c) Shrinkage to the prior.** Equivalently,
$\mu_n = \bar x + w_0 (\mu_0 - \bar x)$. The posterior mean shrinks
the data mean toward the prior mean by a factor $w_0$. As $n$ grows,
$w_0 \to 0$ and the shrinkage vanishes; we develop this asymptotic in
Phase 3.

### 2.6 FYF interpretation and sanity check

Reference parameters from `docs/model-design.md`:
$\mu_0 = 1{,}050{,}000$, $\sigma_0 = 150{,}000$ (so
$\tau_0 \approx 4.444 \times 10^{-11}$),
$\sigma = 80{,}000$ ($\tau \approx 1.5625 \times 10^{-10}$).

After $n = 3$ months with $\bar x_3 = 1{,}098{,}333.\overline{3}$:

- Data precision: $3\tau \approx 4.687 \times 10^{-10}$.
- Posterior precision: $\tau_3 \approx 5.131 \times 10^{-10}$.
- Posterior s.d.: $\sigma_3 \approx \text{R\$ 44{,}131}$.
- Posterior mean: $\mu_3 \approx \text{R\$ 1{,}094{,}163}$.
- Prior weight: $w_0 \approx 0.087$ — the data carries ≈ 91 % of the
  weight already after one quarter.

Both numbers are reproduced by `src/conjugate.py` and the notebook.

### 2.7 Improper uniform prior

The improper prior $\pi(\theta) \propto 1$ on $\mathbb R$ corresponds
to the limit $\sigma_0 \to \infty$, equivalently $\tau_0 \to 0$. The
update rule still applies:

$$
\theta \mid x_{1:n} \;\sim\; N(\bar x, \sigma^2/n).
$$

The posterior is identical, *as a density in $\theta$*, to the
frequentist sampling distribution of $\bar X$. This is the formal
sense in which a flat prior recovers the classical answer.

---

## 3. Normal-Inverse-Gamma (Unknown Mean and Variance)

### 3.1 Set-up

When $\sigma^2$ is unknown we work jointly in $(\theta, \sigma^2)$ on
$\mathbb R \times (0, \infty)$. The conjugate prior is the
**Normal-Inverse-Gamma**, written $N\text{-}IG(\mu_0, \kappa_0, \alpha_0, \beta_0)$:

$$
\sigma^2 \sim \text{Inverse-Gamma}(\alpha_0, \beta_0),
\qquad
\theta \mid \sigma^2 \sim N\!\big(\mu_0,\; \sigma^2 / \kappa_0\big).
$$

Density:

$$
\pi(\theta, \sigma^2) \;\propto\;
(\sigma^2)^{-\alpha_0 - 3/2}
\exp\!\Big(-\tfrac{1}{\sigma^2}\big[\beta_0 + \tfrac{\kappa_0}{2}(\theta - \mu_0)^2\big]\Big).
$$

Hyperparameter reading: $\mu_0$ is the prior centre for $\theta$;
$\kappa_0$ is the **prior pseudo-sample-size** for $\theta$ (large
$\kappa_0$ ⟹ tight prior on the mean conditional on $\sigma^2$);
$\alpha_0$ and $\beta_0$ govern the marginal Inverse-Gamma on
$\sigma^2$ with $\mathbb E[\sigma^2] = \beta_0/(\alpha_0 - 1)$ for
$\alpha_0 > 1$.

### 3.2 Likelihood and posterior — direct multiplication

For $n$ iid observations with sample mean $\bar x$ and corrected sum
of squares
$S \equiv \sum_{i=1}^n (x_i - \bar x)^2$:

$$
L(\theta, \sigma^2)
\;\propto\;
(\sigma^2)^{-n/2}
\exp\!\Big(-\tfrac{1}{2\sigma^2}\big[S + n(\bar x - \theta)^2\big]\Big).
$$

Multiply by the prior and group exponents. The $(\theta - \mu_0)^2$
and $(\theta - \bar x)^2$ pieces combine with the same
"completing the square" identity used in §2.3, this time with weights
$\kappa_0$ and $n$:

$$
\kappa_0 (\theta - \mu_0)^2 + n(\theta - \bar x)^2
\;=\;
(\kappa_0 + n)(\theta - \mu_n)^2
\;+\;
\frac{\kappa_0 n}{\kappa_0 + n}(\bar x - \mu_0)^2,
$$

with $\mu_n = (\kappa_0 \mu_0 + n \bar x)/(\kappa_0 + n)$.

### 3.3 Posterior hyperparameters

The kernel matches $N\text{-}IG(\mu_n, \kappa_n, \alpha_n, \beta_n)$
with

$$
\boxed{\quad
\mu_n \;=\; \frac{\kappa_0 \mu_0 + n\bar x}{\kappa_0 + n},
\qquad
\kappa_n \;=\; \kappa_0 + n,
\quad}
$$

$$
\boxed{\quad
\alpha_n \;=\; \alpha_0 + \frac{n}{2},
\qquad
\beta_n \;=\; \beta_0 + \tfrac{1}{2}\,S
\;+\; \tfrac{1}{2}\,\frac{\kappa_0\, n}{\kappa_0 + n}\,(\bar x - \mu_0)^2.
\quad}
$$

### 3.4 Marginals and asymptotics

**Marginal on $\theta$.** Integrating out $\sigma^2$ gives a
**non-central Student-t** distribution:

$$
\theta \mid x_{1:n} \;\sim\; t_{2\alpha_n}\!\Big(\mu_n,\; \tfrac{\beta_n}{\alpha_n \kappa_n}\Big),
$$

i.e. a $t$ density with location $\mu_n$, scale
$\sqrt{\beta_n / (\alpha_n \kappa_n)}$, and $2\alpha_n$ degrees of
freedom.

**Marginal on $\sigma^2$.** $\sigma^2 \mid x_{1:n} \sim
\text{Inverse-Gamma}(\alpha_n, \beta_n)$, mean
$\beta_n / (\alpha_n - 1)$ for $\alpha_n > 1$.

**Concentration as $n \to \infty$.** Suppose the $x_i$ are iid with
true mean $\theta_\star$ and true variance $\sigma_\star^2$; then
$\bar x \to \theta_\star$ and $S/n \to \sigma_\star^2$ a.s. by the
strong law of large numbers. The posterior hyperparameters satisfy

- $\mu_n \to \theta_\star$ since the second term in the numerator
  dominates;
- $\kappa_n / n \to 1$, so $\kappa_n \to \infty$;
- $\alpha_n / n \to 1/2$;
- $\beta_n / n \to \tfrac{1}{2} \sigma_\star^2$, so the marginal mean
  $\beta_n/(\alpha_n - 1) \to \sigma_\star^2$.

The marginal $t$ on $\theta$ has scale
$\sqrt{\beta_n/(\alpha_n \kappa_n)} \approx
\sqrt{\sigma_\star^2 / n} \to 0$, df $\to \infty$, so the marginal
collapses to a point mass at $\theta_\star$. The posterior
**concentrates on $(\bar x, S/n) \to (\theta_\star, \sigma_\star^2)$**.

---

## 4. Gamma-Poisson — Updating Event Rates

### 4.1 Set-up

Let $\lambda > 0$ be the unknown rate of a Poisson sampling model
applied per unit time (here, per month):

$$
x_1, \ldots, x_n \mid \lambda \;\overset{\text{iid}}{\sim}\; \text{Poisson}(\lambda),
\qquad
\lambda \sim \text{Gamma}(\alpha_0, \beta_0).
$$

We use the **rate parameterisation** of the Gamma:
$\pi(\lambda) \propto \lambda^{\alpha_0 - 1} e^{-\beta_0 \lambda}$.
Mean $\alpha_0/\beta_0$, variance $\alpha_0/\beta_0^2$.

### 4.2 Posterior derivation

Likelihood:

$$
L(\lambda) \;=\; \prod_{i=1}^n \frac{\lambda^{x_i} e^{-\lambda}}{x_i!}
\;\propto\; \lambda^{\sum x_i}\, e^{-n\lambda}.
$$

Multiply by the Gamma prior and read off the kernel in $\lambda$:

$$
\pi(\lambda \mid x_{1:n})
\;\propto\;
\lambda^{\sum x_i} e^{-n\lambda} \cdot \lambda^{\alpha_0 - 1} e^{-\beta_0 \lambda}
\;=\;
\lambda^{(\alpha_0 + \sum x_i) - 1}\, e^{-(\beta_0 + n)\lambda}.
$$

Kernel match against $\text{Gamma}(\alpha, \beta)$:

$$
\boxed{\quad
\lambda \mid x_{1:n} \;\sim\; \text{Gamma}\!\Big(\alpha_0 + \textstyle\sum_{i=1}^n x_i,\; \beta_0 + n\Big).
\quad}
$$

### 4.3 Pseudo-observation reading

Compare prior and posterior:

| Quantity                | Prior                                | Posterior                                           |
|-------------------------|---------------------------------------|-----------------------------------------------------|
| Pseudo-event-count      | $\alpha_0$                            | $\alpha_0 + \sum x_i$                                |
| Pseudo-observation-time | $\beta_0$                             | $\beta_0 + n$                                        |
| Posterior mean of $\lambda$ | $\alpha_0/\beta_0$                | $(\alpha_0 + \sum x_i)/(\beta_0 + n)$                |

The Gamma prior with hyperparameters $(\alpha_0, \beta_0)$ behaves
*as if* the analyst had previously observed $\alpha_0$ events over
$\beta_0$ "imaginary months". After $n$ real months yielding
$\sum x_i$ events, the totals add. This is why
`gamma_prior_from_rate(rate, confidence)` in `src/priors.py` returns
$(\alpha_0, \beta_0) = (\text{confidence} \cdot \text{rate}, \text{confidence})$:
``confidence`` plays the role of pseudo-observation-time.

### 4.4 FYF illustration

Prior $\text{Gamma}(3, 1)$ (mean 3 incidents/month, equivalent to one
month of pseudo-data). Six observed months with counts
$\{2, 4, 1, 3, 5, 2\}$, $\sum x_i = 17$.

Posterior $\text{Gamma}(3 + 17,\, 1 + 6) = \text{Gamma}(20, 7)$,
posterior mean $20/7 \approx 2.857$. The empirical mean
$\bar x = 17/6 \approx 2.833$ and the prior mean $3$ are close, so
shrinkage is mild; with a stronger prior (e.g. $\text{Gamma}(30, 10)$)
the same data would shrink the empirical mean visibly toward $3$.

---

## 5. Beta-Binomial — Updating Proportions

### 5.1 Set-up

Single observation case: $x \mid p \sim \text{Binomial}(n, p)$,
$p \sim \text{Beta}(\alpha_0, \beta_0)$. The kernel of the Beta is
$p^{\alpha_0 - 1}(1 - p)^{\beta_0 - 1}$.

### 5.2 Posterior derivation

$$
\pi(p \mid x) \;\propto\; p^x (1 - p)^{n - x} \cdot p^{\alpha_0 - 1}(1 - p)^{\beta_0 - 1}
\;=\; p^{\alpha_0 + x - 1}(1 - p)^{\beta_0 + n - x - 1}.
$$

Kernel match:

$$
\boxed{\quad
p \mid x \;\sim\; \text{Beta}(\alpha_0 + x,\; \beta_0 + n - x).
\quad}
$$

### 5.3 Pseudo-success reading

$\alpha_0$ is "prior successes", $\beta_0$ is "prior failures",
$\alpha_0 + \beta_0$ is the prior pseudo-sample-size. Each real
observation adds one to either pile. The Beta-Binomial pair is the
pseudo-observation pattern at its simplest.

### 5.4 Multiple independent batches

If the analyst sees several batches $(x_k, n_k)$, the posterior
after batch $K$ is

$$
p \mid x_{1:K} \;\sim\; \text{Beta}\!\Big(\alpha_0 + \textstyle\sum_k x_k,\;
\beta_0 + \textstyle\sum_k (n_k - x_k)\Big).
$$

The proof is the same kernel matching applied to the product of
binomial likelihoods.

---

## 6. The Pattern

All four pairs share a structure that the article will emphasise:

| Pair               | Prior hyperparams        | Update rule                                                                                          | Interpretation of $+n$ part |
|--------------------|--------------------------|------------------------------------------------------------------------------------------------------|------------------------------|
| Normal–Normal      | $\mu_0, \tau_0$          | $\tau_n = \tau_0 + n\tau$;  $\mu_n = (\tau_0 \mu_0 + n\tau\bar x)/\tau_n$                            | Add $n\tau$ units of precision |
| Normal–Inv-Gamma   | $\mu_0, \kappa_0, \alpha_0, \beta_0$ | $\kappa_n = \kappa_0 + n$;  $\alpha_n = \alpha_0 + n/2$; …                                | Add $n$ pseudo-observations  |
| Gamma–Poisson      | $\alpha_0, \beta_0$      | $\alpha_n = \alpha_0 + \sum x_i$;  $\beta_n = \beta_0 + n$                                            | Add events and pseudo-time   |
| Beta–Binomial      | $\alpha_0, \beta_0$      | $\alpha_n = \alpha_0 + x$;  $\beta_n = \beta_0 + n - x$                                               | Add successes and failures   |

In every case the prior contributes a finite "imaginary" sample, the
data contributes a real sample, and the totals add. This is the
intuitive content of conjugacy.

---

## 7. Summary

- The Normal–Normal posterior is $N(\mu_n, 1/\tau_n)$ with
  $\tau_n = \tau_0 + n\tau$ and
  $\mu_n = (\tau_0 \mu_0 + n\tau\bar x)/\tau_n$. **The posterior mean
  is the precision-weighted average of prior mean and sample mean.**
- The Normal–Inverse-Gamma posterior is
  $N\text{-}IG(\mu_n, \kappa_n, \alpha_n, \beta_n)$ with explicit
  updates on all four hyperparameters; as $n \to \infty$ the posterior
  concentrates on $(\bar x, S/n)$.
- The Gamma–Poisson posterior is
  $\text{Gamma}(\alpha_0 + \sum x_i, \beta_0 + n)$.
- The Beta–Binomial posterior is
  $\text{Beta}(\alpha_0 + x, \beta_0 + n - x)$.

Phase 3 chains these update rules across the FYF cycle, proves
sequential = batch, and derives the shrinkage rate. Phase 4 layers the
posterior predictive on top.

## References

- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed., Ch. 2–3.
- DeGroot, M. (1970). *Optimal Statistical Decisions*, Ch. 9 (NIG).
- Hoff, P. (2009). *A First Course in Bayesian Statistical Methods*, Ch. 5–6.
- Robert, C. (2007). *The Bayesian Choice*, Ch. 3 (exponential families).
