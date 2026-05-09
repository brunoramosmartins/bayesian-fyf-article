# Exercises — Phase 2 (Conjugate Families)

> Pencil-and-paper exercises. Numerical answers can be cross-checked
> against `src/conjugate.py` once the module is in place.

---

## A. Proofs

### 1. Derive the Normal-Normal posterior from scratch

Take prior $\theta \sim N(\mu_0, \sigma_0^2)$ and likelihood
$x \mid \theta \sim N(\theta, \sigma^2)$ with $\sigma^2$ known and a
single observation $x$. Show every algebraic step:

(a) Write $\pi(\theta) f(x \mid \theta)$ as the exponential of a
quadratic in $\theta$.
(b) Group by powers of $\theta$, factor out the leading coefficient
$1/\sigma_0^2 + 1/\sigma^2$, and *complete the square*.
(c) Identify the resulting density as
$N(\mu_1, \sigma_1^2)$ and read off $\mu_1, \sigma_1^2$ in terms of the
hyperparameters and $x$.
(d) Verify by setting $\sigma_0^2 \to \infty$ that the answer reduces
to $N(x, \sigma^2)$ — recovering the improper-prior limit.

---

### 2. Precision-weighted mean

Generalise exercise 1 to $n$ iid observations and rewrite the result
in **precision form**. With $\tau_0 \equiv 1/\sigma_0^2$,
$\tau \equiv 1/\sigma^2$:

$$
\mu_n \;=\; \frac{\tau_0 \mu_0 \,+\, n\tau\,\bar x}{\tau_0 + n\tau}.
$$

Show the algebra explicitly (multiply numerator and denominator of the
variance form by $\sigma_0^2 \sigma^2$). State in one sentence what
"precision" means operationally for the FYF cycle.

---

### 3. Posterior precision = prior precision + data precision

Prove $\tau_n = \tau_0 + n\tau$ from the result of exercise 2.
Interpret: each new month of FYF data adds **exactly $\tau$** units of
precision, regardless of the value of the observation. The information
contribution is *fixed*; only the location $\mu_n$ changes with data.

---

### 4. Derive the Gamma-Poisson posterior

Prior $\lambda \sim \text{Gamma}(\alpha_0, \beta_0)$ in the rate
parameterisation. Likelihood
$x_1, \ldots, x_n \mid \lambda \overset{\text{iid}}{\sim} \text{Poisson}(\lambda)$.

(a) Write the unnormalised posterior $\pi(\lambda)\, L(\lambda)$.
(b) Group powers of $\lambda$ and exponentials in $\lambda$.
(c) Identify the kernel as $\text{Gamma}(\alpha_0 + \sum x_i,\, \beta_0 + n)$.
(d) Interpret $\alpha_0$ as a "prior event count" and $\beta_0$ as a
"prior pseudo-observation period". Why does the Beta-Gamma elicitation
helper `gamma_prior_from_rate(rate, confidence)` return
$(\alpha_0, \beta_0) = (\text{confidence}\cdot\text{rate}, \text{confidence})$?

---

### 5. Derive the Beta-Binomial posterior

Prior $p \sim \text{Beta}(\alpha_0, \beta_0)$, single observation
$x \mid p \sim \text{Binomial}(n, p)$. Multiply, identify the kernel
$p^{\alpha_0 + x - 1}(1 - p)^{\beta_0 + n - x - 1}$, conclude
$p \mid x \sim \text{Beta}(\alpha_0 + x,\, \beta_0 + n - x)$. Comment on
why this is the easiest of the four derivations: which property of
the Beta and Binomial densities makes the kernel matching nearly
trivial?

---

### 6. Improper uniform prior on $\theta$ for the Normal model

Take $\pi(\theta) \propto 1$ on $\mathbb R$ and likelihood
$x_1, \ldots, x_n \mid \theta \sim N(\theta, \sigma^2)$. Using the
identity from §2.2 of `notes/phase2-conjugate-families.md`, show

$$
\theta \mid x_{1:n} \;\sim\; N\!\Big(\bar x,\; \tfrac{\sigma^2}{n}\Big).
$$

Compare this with the frequentist sampling distribution of $\bar X$
under iid sampling from $N(\theta, \sigma^2)$. They are numerically
identical but their interpretations differ — write one sentence
explaining how.

---

## B. Computations

### 7. Posterior trajectory for the FYF reference scenario

Default parameters: $\mu_0 = 1{,}050{,}000$, $\sigma_0 = 150{,}000$,
$\sigma = 80{,}000$. Compute the Normal-Normal posterior at four
revision points:

| Revision   | $n$ | $\bar x_n$        | Compute $\mu_n$, $\sigma_n$, 95 % CI |
|------------|----:|--------------------|----------------------------------------|
| Q1 close   |  1  | 1{,}120{,}000     |                                        |
| FYF #1     |  3  | 1{,}095{,}000     |                                        |
| FYF #2     |  6  | 1{,}085{,}000     |                                        |
| Year-end   | 12  | 1{,}078{,}000     |                                        |

For each row, give $\mu_n$ and $\sigma_n$ to the nearest thousand and
the 95 % equal-tailed credible interval $\mu_n \pm 1.96\sigma_n$. Plot
the four credible intervals on the same vertical axis to visualise
shrinkage.

*Approximate expected values (verify):*

| $n$ | $\mu_n$ (R$)         | $\sigma_n$ (R$)   |
|----:|----------------------|--------------------|
|  1  | $\approx 1{,}104{,}500$ | $\approx 70{,}600$ |
|  3  | $\approx 1{,}091{,}100$ | $\approx 44{,}100$ |
|  6  | $\approx 1{,}083{,}400$ | $\approx 31{,}900$ |
| 12  | $\approx 1{,}077{,}400$ | $\approx 22{,}800$ |

Note the shrinkage of $\sigma_n$ by roughly $1/\sqrt n$ — formalised
in Phase 3.

---

### 8. Gamma-Poisson update for the incident-rate prior

Prior $\lambda \sim \text{Gamma}(3, 1)$ (mean 3 incidents/month).
Six months of observed counts: $\{2, 4, 1, 3, 5, 2\}$.

(a) Compute the posterior hyperparameters $(\alpha_n, \beta_n)$.
(b) Compute the posterior mean $\alpha_n / \beta_n$ and compare with
the empirical mean $\bar x = 17/6 \approx 2.833$ and the prior mean
$3.000$. The posterior mean lies between the two — quantify the
shrinkage as $w_0 = \beta_0 / (\beta_0 + n)$ and confirm
$\mu_n^{\text{post}} = w_0 \cdot \mu^{\text{prior}} + (1 - w_0) \cdot \bar x$.
(c) Now repeat with a stronger prior $\text{Gamma}(30, 10)$ (same
mean 3 but ten months of pseudo-data). What changes? Which posterior
is closer to the empirical mean, and why?

*Expected:* posterior $\text{Gamma}(20, 7)$, mean $20/7 \approx 2.857$,
$w_0 = 1/7$, so the posterior is mostly empirical.

---

### 9. (Optional) Normal-Inverse-Gamma posterior for unknown variance

Prior $N\text{-}IG(\mu_0, \kappa_0, \alpha_0, \beta_0) =
N\text{-}IG(1.05\!\times\!10^6,\; 1,\; 2,\; 6.4\!\times\!10^9)$ (chosen so
that the prior mean of $\sigma^2$ is $\beta_0/(\alpha_0 - 1) = 6.4\!\times\!10^9$,
matching $\sigma = 80{,}000$).

Observe three months $x = (1{,}120{,}000;\; 1{,}080{,}000;\; 1{,}095{,}000)$,
$\bar x = 1{,}098{,}333.\overline 3$, sample variance
$S/n \approx 4.\overline{36}\!\times\!10^8$.

Compute $\mu_n, \kappa_n, \alpha_n, \beta_n$. Verify that the marginal
posterior for $\theta$ is a Student-t with df $2\alpha_n$, location
$\mu_n$, and scale $\sqrt{\beta_n/(\alpha_n \kappa_n)}$. Compare the
95 % credible interval from the $t$ marginal with the Normal-Normal
answer from exercise 7 row 2 (same data, but $\sigma$ assumed known) —
which is wider, and by how much? The difference is the *cost of
admitting that $\sigma^2$ is also unknown*.
