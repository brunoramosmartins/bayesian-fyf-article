# Exercises — Phase 1 (Bayesian Foundations)

> Pencil-and-paper exercises. Solutions are not committed to the repo;
> the author works them by hand and ticks each box on completion.
> Numerical exercises can be cross-checked with `src/priors.py` once
> implemented.

---

## A. Proofs

### 1. Derive Bayes' theorem for a continuous parameter $\theta$

Start from the joint density $f(x, \theta)$ and the two equivalent
factorisations

$$
f(x, \theta) \;=\; f(x \mid \theta)\,\pi(\theta) \;=\; \pi(\theta \mid x)\,f(x).
$$

Show every algebraic step that leads to

$$
\pi(\theta \mid x) \;=\; \frac{f(x \mid \theta)\,\pi(\theta)}{f(x)},
\qquad
f(x) \;=\; \int_\Theta f(x \mid \theta)\,\pi(\theta)\,\mathrm d\theta.
$$

State explicitly the assumption needed to divide by $f(x)$.

---

### 2. Improper uniform prior + Normal likelihood ⟹ proper posterior

Take the improper prior $\pi(\theta) \propto 1$ on $\theta \in \mathbb R$
and the likelihood $f(x \mid \theta) = \frac{1}{\sigma\sqrt{2\pi}}
\exp\!\big(-\tfrac{1}{2\sigma^2}(x-\theta)^2\big)$ for $\sigma > 0$
known and one observation $x \in \mathbb R$.

Compute

$$
\int_{-\infty}^{\infty} f(x \mid \theta)\cdot 1\;\mathrm d\theta,
$$

show it equals $1$, and conclude that the posterior
$\pi(\theta \mid x) \propto f(x \mid \theta)$ is itself a Normal density
in $\theta$ — explicitly identify its mean and variance.

*Hint: the integral is the normalising constant of $N(x, \sigma^2)$
viewed as a density in $\theta$.*

---

### 3. MAP with uniform prior = MLE

Suppose $\pi(\theta) \propto 1$ on a region containing the maximiser
of the likelihood $L(\theta) = f(x \mid \theta)$. Show that

$$
\arg\max_\theta \pi(\theta \mid x) \;=\; \arg\max_\theta L(\theta).
$$

Discuss why this *point estimate* coincides with the MLE while the
associated *uncertainty statement* (a credible interval) does not
coincide with a confidence interval.

---

### 4. Symmetric unimodal posterior: MAP = mean = median

Let $\pi(\theta \mid x)$ be unimodal with mode $m$ and symmetric about
$m$, i.e. $\pi(m + h \mid x) = \pi(m - h \mid x)$ for all $h$. Show:

(a) The posterior median equals $m$ (use the symmetry to bisect the
total mass).
(b) The posterior mean equals $m$, assuming it exists (substitute
$\theta = m + u$ in the integral and exploit oddness).
(c) The MAP equals $m$ by definition.

---

### 5. Credible interval ≠ confidence interval

Construct a concrete numerical example for the model
$x_1 \mid \theta \sim N(\theta, 1)$, $\theta \sim N(0, 1)$, with
$x_1 = 2$.

(a) Compute the 95 % equal-tailed credible interval for $\theta$ from
the posterior $N\!\big(\frac{x_1}{2}, \frac{1}{2}\big)$ (preview from
Phase 1 §8 / Phase 2).
(b) Compute the 95 % frequentist confidence interval for $\theta$ from
the same data (treat $\theta$ as fixed; use $x_1 \pm 1.96$).
(c) Write one sentence stating what each interval *claims*. Highlight
where the two statements differ in their object (parameter vs
procedure).

---

## B. Computations

### 6. Translate budget plan into a Normal prior

A budget plan estimates monthly cost at $\mu_0 = \text{R\$ 1{,}050{,}000}$
with $\pm 15\%$ confidence at the 90 % level — i.e. the planner is
"90 % sure" the truth lies in $[\,\text{R\$ 892{,}500},\; \text{R\$ 1{,}207{,}500}\,]$.

Solve for $\sigma_0$ such that
$P(\theta \in [\mu_0 - 1.645\sigma_0,\; \mu_0 + 1.645\sigma_0]) = 0.90$.

(a) Write the explicit equation.
(b) Solve numerically.
(c) Cross-check by calling
`priors.normal_prior_from_budget(plan_value=1_050_000, confidence_pct=0.90, interval_width=0.15)`.

*Expected: $\sigma_0 \approx \text{R\$ 95{,}753}$.* Note: the article's
default uses a slightly weaker $\sigma_0 = \text{R\$ 150{,}000}$ for
round numbers; prior-sensitivity in Phase 6 will quantify the
difference.

---

### 7. One-month posterior for the FYF model

Using the prior from exercise 6 with $\sigma_0 = \text{R\$ 95{,}753}$ and
likelihood $x \mid \theta \sim N(\theta, \sigma^2)$ with
$\sigma = \text{R\$ 80{,}000}$, observe $x_1 = \text{R\$ 1{,}120{,}000}$.

Compute the posterior $\theta \mid x_1 \sim N(\mu_1, \sigma_1^2)$ via
the closed form (preview from §8):

$$
\sigma_1^2 \;=\; \frac{\sigma^2 \sigma_0^2}{\sigma^2 + \sigma_0^2},
\qquad
\mu_1 \;=\; \frac{\sigma^2 \mu_0 + \sigma_0^2 x_1}{\sigma^2 + \sigma_0^2}.
$$

Report:

- $\mu_1$ and $\sigma_1$ to the nearest real.
- The posterior shift $\Delta\mu \equiv \mu_1 - \mu_0$.
- The 95 % equal-tailed credible interval $\mu_1 \pm 1.96\sigma_1$.
- A one-sentence interpretation: by how much did one month of data
  pull the forecast away from the plan, and by how much did the
  uncertainty contract?

*Expected approximate values (verify):*
$\sigma_1 \approx \text{R\$ 61.4K}$,
$\mu_1 \approx \text{R\$ 1{,}091.2K}$,
$\Delta\mu \approx +\text{R\$ 41.2K}$.

Sanity check: with $\sigma_0 > \sigma$ the posterior weight on the data
($\sigma_0^2/(\sigma^2+\sigma_0^2) \approx 0.589$) exceeds the weight on
the plan, so the posterior mean lies closer to $x_1$ than to $\mu_0$.
