# Phase 1 — Bayesian Foundations

> Working notes for the article. Every result is derived in full; "FYF
> link" boxes connect each abstract idea to the budget revision cycle.
> Notation follows `docs/outline.md` §"Notation Conventions".

---

## 1. From Conditional Density to Bayes' Theorem (Continuous Form)

Let $X$ denote observed data and $\theta$ an unknown parameter, both
treated as random variables on a common probability space. Assume the
joint density $f(x, \theta)$ exists. The **product rule for densities**
factors the joint two ways:

$$
f(x, \theta) \;=\; f(x \mid \theta)\,\pi(\theta) \;=\; \pi(\theta \mid x)\,f(x).
$$

Equating the two factorisations and solving for $\pi(\theta \mid x)$:

$$
\boxed{\quad
\pi(\theta \mid x) \;=\; \frac{f(x \mid \theta)\,\pi(\theta)}{f(x)},
\quad}
$$

provided $f(x) > 0$. This is **Bayes' theorem for continuous
parameters**. The four objects in the formula have names worth fixing:

| Symbol               | Name                | Reading                                        |
|----------------------|---------------------|------------------------------------------------|
| $\pi(\theta)$        | prior               | belief about $\theta$ before seeing $x$        |
| $f(x \mid \theta)$    | sampling density    | the data-generating model                      |
| $f(x)$               | marginal likelihood | probability of $x$ averaged over $\pi(\theta)$ |
| $\pi(\theta \mid x)$ | posterior           | belief about $\theta$ after seeing $x$         |

When $f(x \mid \theta)$ is read as a function of $\theta$ for fixed $x$,
it is called the **likelihood**, written $L(\theta) \equiv f(x \mid \theta)$.
The likelihood is not a probability density in $\theta$ — it does not
generally integrate to one over $\theta$.

> **FYF link.** $\theta$ is the unknown mean monthly cost; $x$ is the
> observed monthly actuals; $\pi(\theta)$ is the budget plan expressed
> as a distribution; $\pi(\theta \mid x)$ is the revised forecast.

---

## 2. Proportionality and the Marginal Likelihood

Because $f(x)$ does not depend on $\theta$, it acts only as a
normalising constant. We can drop it and write

$$
\pi(\theta \mid x) \;\propto\; f(x \mid \theta)\,\pi(\theta),
$$

read **"posterior is proportional to likelihood times prior"**. The
constant of proportionality is exactly

$$
f(x) \;=\; \int_\Theta f(x \mid \theta)\,\pi(\theta)\,\mathrm d\theta,
$$

the **marginal likelihood** (or **evidence**). It is the probability of
the observed data averaged over all possible parameter values weighted
by the prior. The integral is over $\Theta$, the parameter space.

The proportionality form is operationally powerful: in conjugate
families (Phase 2) we recognise the kernel of $f(x \mid \theta)\pi(\theta)$ as
that of a known distribution, identify the hyperparameters, and skip the
marginal-likelihood integral entirely. The normaliser is *implied* by
the family.

### Posterior is a proper probability density

A probability density must integrate to one. We verify this directly
from the formula:

$$
\int_\Theta \pi(\theta \mid x)\,\mathrm d\theta
\;=\;
\int_\Theta \frac{f(x \mid \theta)\,\pi(\theta)}{f(x)}\,\mathrm d\theta
\;=\;
\frac{1}{f(x)} \int_\Theta f(x \mid \theta)\,\pi(\theta)\,\mathrm d\theta
\;=\;
\frac{f(x)}{f(x)}
\;=\; 1.
$$

The argument requires $0 < f(x) < \infty$. When the prior is
**proper** (i.e. $\int \pi(\theta)\,\mathrm d\theta = 1$) and the
sampling density is finite, $f(x)$ is automatically finite and positive
for any observable $x$. When the prior is **improper** (e.g.
$\pi(\theta) \propto 1$ on $\mathbb R$) the integral can still be
finite — see §6 — but each case must be checked.

---

## 3. Point Summaries: MAP, Posterior Mean, Posterior Median

The full posterior is the right answer, but business decisions often
need a single number. Three canonical summaries:

- **Maximum a posteriori (MAP):** the mode of the posterior,
  $$\hat\theta_{\text{MAP}} \;=\; \arg\max_{\theta} \pi(\theta \mid x).$$
- **Posterior mean:**
  $$\hat\theta_{\text{PM}} \;=\; \mathbb E[\theta \mid x] \;=\; \int_\Theta \theta\,\pi(\theta \mid x)\,\mathrm d\theta.$$
- **Posterior median:** the value $\theta_{0.5}$ with
  $\Pr(\theta \le \theta_{0.5} \mid x) = 1/2$.

When the posterior is **symmetric and unimodal** (e.g. any Normal
posterior) the three coincide: symmetry forces the mean to equal the
median, and unimodality forces the mode to lie on the axis of symmetry.

When the posterior is skewed, they differ — and the choice matters.
The posterior mean minimises expected squared error, the median
minimises expected absolute error, and the MAP coincides with the
maximum-likelihood estimator (MLE) when the prior is uniform.

> **MAP with uniform prior = MLE.** If $\pi(\theta) \propto 1$ on a
> region containing the MLE, then
> $\pi(\theta \mid x) \propto L(\theta) \cdot 1$, so the mode of the
> posterior coincides with the maximiser of the likelihood. Bayesian
> inference with a flat prior reproduces the frequentist point
> estimator at the level of point estimates, although the *uncertainty
> statements* attached to that estimate are still different (§5).

---

## 4. Credible Intervals

A **$100(1-\alpha)\%$ credible interval** for $\theta$ is any subset
$C \subseteq \Theta$ with

$$
\Pr(\theta \in C \mid x) \;=\; 1 - \alpha.
$$

Two conventional choices:

**(a) Equal-tailed** $[a, b]$ with
$\Pr(\theta < a \mid x) = \Pr(\theta > b \mid x) = \alpha/2$. Easy to
compute from posterior quantiles. Default in the article's experiments.

**(b) Highest Posterior Density (HPD)**: the *shortest* interval (or
union of intervals) of total posterior probability $1-\alpha$.
Equivalently, the level set
$\{\theta : \pi(\theta \mid x) \ge k_\alpha\}$ for the largest $k_\alpha$
such that the set has probability $\ge 1-\alpha$. For symmetric
unimodal posteriors HPD = equal-tailed.

The article uses equal-tailed everywhere and shows HPD once for
comparison in §7 of the article (the "experiments" section).

---

## 5. Credible Interval ≠ Confidence Interval

This is the conceptual hinge of the article and deserves care.

**Bayesian credible interval.** A statement *about $\theta$ given the
observed data*. The interval $[a, b]$ depends on the realised $x$
(through the posterior), and $\theta$ is treated as random:
$$\Pr(\theta \in [a, b] \mid X = x) \;=\; 0.95.$$
Reading: "given what I saw, there is a 95 % chance the parameter is in
$[a, b]$."

**Frequentist confidence interval.** A statement *about the procedure*.
The endpoints $a(X), b(X)$ are random functions of the sample;
$\theta$ is a fixed unknown constant:
$$\Pr_\theta\!\big(\,a(X) \le \theta \le b(X)\,\big) \;=\; 0.95
\quad\text{for every } \theta.$$
Reading: "if I repeat this experiment many times, 95 % of the
intervals I produce will cover the true $\theta$." After observing one
specific $x$, the interval $[a(x), b(x)]$ either contains $\theta$ or
not — there is no probability statement about *this* interval.

The two intervals can numerically agree (Normal-Normal with known
variance and a flat prior gives identical limits) but they answer
different questions. The Bayesian interval admits a **direct
probabilistic interpretation about the parameter**; the confidence
interval does not.

> **FYF link.** "There is a 95 % chance year-end cost lies between R$ 12.6M
> and R$ 13.4M" is a credible-interval statement and the one budget
> committees actually want. The frequentist alternative — "if I
> re-ran 2026 many times, 95 % of intervals would cover the truth" —
> is awkward and operationally vacuous.

---

## 6. Prior Elicitation Strategies

Every Bayesian analysis must declare a prior. Four strategies, in
decreasing order of information content:

### 6.1 Informative prior

The prior encodes substantive subject-matter knowledge. **In FYF, the
budget plan is the prior.** A planner who states "we expect cost $\mu_0$
with confidence $\gamma$ that the truth lies within $\pm w$ of the plan"
is implicitly specifying a Normal prior $N(\mu_0, \sigma_0^2)$ with

$$
\sigma_0 \;=\; \frac{w}{z_{(1+\gamma)/2}},
$$

where $z_q$ is the standard-Normal quantile. For $\gamma = 0.90$,
$z_{0.95} = 1.645$.

### 6.2 Weakly informative prior

A proper distribution that is broad enough to let the data dominate but
narrow enough to rule out absurd values (negative costs, billion-real
salaries). Recommended default when no strong subject-matter prior is
available.

### 6.3 Improper prior

A non-negative function $\pi(\theta)$ with
$\int \pi(\theta)\,\mathrm d\theta = \infty$, e.g. the uniform prior
$\pi(\theta) \propto 1$ on $\mathbb R$. Improper priors are not
distributions but are often used as limits or to encode "complete
ignorance". The posterior may still be proper — see exercise 2.

### 6.4 Jeffreys' prior

The objective prior
$\pi_J(\theta) \;\propto\; \sqrt{I(\theta)},$ where $I(\theta)$ is the
Fisher information. Invariant under reparameterisation. We *cite* this
result; the full proof of invariance is outside scope.

The article's reference scenario uses an informative prior: the budget
plan with the planner's stated uncertainty.

---

## 7. Worked Example A — Beta-Binomial (Coin Flipping)

This example is the discrete-data counterpart that makes the Bayesian
machine concrete in two pages.

**Setting.** A coin has unknown probability of heads $\theta \in (0, 1)$.
We flip it $n$ times and observe $x$ heads. We want the posterior
$\pi(\theta \mid x)$.

**Prior.** $\theta \sim \text{Beta}(\alpha_0, \beta_0)$ with density

$$
\pi(\theta) \;=\; \frac{\theta^{\alpha_0 - 1} (1-\theta)^{\beta_0 - 1}}{B(\alpha_0, \beta_0)},
\quad
B(\alpha, \beta) \;=\; \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}.
$$

**Likelihood.** $X \mid \theta \sim \text{Binomial}(n, \theta)$,

$$
f(x \mid \theta) \;=\; \binom{n}{x} \theta^x (1-\theta)^{n-x}.
$$

**Posterior.** By Bayes' theorem in proportional form, ignoring factors
that do not depend on $\theta$:

$$
\pi(\theta \mid x)
\;\propto\;
\theta^x (1-\theta)^{n-x} \cdot \theta^{\alpha_0-1}(1-\theta)^{\beta_0-1}
\;=\;
\theta^{\alpha_0 + x - 1}(1-\theta)^{\beta_0 + n - x - 1}.
$$

This is the kernel of a $\text{Beta}(\alpha_0 + x, \beta_0 + n - x)$
density. By **kernel matching** (the integrate-to-one property forces
the constant), we conclude

$$
\boxed{\quad \theta \mid x \;\sim\; \text{Beta}(\alpha_0 + x,\; \beta_0 + n - x). \quad}
$$

**Interpretation.** The Beta parameters act as *pseudo-counts*:
$\alpha_0$ behaves like prior heads, $\beta_0$ like prior tails. After
observing $x$ heads in $n$ flips the counts simply add. This is the
discrete-data shadow of the precision-additivity that will appear in
the Normal-Normal case.

**Numerical instance.** Prior $\text{Beta}(2, 2)$ (mean $0.5$,
mildly informative). Observe $n = 10$, $x = 7$. Posterior:
$\text{Beta}(9, 5)$, mean $9/14 \approx 0.643$. The posterior mean lies
between the prior mean $0.5$ and the empirical proportion $0.7$,
weighted by the prior pseudo-count $\alpha_0 + \beta_0 = 4$ vs the data
sample size $n = 10$.

---

## 8. Worked Example B — Normal-Normal (Preview)

We preview the Normal-Normal pair that drives the FYF model. The full
derivation is in Phase 2; here we only state the result and use it.

**Setting.** $\theta$ is the unknown mean monthly cost. Observation
noise has known standard deviation $\sigma$. We observe a single month
$x_1$.

**Prior.** $\theta \sim N(\mu_0, \sigma_0^2)$.
**Likelihood.** $x_1 \mid \theta \sim N(\theta, \sigma^2)$.
**Posterior.** $\theta \mid x_1 \sim N(\mu_1, \sigma_1^2)$ with

$$
\mu_1 \;=\; \frac{\sigma^2 \mu_0 + \sigma_0^2 x_1}{\sigma^2 + \sigma_0^2},
\qquad
\sigma_1^2 \;=\; \frac{\sigma^2 \sigma_0^2}{\sigma^2 + \sigma_0^2}.
$$

In **precision form** ($\tau \equiv 1/\sigma^2$):
$\tau_1 = \tau_0 + \tau$ and
$\mu_1 = (\tau_0 \mu_0 + \tau x_1)/\tau_1$.

**FYF instance.** Reference parameters from `docs/model-design.md`:
$\mu_0 = 1{,}050{,}000$, $\sigma_0 = 150{,}000$, $\sigma = 80{,}000$,
$x_1 = 1{,}120{,}000$. Then

$$
\sigma_1^2 \;=\; \frac{80000^2 \cdot 150000^2}{80000^2 + 150000^2}
\;\approx\; 4.984\times 10^9
\;\Longrightarrow\;
\sigma_1 \approx \text{R\$ 70{,}598},
$$

$$
\mu_1 \;=\; \frac{80000^2 \cdot 1050000 + 150000^2 \cdot 1120000}{80000^2 + 150000^2}
\;\approx\; \text{R\$ 1{,}104{,}465}.
$$

A single month of data shrank the prior standard deviation from
R$ 150K to R$ 71K (~53 % reduction), and pulled the posterior mean
from R$ 1.050M to R$ 1.104M — most of the way toward the actual.

The fact that $\sigma_1 < \sigma_0$ for *any* finite $\sigma$ and
non-zero $\sigma_0$ is the **monotonic shrinkage** property,
formalised in Phase 3.

---

## 9. Summary

The continuous Bayesian machine has three moving parts:

1. **Bayes' theorem**: $\pi(\theta \mid x) \propto f(x \mid \theta)\pi(\theta)$.
2. **Posterior summaries**: MAP, posterior mean, credible intervals.
3. **Interpretation**: credible intervals make probability statements
   about $\theta$; confidence intervals make statements about the
   procedure.

In the FYF context, the prior is the budget plan, the likelihood is the
data-generating model for monthly cost, and the posterior is the
revised forecast. Phase 2 turns this template into closed-form formulas
for four conjugate pairs. Phase 3 chains the updates across months.

## References

- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed., Ch. 1–2.
- Hoff, P. (2009). *A First Course in Bayesian Statistical Methods*, Ch. 3.
- Robert, C. (2007). *The Bayesian Choice*, Ch. 1, 3 (Jeffreys' prior).
