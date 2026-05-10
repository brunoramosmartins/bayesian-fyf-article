# Phase 3 — Sequential Updating and Shrinkage

> Working notes for the article. We prove that the FYF cycle (one
> revision per quarter, fed by months of incoming actuals) is
> mathematically identical to the single-batch posterior, derive the
> shrinkage rate, give the recursive Kalman-gain form, and analyse
> prior sensitivity. Notation follows `docs/outline.md`.

---

## 1. Sequential = Batch — the Theorem the FYF Cycle Rests On

### 1.1 Statement

Let $\pi(\theta;\xi_0)$ be a prior in a conjugate family with update rule
$\xi \mapsto \mathrm{update}(\xi, x)$ for a single observation. Let
$x_{1:n} = (x_1, \ldots, x_n)$ be a sequence of conditionally iid
observations from the sampling model.

Define two posteriors:

- **Sequential**: starting from $\xi_0$, apply $\mathrm{update}$ once
  per observation, reusing the previous posterior as the next prior:
  $\xi_k^{\mathrm{seq}} = \mathrm{update}\big(\xi_{k-1}^{\mathrm{seq}}, x_k\big)$
  for $k = 1, \ldots, n$.
- **Batch**: apply the conjugate update once with all data:
  $\xi_n^{\mathrm{bat}} = \mathrm{update\_batch}(\xi_0, x_{1:n})$.

**Theorem (sequential = batch).** Under conditional independence,
$\xi_n^{\mathrm{seq}} = \xi_n^{\mathrm{bat}}$.

The theorem is the formal justification for treating each FYF cycle as
the single Bayesian update from one quarter's data, with last quarter's
posterior as prior. Re-deriving from scratch (single-batch from $\xi_0$
plus all 12 months) gives the same answer.

### 1.2 Proof for Normal-Normal — direct computation

Take $\theta \sim N(\mu_0, \sigma_0^2)$ and the Normal-Normal update
rule from §2 of `notes/phase2-conjugate-families.md`:
$\tau_n = \tau_0 + n\tau$ and
$\mu_n = (\tau_0 \mu_0 + n\tau\bar x_n)/\tau_n$, where
$\tau \equiv 1/\sigma^2$ and $\bar x_n = (x_1 + \cdots + x_n)/n$.

**Step 1 — Sequential after one observation:**

$$
\tau_1 = \tau_0 + \tau,
\qquad
\mu_1 = \frac{\tau_0 \mu_0 + \tau x_1}{\tau_1}.
$$

**Step 2 — Sequential after two observations**, treating $(\mu_1, \tau_1)$
as the new prior and applying the same rule with the single $x_2$:

$$
\tau_2^{\mathrm{seq}} = \tau_1 + \tau = (\tau_0 + \tau) + \tau = \tau_0 + 2\tau.
$$

For the mean,

$$
\mu_2^{\mathrm{seq}}
\;=\; \frac{\tau_1 \mu_1 + \tau x_2}{\tau_1 + \tau}
\;=\; \frac{(\tau_0 + \tau) \cdot \frac{\tau_0 \mu_0 + \tau x_1}{\tau_0 + \tau} + \tau x_2}{\tau_0 + 2\tau}
\;=\; \frac{\tau_0 \mu_0 + \tau x_1 + \tau x_2}{\tau_0 + 2\tau}.
$$

**Step 3 — Batch after two observations.** $\bar x_2 = (x_1 + x_2)/2$,

$$
\mu_2^{\mathrm{bat}}
\;=\; \frac{\tau_0 \mu_0 + 2\tau \bar x_2}{\tau_0 + 2\tau}
\;=\; \frac{\tau_0 \mu_0 + \tau(x_1 + x_2)}{\tau_0 + 2\tau}.
$$

The two are equal. The argument extends to any $n$ by induction (the
inductive step is exactly the same algebra applied to
$(\tau_{n-1}^{\mathrm{seq}}, \mu_{n-1}^{\mathrm{seq}})$).

### 1.3 Proof for general exponential families

For a sampling model in the canonical exponential family
$f(x \mid \theta) = h(x)\, g(\theta)\, \exp(\eta(\theta)^\top T(x))$,
the conjugate prior has the form
$\pi(\theta;\nu_0, \tau_0) \propto g(\theta)^{\nu_0} \exp(\eta(\theta)^\top \tau_0)$.
The conjugate update rule on hyperparameters is

$$
\nu_0 \;\mapsto\; \nu_0 + n,
\qquad
\tau_0 \;\mapsto\; \tau_0 + \sum_{i=1}^n T(x_i).
$$

The map is **additive in $n$ and in the sufficient statistic
$\sum T(x_i)$**. Sequentially feeding $x_k$ adds $1$ to $\nu$ and
$T(x_k)$ to $\tau$ at each step; after $n$ steps the cumulative
increments are $n$ and $\sum_{i=1}^n T(x_i)$ — identical to feeding the
batch in one go. Therefore
$\xi_n^{\mathrm{seq}} = \xi_n^{\mathrm{bat}}$.

The four pairs in `notes/phase2-conjugate-families.md` all admit this
form; the explicit hyperparameter updates we derived in Phase 2 are
specialisations of the additive rule above.

> **Why this matters.** The theorem says the FYF cycle is **not** a
> heuristic — it is literally the mathematically correct way to chain
> updates. An analyst who saves the previous posterior and applies the
> conjugate rule to incoming data will, by the end of the year, hold
> exactly the posterior they would have computed had they waited and
> done one batch update on December 31.

---

## 2. The Shrinkage Formula and Its Three Consequences

### 2.1 The precision-weighted mean as a shrinkage estimator

The Normal-Normal posterior mean from Phase 2 admits two equivalent
forms — both are useful. Let $\tau_0 = 1/\sigma_0^2$, $\tau = 1/\sigma^2$.

**Convex-combination form.**

$$
\mu_n
\;=\; w_0(n)\,\mu_0 \;+\; \big(1 - w_0(n)\big)\,\bar x_n,
\qquad
w_0(n) \;\equiv\; \frac{\tau_0}{\tau_0 + n\tau}.
$$

The posterior mean is a convex combination of prior mean and sample
mean; the **shrinkage weight** $w_0(n)$ is the share contributed by the
prior. By construction $w_0(n) \in (0, 1)$.

**Recursive (Kalman-gain) form.**

$$
\mu_n \;=\; \mu_{n-1} + K_n (x_n - \mu_{n-1}),
\qquad
K_n \;=\; \frac{\sigma_{n-1}^2}{\sigma_{n-1}^2 + \sigma^2}.
$$

Derivation: write
$\mu_n = \frac{\tau_{n-1} \mu_{n-1} + \tau x_n}{\tau_{n-1} + \tau}$,
multiply numerator and denominator by $\sigma_{n-1}^2 \sigma^2$, and
solve for $\mu_n - \mu_{n-1}$. The "innovation" $x_n - \mu_{n-1}$ is
the surprise; the gain $K_n$ scales how much of that surprise the
posterior absorbs. As the prior tightens ($\sigma_{n-1}^2 \to 0$),
$K_n \to 0$ and the posterior stops moving. **This is the discrete
Kalman filter for a static parameter** — preview of the Kalman article.

### 2.2 Three asymptotic results

**(C1) Shrinkage weight decays at rate $1/n$.** Direct computation:

$$
w_0(n) \;=\; \frac{\tau_0}{\tau_0 + n\tau}
\;=\; \frac{1}{1 + n\,(\tau/\tau_0)}
\;=\; \frac{\sigma_0^2/\sigma^2}{\sigma_0^2/\sigma^2 + n}.
$$

Hence $w_0(n) = \Theta(1/n)$. More precisely
$w_0(n) \to 0$ as $n \to \infty$ at exact rate $1/n$ (the leading
constant is $\sigma^2/\sigma_0^2$).

**(C2) Posterior variance also decays at rate $1/n$.**

$$
\sigma_n^2 \;=\; \frac{1}{\tau_0 + n\tau}
\;=\; \frac{\sigma^2 \sigma_0^2}{\sigma^2 + n \sigma_0^2}
\;\sim\; \frac{\sigma^2}{n} \quad (n \to \infty).
$$

The leading-order rate $\sigma^2/n$ matches the *frequentist* sampling
variance of the sample mean: at large $n$, the prior is asymptotically
irrelevant and the Bayesian and frequentist intervals coincide. This
is the formal sense in which the prior "washes out".

**(C3) Posterior mean converges to the data mean.** By (C1),
$\mu_n - \bar x_n = w_0(n)(\mu_0 - \bar x_n) \to 0$. If
$\bar x_n \to \theta_\star$ (LLN under the true model), then
$\mu_n \to \theta_\star$. Combining with (C2), the posterior collapses
to a point mass at $\theta_\star$.

### 2.3 FYF reading: when does the data dominate?

With reference parameters $\sigma_0 = 150{,}000$, $\sigma = 80{,}000$
(so $\sigma^2/\sigma_0^2 \approx 0.2844$):

$$
w_0(n) \;=\; \frac{0.2844}{0.2844 + n}.
$$

| Month $n$ | $w_0(n)$ (prior weight) | $1-w_0(n)$ (data weight) |
|----------:|------------------------:|-------------------------:|
| 1         | 0.2213                  | 0.7787                   |
| 2         | 0.1245                  | 0.8755                   |
| 3         | 0.0866                  | 0.9134                   |
| 6         | 0.0452                  | 0.9548                   |
| 9         | 0.0306                  | 0.9694                   |
| 12        | 0.0231                  | 0.9769                   |

The data's share crosses **80 % at $n = 2$** (already by February-end if
we revise monthly), **95 % at $n = 6$** (mid-year FYF). By the year-end
posterior the budget plan accounts for ≈ 2.3 % of the answer.

To find the exact crossover algebraically, set $1 - w_0(n) = c$ and
solve for $n$:

$$
n_{\!c} \;=\; \frac{c}{1-c} \cdot \frac{\sigma^2}{\sigma_0^2}.
$$

For $c = 0.80$: $n = 4 \cdot 0.2844 = 1.14$, so the threshold is first
crossed at $n = 2$. For $c = 0.95$: $n = 19 \cdot 0.2844 = 5.40$,
crossed at $n = 6$. For $c = 0.99$: $n = 99 \cdot 0.2844 = 28.16$ —
which would only be reached after several years of monthly data.

---

## 3. Prior Sensitivity

### 3.1 Two priors with the same $\sigma_0$ converge

Let two analysts hold priors $N(\mu_0^{(A)}, \sigma_0^2)$ and
$N(\mu_0^{(B)}, \sigma_0^2)$. After observing the *same data*, their
posterior means are

$$
\mu_n^{(A)} \;=\; w_0(n)\, \mu_0^{(A)} + (1-w_0(n))\,\bar x_n,
\qquad
\mu_n^{(B)} \;=\; w_0(n)\, \mu_0^{(B)} + (1-w_0(n))\,\bar x_n.
$$

Subtracting:

$$
\boxed{\quad
\mu_n^{(A)} - \mu_n^{(B)} \;=\; w_0(n)\,\big(\mu_0^{(A)} - \mu_0^{(B)}\big).
\quad}
$$

The disagreement between the two analysts shrinks by exactly $w_0(n)$
each step. By month 6, an initial gap of R$ 150{,}000 has shrunk to
$0.0452 \times 150{,}000 \approx \text{R\$ 6{,}780}$. By year-end, to
≈ R$ 3{,}460. The data forces consensus.

### 3.2 Different $\sigma_0$: the more confident prior persists longer

When $\sigma_0^{(A)} \ne \sigma_0^{(B)}$ the analysis is the same per
analyst but the shrinkage rates differ. The analyst with the *smaller*
$\sigma_0$ has a larger $\tau_0$, hence a slower decay of $w_0(n)$.
Their posterior takes longer to wash out. Asymptotically both converge
to $\bar x_n$ regardless.

### 3.3 Prior–data conflict

Define the **prior–data discrepancy**

$$
D_n \;\equiv\; \frac{|\mu_0 - \bar x_n|}{\sigma_0}.
$$

When $D_n$ is small (≲ 1), the data "agrees with the plan" in the sense
that the sample mean is within the prior's range of uncertainty.
**Prior–data conflict** is the regime $D_n \gg 3$ (more than three
prior standard deviations).

What happens? Mechanically the posterior just sits between $\mu_0$ and
$\bar x_n$ — but its location is dictated by precision, not by which
source is "right". If the prior is misspecified (e.g. the planner was
overconfident), the posterior continues to shrink toward $\bar x_n$;
the cost is slow convergence and a posterior credible interval that
does not contain the truth in finite samples.

**Operational rule for the FYF.** If $D_n > 3$ at any FYF, **stop and
diagnose** before mechanically accepting the posterior. Either the
prior was wrong (re-elicit) or the sampling model is wrong (the world
changed — a re-org, a vendor switch, a structural shock). The Bayesian
machine assumes both are right; when one is not, the answer is
mechanical but not meaningful.

This connects to the **posterior predictive check** developed in
Phase 5: large $D_n$ should produce predictive p-values close to 0
or 1, flagging the conflict automatically.

### 3.4 (Brief) Connection to James–Stein

The shrinkage estimator $\mu_n = w_0 \mu_0 + (1-w_0) \bar x_n$ is, in
the case of a fixed $w_0$ (large $n$ with appropriate scaling), close
to the **James–Stein estimator** for normal means in three or more
dimensions. The James–Stein result shows that in $d \ge 3$ dimensions a
shrinkage estimator dominates the MLE in mean-squared error. The
Bayesian posterior mean is a principled, dimension-agnostic version of
this: shrink toward the prior, with the shrinkage strength dictated by
relative precision rather than chosen ad hoc. We do not develop this
further in the article — Stein, *Inadmissibility of the Usual Estimator
for the Mean of a Multivariate Normal Distribution* (1956) is the
classic reference for the curious reader.

---

## 4. Summary

- **Sequential = batch** (proved for Normal-Normal directly and for
  exponential families via additive sufficient statistics). The FYF
  cycle is mathematically identical to a single big update.
- **Shrinkage rate $1/n$**: $w_0(n) = (\sigma^2/\sigma_0^2)/(\sigma^2/\sigma_0^2 + n)$,
  $\sigma_n^2 \sim \sigma^2/n$. Both decay at the same rate.
- **Recursive Kalman-gain form**: $\mu_n = \mu_{n-1} + K_n(x_n - \mu_{n-1})$.
- **Disagreement between two priors with the same $\sigma_0$ shrinks
  by $w_0(n)$ per step**. Two analysts converge.
- **Prior-data conflict** ($D_n > 3$) flags model misspecification and
  is the right hand-off into Phase 5 model checking.

Phase 4 layers the **posterior predictive distribution** on top of the
posterior, turning $\theta$-estimates into forecasts of next month's
cost and the year-end total.

## References

- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed., Ch. 2.6.
- Berger, J. (1985). *Statistical Decision Theory and Bayesian Analysis*, Ch. 4.
- Hoff, P. (2009). *A First Course in Bayesian Statistical Methods*, Ch. 5–6.
- Stein, C. (1956). "Inadmissibility of the usual estimator for the
  mean of a multivariate normal distribution." *Proc. Third Berkeley
  Symp.* (James-Stein reference).
