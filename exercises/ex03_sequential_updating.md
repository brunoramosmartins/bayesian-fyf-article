# Exercises — Phase 3 (Sequential Updating and Shrinkage)

> Pencil-and-paper exercises. Numerical answers can be cross-checked
> against `src/updating.py` once the module is in place.

---

## A. Proofs

### 1. Sequential = batch for Normal-Normal

Prove that, for the Normal-Normal model, applying the conjugate update
to $x_1$ and then to $x_2$ (using the posterior of step 1 as the
prior of step 2) gives the same posterior as applying the conjugate
update once to the batch $(x_1, x_2)$.

(a) Compute $(\mu_1, \tau_1)$ from the prior $(\mu_0, \tau_0)$ and $x_1$.
(b) Apply the rule again with $(\mu_1, \tau_1)$ as prior and $x_2$ as
the single observation; obtain $(\mu_2^{\mathrm{seq}}, \tau_2^{\mathrm{seq}})$.
(c) Compute $(\mu_2^{\mathrm{bat}}, \tau_2^{\mathrm{bat}})$ from the
batch rule with $\bar x_2 = (x_1+x_2)/2$.
(d) Show $\mu_2^{\mathrm{seq}} = \mu_2^{\mathrm{bat}}$ and
$\tau_2^{\mathrm{seq}} = \tau_2^{\mathrm{bat}}$ algebraically.

*Generalise by induction to $n$ observations.*

---

### 2. Shrinkage weight is monotone-decreasing in $n$

Prove that $w_0(n) = \tau_0 / (\tau_0 + n\tau)$ satisfies:

(a) $w_0(n) \in (0, 1)$ for all $n \ge 0$.
(b) $w_0(n+1) < w_0(n)$ (strictly decreasing).
(c) $w_0(n) \to 0$ as $n \to \infty$ at rate $1/n$ (state and prove
the precise asymptotic
$w_0(n) \sim \sigma^2/(n\sigma_0^2)$ as $n \to \infty$).

*Hint for (c): write $w_0(n) = (\sigma^2/\sigma_0^2)/(\sigma^2/\sigma_0^2 + n)$.*

---

### 3. Posterior variance and the data-mean comparison

Prove $\sigma_n^2 = 1/(\tau_0 + n\tau)$ is monotone-decreasing and that
$\sigma_n^2 \sim \sigma^2/n$ for large $n$. Compare with the
**frequentist** sampling variance of $\bar X_n$, which is exactly
$\sigma^2/n$ for any $n$. Why do the two coincide asymptotically
even though they disagree at small $n$? (Hint: the prior contribution
is bounded, so its share is $O(1/n)$ in the variance.)

*Comment on the connection to the Monte Carlo standard error
$\sigma/\sqrt n$ from Article 1: both reflect the same square-root
shrinkage, applied to a different object (sample mean vs posterior).*

---

### 4. Two analysts, same $\sigma_0$: their disagreement decays as $w_0(n)$

Two analysts share the same prior precision $\tau_0$ but different
prior means $\mu_0^{(A)} \ne \mu_0^{(B)}$. After observing the same
data $x_{1:n}$, prove

$$
\big|\mu_n^{(A)} - \mu_n^{(B)}\big|
\;=\; w_0(n)\,\big|\mu_0^{(A)} - \mu_0^{(B)}\big|.
$$

Interpret: at each step, prior disagreement shrinks by exactly $w_0$.
This is *the* reason informed Bayesian disagreement is self-correcting
in the long run.

---

### 5. Recursive (Kalman-gain) form

Prove that the Normal-Normal posterior mean admits the recursion

$$
\mu_n \;=\; \mu_{n-1} \;+\; K_n\,(x_n - \mu_{n-1}),
\qquad
K_n \;=\; \frac{\sigma_{n-1}^2}{\sigma_{n-1}^2 + \sigma^2}.
$$

(a) Start from the closed form
$\mu_n = (\tau_{n-1} \mu_{n-1} + \tau x_n)/(\tau_{n-1} + \tau)$
(treating $(\mu_{n-1}, \tau_{n-1})$ as the prior at step $n$).
(b) Multiply numerator and denominator by $\sigma_{n-1}^2 \sigma^2$.
(c) Subtract $\mu_{n-1}$ from both sides and isolate the gain.

*This is the discrete Kalman filter for a static parameter.* The
"innovation" $x_n - \mu_{n-1}$ is the surprise at step $n$; the gain
$K_n$ is the posterior's responsiveness to that surprise. As the
posterior tightens, $K_n \to 0$ and the posterior stops moving.

---

## B. Computations

### 6. Shrinkage table for the FYF reference scenario

With $\mu_0 = 1{,}050{,}000$, $\sigma_0 = 150{,}000$,
$\sigma = 80{,}000$, the ratio $\sigma^2/\sigma_0^2 \approx 0.2844$.

(a) Compute $w_0(n)$ and the data weight $1 - w_0(n)$ for
$n \in \{1, 3, 6, 9, 12\}$. Round to 4 decimal places.
(b) Find the smallest $n$ such that $1 - w_0(n) > 0.80$.
(c) Find the smallest $n$ such that $1 - w_0(n) > 0.95$.
(d) Use the closed form $n_c = c/(1-c) \cdot \sigma^2/\sigma_0^2$ to
state the *exact threshold* and verify it crosses an integer at the
expected month.

*Expected:* $1 - w_0(2) \approx 0.876 > 0.80$;
$1 - w_0(6) \approx 0.955 > 0.95$. The 80 % threshold is first crossed
at $n = 2$, the 95 % threshold at $n = 6$.

---

### 7. Two analysts disagree on $\mu_0$ but agree on $\sigma_0$

Analyst A has prior $N(1{,}050{,}000,\ 150{,}000^2)$.
Analyst B has prior $N(1{,}200{,}000,\ 150{,}000^2)$.
Same likelihood $\sigma = 80{,}000$. After 6 months of actuals with
sample mean $\bar x_6 = 1{,}085{,}000$:

(a) Compute $\mu_6^{(A)}$ and $\mu_6^{(B)}$.
(b) Compute the gap $\mu_6^{(B)} - \mu_6^{(A)}$.
(c) Compare with the original disagreement $\mu_0^{(B)} - \mu_0^{(A)} = 150{,}000$.
What fraction of the original disagreement remains? Cross-check against
$w_0(6)$.

*Expected:* gap shrinks from R$ 150{,}000 to ≈ R$ 6{,}780, a
$w_0(6) \approx 0.0452$ retention. After year-end ($n = 12$) the gap
would be ≈ R$ 3{,}460.

---

### 8. (Optional) Prior-data conflict simulation

Take the reference prior and simulate 12 months of actuals from a
**misspecified** sampling model with mean $\theta_\star = \mu_0 + 5\sigma_0$
(five prior standard deviations above the planned mean) and noise
$\sigma$.

(a) Compute the discrepancy $D_n = |\mu_0 - \bar x_n|/\sigma_0$ at each
month. Confirm $D_n$ exceeds 3 quickly.
(b) Compute the posterior mean trajectory. Does it converge to
$\theta_\star$, to $\mu_0$, or somewhere in between? Why?
(c) Comment on the operational rule from §3.3 of the theory notes:
when $D_n > 3$, stop and diagnose. Which months would trigger the rule?
