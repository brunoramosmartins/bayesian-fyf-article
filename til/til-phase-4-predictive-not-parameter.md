# TIL — Your CFO Doesn't Want the Parameter. They Want the Prediction.

**Phase:** 4 · **Topic:** posterior predictive · **Domain:** FYF

## Hook

Bayesian inference estimates parameters. Bayesian *prediction*
estimates what will actually happen. The second one pays the bills.

## Insight

The posterior $\pi(\theta \mid x)$ tells you about the mean cost. The
posterior **predictive** $p(\tilde x \mid x)$ tells you what next month
will actually cost. The difference: the predictive includes both
parameter uncertainty AND observation noise. The predictive interval
is always wider than the credible interval — and that wider interval
is the honest one.

A CFO who asks "will we stay under budget?" is asking a predictive
question, not a parameter question. The answer is
$P(\text{Total} > B)$, which integrates over both $\theta$ and future
noise. And — the part most people miss — the future months *share*
$\theta$, so the variance of the year-end total is **not** the sum of
per-month variances. Treating the months as independent under-
estimates the over-budget risk.

## Example

Posterior after 6 months: $\theta \sim N(1{,}085\text{K}, 32\text{K}^2)$.
Credible interval for $\theta$: $[\,1{,}022\text{K},\; 1{,}148\text{K}\,]$.
Predictive interval for next month: $[\,916\text{K},\; 1{,}254\text{K}\,]$.

The prediction is ≈ 2.7× wider. That is not pessimism — it is honesty.
The parameter is well-estimated; the future is still noisy.

For the year-end total the gap is even larger. The correct variance
$(12-m)^2\sigma_m^2 + (12-m)\sigma^2$ adds 70 % more variance than the
naïve "iid future months" formula. At a R$ 13.2M budget ceiling, that
is 26 % over-budget probability instead of 20 % — a six-point
difference that matters.

## Takeaway

Always report the predictive interval, not the credible interval, when
the question is about future observations. The credible interval is
for the statistician; the predictive interval is for the
decision-maker.
