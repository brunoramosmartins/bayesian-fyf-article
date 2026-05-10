# TIL — The Posterior Mean Is a Precision-Weighted Average

**Phase:** 3 · **Topic:** shrinkage, sequential updating · **Domain:** FYF

## Hook

The Bayesian posterior mean has a formula so clean it should be framed
on the wall of every FP&A team.

## Insight

$$
\mu_n
\;=\;
\frac{\tau_0}{\tau_0 + n\tau} \cdot \mu_0
\;+\;
\frac{n\tau}{\tau_0 + n\tau} \cdot \bar x_n,
$$

where $\tau_0 = 1/\sigma_0^2$ (prior precision) and
$\tau = 1/\sigma^2$ (data precision per observation).

In words: the revised forecast is a weighted average of the plan and
the actuals. The weight on the plan is proportional to how confident
we were in it. The weight on the data is proportional to how many
months we have observed and how precise each observation is.

After 1 month the data already carries about 78 % of the answer.
After 6 months, about 95 %. After 12 months, 98 %. The plan does not
"go away" — it just becomes a rounding error.

## Example

Reference parameters: $\sigma_0 = 150{,}000$, $\sigma = 80{,}000$, so
$\sigma^2/\sigma_0^2 \approx 0.2844$.

| Month $n$ | Data weight $1-w_0(n)$ |
|----------:|----------------------:|
| 1         | 78 %                  |
| 2         | 88 %                  |
| 3         | 91 %                  |
| 6         | 95 %                  |
| 12        | 98 %                  |

The 80 % threshold is first crossed at $n = 2$; the 95 % threshold at
$n = 6$. By July's mid-year FYF the data has already taken over.

## Takeaway

You do not need to "choose" between the plan and the actuals. The
precision-weighted mean does it for you, optimally, automatically, and
with a mathematical proof that it minimises expected squared error.
