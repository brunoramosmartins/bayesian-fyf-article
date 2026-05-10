"""Posterior predictive distributions and year-end forecasting.

The closed-form derivations live in
``notes/phase4-predictive-inference.md``; this module is a numerical
layer on top:

- ``posterior_predictive_normal``   — wraps ``scipy.stats.norm``.
- ``posterior_predictive_gamma_poisson`` — wraps ``scipy.stats.nbinom``.
- ``posterior_predictive_beta_binomial`` — wraps ``scipy.stats.betabinom``.
- ``year_end_predictive_total`` — Normal predictive for the annual
  total, using the **correct** variance accounting for the dependence
  between future months that share :math:`\\theta`.
- ``prob_over_budget`` — the convenience tail probability.
- ``posterior_predictive_sample`` — Monte Carlo posterior predictive
  sampling, the Article-1 strategy applied to the posterior.
- ``log_marginal_likelihood_normal_normal`` and ``bayes_factor`` — the
  bare minimum for a closed-form Bayes factor between two Normal-Normal
  models.

Future months in a multi-period predictive sum **must not** be drawn
independently from the marginal predictive — they share :math:`\\theta`.
The Monte Carlo helper preserves this correlation by drawing one
:math:`\\theta` per replication and conditioning all subsequent draws on
it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from src.conjugate import (
    BetaPosterior,
    GammaPosterior,
    NormalPosterior,
)

# =============================================================================
# Single-period predictives
# =============================================================================


def posterior_predictive_normal(
    posterior: NormalPosterior, sigma_sq: float
) -> stats.rv_frozen:
    """Posterior predictive ``N(mu_n, sigma_n^2 + sigma^2)``.

    Parameters
    ----------
    posterior
        Normal posterior on :math:`\\theta`.
    sigma_sq
        Known sampling variance :math:`\\sigma^2`. Must be positive.
    """
    if sigma_sq <= 0:
        raise ValueError(f"sigma_sq must be > 0; got {sigma_sq}")
    var = posterior.variance() + sigma_sq
    return stats.norm(loc=posterior.mean(), scale=math.sqrt(var))


def posterior_predictive_gamma_poisson(
    posterior: GammaPosterior,
) -> stats.rv_frozen:
    """Posterior predictive Negative Binomial.

    Returns ``scipy.stats.nbinom(n=alpha_n, p=beta_n/(beta_n+1))`` where
    :math:`n` is the shape (number of "successes") and :math:`p` is the
    success probability per trial. Mean
    :math:`\\alpha_n/\\beta_n`, variance
    :math:`\\alpha_n(\\beta_n+1)/\\beta_n^2`.
    """
    a, b = posterior.alpha, posterior.beta
    return stats.nbinom(n=a, p=b / (b + 1.0))


def posterior_predictive_beta_binomial(
    posterior: BetaPosterior, n_trials: int
) -> stats.rv_frozen:
    """Posterior predictive Beta-Binomial for a future batch of ``n_trials``."""
    if n_trials < 0:
        raise ValueError(f"n_trials must be >= 0; got {n_trials}")
    return stats.betabinom(n=int(n_trials), a=posterior.alpha, b=posterior.beta)


# =============================================================================
# Year-end forecast
# =============================================================================


@dataclass(frozen=True)
class YearEndForecast:
    """Predictive distribution of the annual total.

    Attributes
    ----------
    mean
        :math:`S_m + (12 - m)\\,\\mu_m`.
    variance
        :math:`(12-m)^2\\,\\sigma_m^2 + (12-m)\\,\\sigma^2`.
        Note the **quadratic** dependence on the horizon for the
        parameter-uncertainty term — see
        ``notes/phase4-predictive-inference.md`` §5.
    n_remaining
        Number of unobserved months :math:`12 - m`.
    observed_total
        :math:`S_m`, the sum of observed months.
    """

    mean: float
    variance: float
    n_remaining: int
    observed_total: float

    def std(self) -> float:
        return math.sqrt(self.variance)

    def frozen(self) -> stats.rv_frozen:
        return stats.norm(loc=self.mean, scale=self.std())

    def prob_above(self, threshold: float) -> float:
        """``P(T > threshold)``; uses ``norm.sf`` for numerical stability."""
        return float(self.frozen().sf(threshold))

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        if not 0 < level < 1:
            raise ValueError(f"level must be in (0, 1); got {level}")
        rv = self.frozen()
        tail = (1.0 - level) / 2.0
        return (float(rv.ppf(tail)), float(rv.ppf(1.0 - tail)))


def year_end_predictive_total(
    posterior: NormalPosterior,
    sigma_sq: float,
    n_remaining: int,
    observed_total: float = 0.0,
) -> YearEndForecast:
    """Predictive distribution of the annual total under Normal-Normal.

    Future months share the unknown :math:`\\theta`; correlation is
    captured by the quadratic term :math:`(12-m)^2 \\sigma_m^2` in the
    variance. The naïve "iid future months" formula —
    :math:`(12-m)(\\sigma_m^2 + \\sigma^2)` — would under-estimate the
    variance and is **not** what this function returns.
    """
    if sigma_sq <= 0:
        raise ValueError(f"sigma_sq must be > 0; got {sigma_sq}")
    if n_remaining < 0:
        raise ValueError(f"n_remaining must be >= 0; got {n_remaining}")
    h = float(n_remaining)
    mean = float(observed_total) + h * posterior.mean()
    variance = (h * h) * posterior.variance() + h * sigma_sq
    return YearEndForecast(
        mean=mean,
        variance=variance,
        n_remaining=int(n_remaining),
        observed_total=float(observed_total),
    )


def prob_over_budget(
    posterior: NormalPosterior,
    sigma_sq: float,
    n_remaining: int,
    budget_ceiling: float,
    observed_total: float = 0.0,
) -> float:
    """``P(annual total > budget_ceiling)`` under Normal-Normal."""
    forecast = year_end_predictive_total(
        posterior=posterior,
        sigma_sq=sigma_sq,
        n_remaining=n_remaining,
        observed_total=observed_total,
    )
    return forecast.prob_above(budget_ceiling)


# =============================================================================
# Monte Carlo posterior predictive sampling
# =============================================================================


def posterior_predictive_sample(
    posterior: NormalPosterior,
    sigma_sq: float,
    n_samples: int,
    n_periods: int = 1,
    seed: int | np.random.Generator | None = None,
    correlated: bool = True,
) -> NDArray[np.float64]:
    """Draw posterior predictive samples for ``n_periods`` future months.

    Implements the Article-1 Monte Carlo recipe applied to the posterior:

    1. For each replication ``s``, draw :math:`\\theta_s` from the
       posterior.
    2. For each period :math:`t`, draw
       :math:`\\tilde x_{s,t} \\sim N(\\theta_s, \\sigma^2)`.

    When ``correlated=True`` (the default and the only correct
    treatment for multi-period predictives), all periods within a
    replication share the same :math:`\\theta_s`; when ``False`` (only
    useful for didactic illustration of the trap), each
    :math:`\\tilde x_{s,t}` is drawn independently from the marginal
    predictive — see §5.1 of the theory notes.

    Parameters
    ----------
    posterior
        Normal posterior on :math:`\\theta`.
    sigma_sq
        Known sampling variance.
    n_samples
        Number of Monte Carlo replications :math:`S`.
    n_periods
        Number of future periods per replication. ``1`` returns a 1-D
        array; otherwise a ``(n_samples, n_periods)`` array.
    seed
        ``int`` or ``Generator``. ``None`` draws a fresh seed.
    correlated
        Preserve cross-period correlation by reusing one
        :math:`\\theta` per replication. Default ``True``.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0; got {n_samples}")
    if n_periods <= 0:
        raise ValueError(f"n_periods must be > 0; got {n_periods}")
    if sigma_sq <= 0:
        raise ValueError(f"sigma_sq must be > 0; got {sigma_sq}")

    rng = np.random.default_rng(seed)
    sigma = math.sqrt(sigma_sq)
    mu = posterior.mean()
    sd = posterior.std()

    if not correlated:
        marginal_var = posterior.variance() + sigma_sq
        samples_flat = rng.normal(mu, math.sqrt(marginal_var), size=n_samples * n_periods)
        if n_periods == 1:
            return samples_flat
        return samples_flat.reshape(n_samples, n_periods)

    thetas = rng.normal(mu, sd, size=n_samples)
    if n_periods == 1:
        return rng.normal(thetas, sigma)
    eps = rng.normal(0.0, sigma, size=(n_samples, n_periods))
    return thetas[:, None] + eps


# =============================================================================
# Bayes factor (Normal-Normal closed form)
# =============================================================================


def log_marginal_likelihood_normal_normal(
    data: NDArray[np.float64] | list[float],
    mu0: float,
    sigma0_sq: float,
    sigma_sq: float,
) -> float:
    """Log marginal likelihood of the Normal-Normal model.

    Closed-form expression derived from
    :math:`p(x) = \\int \\prod_i N(x_i \\mid \\theta, \\sigma^2)\\,N(\\theta \\mid \\mu_0, \\sigma_0^2)\\,\\mathrm d\\theta`.
    Evaluating the conjugate convolution gives

    .. math::

        \\log p(x_{1:n})
        = -\\tfrac{n}{2}\\log(2\\pi\\sigma^2)
          + \\tfrac{1}{2}\\log\\!\\Big(\\frac{\\sigma_n^2}{\\sigma_0^2}\\Big)
          - \\tfrac{1}{2\\sigma^2}\\sum (x_i - \\bar x)^2
          - \\tfrac{n}{2\\sigma_n^{-2}\\,\\sigma^2 / (n)}\\big(\\bar x - \\mu_0\\big)^2
            \\cdot \\tfrac{1}{1 + \\sigma^2/(n\\sigma_0^2)},

    or equivalently the more compact form below.
    """
    x = np.asarray(data, dtype=float).ravel()
    n = int(x.size)
    if n == 0:
        return 0.0
    if sigma0_sq <= 0 or sigma_sq <= 0:
        raise ValueError("variances must be positive.")
    x_bar = float(x.mean())
    sse = float(np.sum((x - x_bar) ** 2))
    tau0 = 1.0 / sigma0_sq
    tau = 1.0 / sigma_sq
    tau_n = tau0 + n * tau
    sigma_n_sq = 1.0 / tau_n

    # log p(x) = log [ (2π σ²)^{-n/2} · √(σ_n² / σ₀²) ]
    #           + [ -SSE/(2σ²) - (n τ₀ τ)/(2(τ₀+nτ)) (x̄ - μ₀)² ]
    # (the last bracket is just -(1/2) (μ₀-x̄)² / (σ₀² + σ²/n) · (n))
    log_norm = (
        -0.5 * n * math.log(2.0 * math.pi * sigma_sq)
        + 0.5 * (math.log(sigma_n_sq) - math.log(sigma0_sq))
    )
    quad = (
        -0.5 * sse / sigma_sq
        - 0.5 * (n * tau0 * tau / tau_n) * (x_bar - mu0) ** 2
    )
    return float(log_norm + quad)


def bayes_factor(log_marginal_a: float, log_marginal_b: float) -> float:
    """Bayes factor :math:`BF_{AB} = p(x \\mid M_A)/p(x \\mid M_B)`.

    Inputs are *log* marginal likelihoods to keep the computation
    numerically stable. Returns the Bayes factor on the natural scale.
    """
    return math.exp(log_marginal_a - log_marginal_b)


__all__ = [
    "YearEndForecast",
    "bayes_factor",
    "log_marginal_likelihood_normal_normal",
    "posterior_predictive_beta_binomial",
    "posterior_predictive_gamma_poisson",
    "posterior_predictive_normal",
    "posterior_predictive_sample",
    "prob_over_budget",
    "year_end_predictive_total",
]
