"""Conjugate updating for the four families used in the FYF model.

Each family is a pair *(Updater, Posterior)*:

- ``NormalNormalUpdater`` / ``NormalPosterior``
- ``NormalInverseGammaUpdater`` / ``NormalInverseGammaPosterior``
- ``GammaPoissonUpdater`` / ``GammaPosterior``
- ``BetaBinomialUpdater`` / ``BetaPosterior``

Updaters are constructed from prior hyperparameters; ``.update(data)``
returns the corresponding posterior in closed form. Posteriors are
frozen dataclasses with a uniform read-only interface:
``.mean()``, ``.variance()``, ``.std()``, ``.credible_interval(level)``,
``.summary()``.

The closed-form derivations live in
``notes/phase2-conjugate-families.md``; this module is a thin numerical
layer on top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

# =============================================================================
# Posteriors
# =============================================================================


@dataclass(frozen=True)
class NormalPosterior:
    """Normal posterior on a mean parameter (known sampling variance).

    Attributes
    ----------
    mu
        Posterior mean :math:`\\mu_n`.
    sigma_sq
        Posterior variance :math:`\\sigma_n^2`.
    n_obs
        Number of observations consumed to produce the posterior. Useful
        for downstream diagnostics; carries no information that affects
        ``mean``/``variance``.
    """

    mu: float
    sigma_sq: float
    n_obs: int = 0

    def mean(self) -> float:
        return float(self.mu)

    def variance(self) -> float:
        return float(self.sigma_sq)

    def std(self) -> float:
        return math.sqrt(self.sigma_sq)

    def precision(self) -> float:
        """Posterior precision :math:`\\tau_n = 1/\\sigma_n^2`."""
        return 1.0 / self.sigma_sq

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        _check_level(level)
        z = stats.norm.ppf(0.5 + level / 2.0)
        half = z * self.std()
        return (self.mu - half, self.mu + half)

    def summary(self, level: float = 0.95) -> dict[str, float | tuple[float, float]]:
        return {
            "mean": self.mean(),
            "variance": self.variance(),
            "std": self.std(),
            "precision": self.precision(),
            "credible_interval": self.credible_interval(level),
            "credible_level": float(level),
            "n_obs": int(self.n_obs),
        }


@dataclass(frozen=True)
class GammaPosterior:
    """Gamma posterior on a positive rate (rate parameterisation).

    The Gamma is parameterised as
    :math:`\\pi(\\lambda) \\propto \\lambda^{\\alpha-1} e^{-\\beta\\lambda}`
    so that the mean is :math:`\\alpha/\\beta`.
    """

    alpha: float
    beta: float  # rate
    n_obs: int = 0

    def mean(self) -> float:
        return float(self.alpha / self.beta)

    def variance(self) -> float:
        return float(self.alpha / (self.beta**2))

    def std(self) -> float:
        return math.sqrt(self.variance())

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        _check_level(level)
        rv = stats.gamma(a=self.alpha, scale=1.0 / self.beta)
        tail = (1.0 - level) / 2.0
        return (float(rv.ppf(tail)), float(rv.ppf(1.0 - tail)))

    def summary(self, level: float = 0.95) -> dict[str, float | tuple[float, float]]:
        return {
            "mean": self.mean(),
            "variance": self.variance(),
            "std": self.std(),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "credible_interval": self.credible_interval(level),
            "credible_level": float(level),
            "n_obs": int(self.n_obs),
        }


@dataclass(frozen=True)
class BetaPosterior:
    """Beta posterior on a proportion."""

    alpha: float
    beta: float
    n_trials: int = 0

    def mean(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return float((a * b) / (((a + b) ** 2) * (a + b + 1)))

    def std(self) -> float:
        return math.sqrt(self.variance())

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        _check_level(level)
        rv = stats.beta(a=self.alpha, b=self.beta)
        tail = (1.0 - level) / 2.0
        return (float(rv.ppf(tail)), float(rv.ppf(1.0 - tail)))

    def summary(self, level: float = 0.95) -> dict[str, float | tuple[float, float]]:
        return {
            "mean": self.mean(),
            "variance": self.variance(),
            "std": self.std(),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "credible_interval": self.credible_interval(level),
            "credible_level": float(level),
            "n_trials": int(self.n_trials),
        }


@dataclass(frozen=True)
class NormalInverseGammaPosterior:
    """Joint Normal-Inverse-Gamma posterior on ``(theta, sigma^2)``.

    The marginal posterior of ``theta`` is a non-standard Student-t with
    location ``mu``, scale :math:`\\sqrt{\\beta/(\\alpha\\kappa)}`, and
    :math:`2\\alpha` degrees of freedom. The marginal of ``sigma^2`` is
    :math:`\\text{Inverse-Gamma}(\\alpha,\\beta)`, mean
    :math:`\\beta/(\\alpha-1)` for :math:`\\alpha>1`.

    The default ``mean()``/``variance()``/``credible_interval()`` methods
    refer to the **marginal of theta** (the parameter of primary
    interest in the FYF model). Use ``mean_sigma_sq()`` for the variance
    parameter.
    """

    mu: float
    kappa: float
    alpha: float
    beta: float
    n_obs: int = 0

    # ---- theta marginal (Student-t) ----
    def _t_scale(self) -> float:
        return math.sqrt(self.beta / (self.alpha * self.kappa))

    def _t_df(self) -> float:
        return 2.0 * self.alpha

    def mean(self) -> float:
        # E[theta] = mu provided df > 1, i.e. alpha > 0.5.
        if self._t_df() <= 1.0:
            raise ValueError(
                "theta marginal mean undefined for 2*alpha <= 1."
            )
        return float(self.mu)

    def variance(self) -> float:
        df = self._t_df()
        if df <= 2.0:
            raise ValueError(
                "theta marginal variance undefined for 2*alpha <= 2."
            )
        scale = self._t_scale()
        return float(scale * scale * df / (df - 2.0))

    def std(self) -> float:
        return math.sqrt(self.variance())

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        _check_level(level)
        rv = stats.t(df=self._t_df(), loc=self.mu, scale=self._t_scale())
        tail = (1.0 - level) / 2.0
        return (float(rv.ppf(tail)), float(rv.ppf(1.0 - tail)))

    # ---- sigma^2 marginal (Inverse-Gamma) ----
    def mean_sigma_sq(self) -> float:
        if self.alpha <= 1.0:
            raise ValueError(
                "Inverse-Gamma mean undefined for alpha <= 1."
            )
        return float(self.beta / (self.alpha - 1.0))

    def variance_sigma_sq(self) -> float:
        if self.alpha <= 2.0:
            raise ValueError(
                "Inverse-Gamma variance undefined for alpha <= 2."
            )
        a, b = self.alpha, self.beta
        return float((b * b) / (((a - 1.0) ** 2) * (a - 2.0)))

    def summary(self, level: float = 0.95) -> dict[str, float | tuple[float, float]]:
        ci = self.credible_interval(level)
        out: dict[str, float | tuple[float, float]] = {
            "mu": float(self.mu),
            "kappa": float(self.kappa),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "theta_mean": self.mean(),
            "theta_credible_interval": ci,
            "credible_level": float(level),
            "n_obs": int(self.n_obs),
        }
        # Sigma^2 marginal — only defined for alpha > 1.
        if self.alpha > 1.0:
            out["sigma_sq_mean"] = self.mean_sigma_sq()
        return out


# =============================================================================
# Updaters
# =============================================================================


@dataclass(frozen=True)
class NormalNormalUpdater:
    """Closed-form Bayesian updating for the Normal-Normal model.

    Prior :math:`\\theta \\sim N(\\mu_0, \\sigma_0^2)`, likelihood
    :math:`x_i \\mid \\theta \\sim N(\\theta, \\sigma^2)` with
    :math:`\\sigma^2` known. The posterior is

    .. math::

        \\theta \\mid x_{1:n} \\sim N(\\mu_n, \\sigma_n^2),
        \\quad
        \\tau_n = \\tau_0 + n\\tau,
        \\quad
        \\mu_n = \\frac{\\tau_0 \\mu_0 + n\\tau\\bar x}{\\tau_n}.

    Parameters
    ----------
    mu0
        Prior mean :math:`\\mu_0`.
    sigma0_sq
        Prior variance :math:`\\sigma_0^2`. Must be strictly positive.
    sigma_sq
        Known sampling variance :math:`\\sigma^2`. Must be strictly
        positive.
    """

    mu0: float
    sigma0_sq: float
    sigma_sq: float

    def __post_init__(self) -> None:
        if self.sigma0_sq <= 0:
            raise ValueError("sigma0_sq must be > 0.")
        if self.sigma_sq <= 0:
            raise ValueError("sigma_sq must be > 0.")

    def update(self, data: ArrayLike) -> NormalPosterior:
        """Return the posterior after observing ``data``.

        Parameters
        ----------
        data
            Array-like of monthly observations. Empty arrays are
            allowed and return the prior unchanged (as a posterior with
            ``n_obs=0``).
        """
        x = np.asarray(data, dtype=float).ravel()
        n = int(x.size)
        tau0 = 1.0 / self.sigma0_sq
        tau = 1.0 / self.sigma_sq

        if n == 0:
            return NormalPosterior(mu=float(self.mu0), sigma_sq=float(self.sigma0_sq))

        x_bar = float(x.mean())
        tau_n = tau0 + n * tau
        mu_n = (tau0 * self.mu0 + n * tau * x_bar) / tau_n
        sigma_n_sq = 1.0 / tau_n
        return NormalPosterior(mu=float(mu_n), sigma_sq=float(sigma_n_sq), n_obs=n)


@dataclass(frozen=True)
class GammaPoissonUpdater:
    """Closed-form Bayesian updating for the Gamma-Poisson model.

    Prior :math:`\\lambda \\sim \\text{Gamma}(\\alpha_0, \\beta_0)` (rate
    parameterisation), likelihood
    :math:`x_i \\mid \\lambda \\sim \\text{Poisson}(\\lambda)`. Posterior:

    .. math::

        \\lambda \\mid x_{1:n} \\sim
        \\text{Gamma}\\big(\\alpha_0 + \\textstyle\\sum x_i,\\; \\beta_0 + n\\big).
    """

    alpha0: float
    beta0: float

    def __post_init__(self) -> None:
        if self.alpha0 <= 0:
            raise ValueError("alpha0 must be > 0.")
        if self.beta0 <= 0:
            raise ValueError("beta0 must be > 0.")

    def update(self, data: ArrayLike) -> GammaPosterior:
        x = np.asarray(data).ravel()
        n = int(x.size)

        if n == 0:
            return GammaPosterior(alpha=float(self.alpha0), beta=float(self.beta0))

        if not np.all(x >= 0):
            raise ValueError("Poisson counts must be non-negative.")
        if not np.all(np.equal(np.mod(x, 1.0), 0)):
            raise ValueError("Poisson counts must be integers.")

        s = float(x.sum())
        return GammaPosterior(
            alpha=float(self.alpha0 + s),
            beta=float(self.beta0 + n),
            n_obs=n,
        )


@dataclass(frozen=True)
class BetaBinomialUpdater:
    """Closed-form Bayesian updating for the Beta-Binomial model.

    Prior :math:`p \\sim \\text{Beta}(\\alpha_0, \\beta_0)`, likelihood
    :math:`x \\mid p \\sim \\text{Binomial}(n, p)`. Posterior:

    .. math::

        p \\mid x \\sim \\text{Beta}(\\alpha_0 + x,\\; \\beta_0 + n - x).

    The ``update`` method accepts either a single ``(successes, trials)``
    pair or arrays of independent batches.
    """

    alpha0: float
    beta0: float

    def __post_init__(self) -> None:
        if self.alpha0 <= 0:
            raise ValueError("alpha0 must be > 0.")
        if self.beta0 <= 0:
            raise ValueError("beta0 must be > 0.")

    def update(self, successes: ArrayLike, trials: ArrayLike) -> BetaPosterior:
        s = np.asarray(successes).ravel()
        t = np.asarray(trials).ravel()
        if s.shape != t.shape:
            raise ValueError("successes and trials must have the same shape.")
        if np.any(s < 0) or np.any(t < 0):
            raise ValueError("successes and trials must be non-negative.")
        if np.any(s > t):
            raise ValueError("successes cannot exceed trials.")

        total_succ = int(s.sum())
        total_trials = int(t.sum())
        return BetaPosterior(
            alpha=float(self.alpha0 + total_succ),
            beta=float(self.beta0 + total_trials - total_succ),
            n_trials=total_trials,
        )


@dataclass(frozen=True)
class NormalInverseGammaUpdater:
    """Closed-form Bayesian updating for the Normal-Inverse-Gamma model.

    Prior :math:`(\\theta, \\sigma^2) \\sim N\\text{-}IG(\\mu_0,
    \\kappa_0, \\alpha_0, \\beta_0)` parameterised so that
    :math:`\\sigma^2 \\sim \\text{Inverse-Gamma}(\\alpha_0, \\beta_0)`
    and :math:`\\theta \\mid \\sigma^2 \\sim N(\\mu_0, \\sigma^2/\\kappa_0)`.

    Likelihood :math:`x_i \\mid \\theta, \\sigma^2 \\sim N(\\theta, \\sigma^2)`.
    Posterior hyperparameters (see ``notes/phase2-conjugate-families.md``):

    .. math::

        \\mu_n &= \\frac{\\kappa_0 \\mu_0 + n \\bar x}{\\kappa_0 + n},
        \\qquad \\kappa_n = \\kappa_0 + n, \\\\
        \\alpha_n &= \\alpha_0 + n/2, \\\\
        \\beta_n &= \\beta_0 + \\tfrac{1}{2} S
            + \\tfrac{1}{2}\\frac{\\kappa_0 n}{\\kappa_0 + n}(\\bar x - \\mu_0)^2,

    with :math:`S = \\sum (x_i - \\bar x)^2`.
    """

    mu0: float
    kappa0: float
    alpha0: float
    beta0: float

    def __post_init__(self) -> None:
        if self.kappa0 <= 0:
            raise ValueError("kappa0 must be > 0.")
        if self.alpha0 <= 0:
            raise ValueError("alpha0 must be > 0.")
        if self.beta0 <= 0:
            raise ValueError("beta0 must be > 0.")

    def update(self, data: ArrayLike) -> NormalInverseGammaPosterior:
        x = np.asarray(data, dtype=float).ravel()
        n = int(x.size)

        if n == 0:
            return NormalInverseGammaPosterior(
                mu=float(self.mu0),
                kappa=float(self.kappa0),
                alpha=float(self.alpha0),
                beta=float(self.beta0),
            )

        x_bar = float(x.mean())
        # Corrected sum of squares about the sample mean. ddof=0 in numpy.
        sse = float(np.sum((x - x_bar) ** 2))

        kappa_n = self.kappa0 + n
        mu_n = (self.kappa0 * self.mu0 + n * x_bar) / kappa_n
        alpha_n = self.alpha0 + n / 2.0
        beta_n = (
            self.beta0
            + 0.5 * sse
            + 0.5 * (self.kappa0 * n / kappa_n) * (x_bar - self.mu0) ** 2
        )
        return NormalInverseGammaPosterior(
            mu=float(mu_n),
            kappa=float(kappa_n),
            alpha=float(alpha_n),
            beta=float(beta_n),
            n_obs=n,
        )


# =============================================================================
# Helpers
# =============================================================================


def _check_level(level: float) -> None:
    if not 0 < level < 1:
        raise ValueError(f"credible level must be in (0, 1); got {level}")


__all__ = [
    "BetaBinomialUpdater",
    "BetaPosterior",
    "GammaPoissonUpdater",
    "GammaPosterior",
    "NormalInverseGammaPosterior",
    "NormalInverseGammaUpdater",
    "NormalNormalUpdater",
    "NormalPosterior",
]
