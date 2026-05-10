"""Sequential Bayesian updating across the FYF cycle.

Wraps the conjugate updaters from ``src.conjugate`` into a stateful
engine that consumes one observation at a time, stores the full
trajectory of posteriors, and exposes the shrinkage weight at each
step.

The design is type-dispatched on the underlying conjugate updater:

- :class:`NormalNormalUpdater` — observation is a single ``float``.
- :class:`GammaPoissonUpdater` — observation is a single integer count.
- :class:`BetaBinomialUpdater` — observation is a ``(successes, trials)`` tuple.
- :class:`NormalInverseGammaUpdater` — observation is a single ``float``.

The "sequential = batch" theorem (``notes/phase3-sequential-updating.md``)
guarantees the trajectory ends at the same posterior as
``conjugate_updater.update(all_observations)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.conjugate import (
    BetaBinomialUpdater,
    BetaPosterior,
    GammaPoissonUpdater,
    GammaPosterior,
    NormalInverseGammaPosterior,
    NormalInverseGammaUpdater,
    NormalNormalUpdater,
    NormalPosterior,
)

ConjugateUpdater = (
    NormalNormalUpdater
    | GammaPoissonUpdater
    | BetaBinomialUpdater
    | NormalInverseGammaUpdater
)
ConjugatePosterior = (
    NormalPosterior | GammaPosterior | BetaPosterior | NormalInverseGammaPosterior
)


@dataclass
class HistoryEntry:
    """One step of the sequential trajectory.

    Attributes
    ----------
    step
        1-indexed step number (``step == k`` means "after observing the
        :math:`k`-th observation"). Step 0 is the prior and is stored
        in :attr:`SequentialUpdater._prior_entry`.
    observation
        The observation consumed at this step. Type depends on the
        underlying conjugate updater (see module docstring).
    posterior
        The posterior immediately after the update.
    prior_weight
        Share of the prior in the posterior mean *of this step*. For
        Normal-Normal this is the canonical shrinkage weight
        :math:`w_0(n) = \\tau_0/(\\tau_0 + n\\tau)`. For other families
        it is the analogous prior-strength share — see
        :meth:`SequentialUpdater._prior_weight_after`.
    """

    step: int
    observation: Any
    posterior: ConjugatePosterior
    prior_weight: float


class SequentialUpdater:
    """Stateful sequential Bayesian updating engine.

    Parameters
    ----------
    conjugate_updater
        A conjugate updater from ``src.conjugate`` whose hyperparameters
        define the prior. The engine treats this prior as step 0 and
        maintains the latest posterior internally.

    Notes
    -----
    The engine is **stateful**: ``feed`` mutates the internal posterior.
    Use ``reset()`` to return to the prior. The original conjugate
    updater is preserved unchanged on the instance attribute
    :attr:`prior_updater`.
    """

    def __init__(self, conjugate_updater: ConjugateUpdater) -> None:
        if not isinstance(
            conjugate_updater,
            (
                NormalNormalUpdater,
                GammaPoissonUpdater,
                BetaBinomialUpdater,
                NormalInverseGammaUpdater,
            ),
        ):
            raise TypeError(
                f"Unsupported conjugate updater: {type(conjugate_updater).__name__}"
            )
        self.prior_updater: ConjugateUpdater = conjugate_updater
        self._history: list[HistoryEntry] = []
        # The "current updater" is reconstructed at each step from the
        # latest posterior so that single-observation updates have the
        # current posterior playing the role of prior.
        self._current_updater: ConjugateUpdater = conjugate_updater

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, observation: Any) -> ConjugatePosterior:
        """Consume a single observation; return the new posterior."""
        post = _update_single(self._current_updater, observation)
        step = len(self._history) + 1
        entry = HistoryEntry(
            step=step,
            observation=observation,
            posterior=post,
            prior_weight=float("nan"),  # filled in below
        )
        # Append first so _prior_weight_after sees up-to-date trial counts
        # (relevant for Beta-Binomial, harmless for the other families).
        self._history.append(entry)
        entry.prior_weight = self._prior_weight_after(step)
        self._current_updater = _next_updater(self._current_updater, post)
        return post

    def feed_batch(self, observations: list[Any]) -> ConjugatePosterior:
        """Sequentially consume each observation; return the final posterior.

        ``feed_batch([x_1, ..., x_n])`` is exactly equivalent to calling
        ``feed(x_k)`` in a loop. By the sequential = batch theorem
        (Phase 3) the result also equals
        ``prior_updater.update([x_1, ..., x_n])``.
        """
        last: ConjugatePosterior | None = None
        for obs in observations:
            last = self.feed(obs)
        if last is None:
            return self.current_posterior()
        return last

    def history(self) -> list[HistoryEntry]:
        """Read-only view of every step's posterior and weight."""
        return list(self._history)

    def shrinkage_weights(self) -> list[float]:
        """Prior weight at each step (smaller = data dominates more)."""
        return [entry.prior_weight for entry in self._history]

    def current_posterior(self) -> ConjugatePosterior:
        """Latest posterior, or the prior if no observation has been fed."""
        if not self._history:
            return _prior_posterior(self.prior_updater)
        return self._history[-1].posterior

    def reset(self) -> None:
        """Return the engine to its starting state (prior, no history)."""
        self._history.clear()
        self._current_updater = self.prior_updater

    @property
    def n_steps(self) -> int:
        return len(self._history)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prior_weight_after(self, n: int) -> float:
        """Analytic prior weight after ``n`` observations.

        For each conjugate family this is the share contributed by the
        *original* prior to the posterior mean / proportion / rate after
        ``n`` total updates. Defined in closed form.
        """
        upd = self.prior_updater
        if isinstance(upd, NormalNormalUpdater):
            tau0 = 1.0 / upd.sigma0_sq
            tau = 1.0 / upd.sigma_sq
            return float(tau0 / (tau0 + n * tau))
        if isinstance(upd, NormalInverseGammaUpdater):
            return float(upd.kappa0 / (upd.kappa0 + n))
        if isinstance(upd, GammaPoissonUpdater):
            return float(upd.beta0 / (upd.beta0 + n))
        if isinstance(upd, BetaBinomialUpdater):
            # n here counts trials, not batches; the relevant pseudo-N is
            # alpha0 + beta0 vs. cumulative trials so far.
            n_trials_total = self._cumulative_trials_at(n)
            return float(
                (upd.alpha0 + upd.beta0)
                / (upd.alpha0 + upd.beta0 + n_trials_total)
            )
        raise TypeError(  # pragma: no cover — guarded by __init__
            f"Unhandled updater type: {type(upd).__name__}"
        )

    def _cumulative_trials_at(self, step: int) -> int:
        total = 0
        for entry in self._history[:step]:
            obs = entry.observation
            if isinstance(obs, tuple) and len(obs) == 2:
                total += int(obs[1])
            else:  # pragma: no cover — guarded by feed
                raise TypeError(
                    "Beta-Binomial step expected a (successes, trials) tuple."
                )
        # If the call asks about a step we have already processed, this
        # is the cumulative trials *after* that step. If the call asks
        # for a step we have just processed, the latest entry is already
        # appended in `feed`, so this is the right total.
        return total


# =============================================================================
# Type-dispatched single-observation update
# =============================================================================


def _update_single(updater: ConjugateUpdater, observation: Any) -> ConjugatePosterior:
    if isinstance(updater, NormalNormalUpdater):
        return updater.update([float(observation)])
    if isinstance(updater, NormalInverseGammaUpdater):
        return updater.update([float(observation)])
    if isinstance(updater, GammaPoissonUpdater):
        return updater.update([int(observation)])
    if isinstance(updater, BetaBinomialUpdater):
        if not (isinstance(observation, tuple) and len(observation) == 2):
            raise TypeError(
                "Beta-Binomial expects a (successes, trials) tuple."
            )
        s, t = observation
        return updater.update(successes=int(s), trials=int(t))
    raise TypeError(  # pragma: no cover
        f"Unhandled updater type: {type(updater).__name__}"
    )


# =============================================================================
# Build a "next-step updater" whose prior is the current posterior
# =============================================================================


def _next_updater(
    current: ConjugateUpdater, posterior: ConjugatePosterior
) -> ConjugateUpdater:
    if isinstance(current, NormalNormalUpdater) and isinstance(
        posterior, NormalPosterior
    ):
        return NormalNormalUpdater(
            mu0=posterior.mu,
            sigma0_sq=posterior.sigma_sq,
            sigma_sq=current.sigma_sq,
        )
    if isinstance(current, NormalInverseGammaUpdater) and isinstance(
        posterior, NormalInverseGammaPosterior
    ):
        return NormalInverseGammaUpdater(
            mu0=posterior.mu,
            kappa0=posterior.kappa,
            alpha0=posterior.alpha,
            beta0=posterior.beta,
        )
    if isinstance(current, GammaPoissonUpdater) and isinstance(
        posterior, GammaPosterior
    ):
        return GammaPoissonUpdater(alpha0=posterior.alpha, beta0=posterior.beta)
    if isinstance(current, BetaBinomialUpdater) and isinstance(
        posterior, BetaPosterior
    ):
        return BetaBinomialUpdater(alpha0=posterior.alpha, beta0=posterior.beta)
    raise TypeError(  # pragma: no cover
        f"Type mismatch between updater {type(current).__name__} "
        f"and posterior {type(posterior).__name__}."
    )


# =============================================================================
# Build the prior-as-posterior (used when no observations have been fed yet)
# =============================================================================


def _prior_posterior(updater: ConjugateUpdater) -> ConjugatePosterior:
    if isinstance(updater, NormalNormalUpdater):
        return NormalPosterior(mu=updater.mu0, sigma_sq=updater.sigma0_sq, n_obs=0)
    if isinstance(updater, NormalInverseGammaUpdater):
        return NormalInverseGammaPosterior(
            mu=updater.mu0,
            kappa=updater.kappa0,
            alpha=updater.alpha0,
            beta=updater.beta0,
            n_obs=0,
        )
    if isinstance(updater, GammaPoissonUpdater):
        return GammaPosterior(alpha=updater.alpha0, beta=updater.beta0, n_obs=0)
    if isinstance(updater, BetaBinomialUpdater):
        return BetaPosterior(alpha=updater.alpha0, beta=updater.beta0, n_trials=0)
    raise TypeError(  # pragma: no cover
        f"Unhandled updater type: {type(updater).__name__}"
    )


# =============================================================================
# Standalone helpers for analysis (no engine state required)
# =============================================================================


def normal_normal_shrinkage_weight(
    sigma0: float, sigma: float, n: int | np.ndarray
) -> float | np.ndarray:
    """Closed-form shrinkage weight ``w_0(n) = tau_0 / (tau_0 + n*tau)``.

    Defined for the Normal-Normal model.

    Parameters
    ----------
    sigma0
        Prior standard deviation. Must be positive.
    sigma
        Sampling standard deviation. Must be positive.
    n
        Number of observations. Scalar or array-like.

    Returns
    -------
    Same dtype/shape as ``n``.
    """
    if sigma0 <= 0 or sigma <= 0:
        raise ValueError("sigma0 and sigma must be positive.")
    ratio = (sigma * sigma) / (sigma0 * sigma0)
    return ratio / (ratio + n)


def months_to_data_weight(
    sigma0: float, sigma: float, target_data_weight: float
) -> int:
    """Smallest integer ``n`` with ``1 - w_0(n) >= target_data_weight``.

    Solves ``n >= c/(1-c) * sigma**2/sigma0**2`` and returns the ceiling.
    """
    if not 0 < target_data_weight < 1:
        raise ValueError(
            f"target_data_weight must be in (0, 1); got {target_data_weight}"
        )
    c = target_data_weight
    threshold = (c / (1.0 - c)) * (sigma * sigma) / (sigma0 * sigma0)
    return int(np.ceil(threshold))


__all__ = [
    "HistoryEntry",
    "SequentialUpdater",
    "months_to_data_weight",
    "normal_normal_shrinkage_weight",
]
