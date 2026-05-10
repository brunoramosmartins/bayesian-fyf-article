"""Applied FYF model — sequential Bayesian budget revision.

Wires together the conjugate Normal-Normal updater (``src.conjugate``),
the sequential engine (``src.updating``), and the year-end predictive
(``src.predictive``) into a single stateful object that walks one full
annual revision cycle.

Public API:

- :class:`FYFConfig`         — frozen dataclass of model parameters.
- :class:`MonthlyReview`     — record produced by ``process_month``.
- :class:`QuarterlyReview`   — record produced by ``fyf_review``.
- :class:`FYFModel`          — the stateful engine.
- :class:`Scenario`          — frozen dataclass for a named scenario.
- ``build_scenarios``        — five canonical scenarios with fixed seeds.

The default model is a 12-month annual cycle. Quarterly reviews fall on
months 3, 6, 9, 12. The cost decomposition follows
``docs/model-design.md`` §2; only the Normal-Normal layer is exercised
here (incident counts and overtime proportion stay as the conjugate
extensions described in Phase 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.conjugate import NormalNormalUpdater, NormalPosterior
from src.predictive import (
    YearEndForecast,
    posterior_predictive_normal,
    year_end_predictive_total,
)
from src.updating import SequentialUpdater

ANNUAL_HORIZON: int = 12
QUARTER_END_MONTHS: tuple[int, ...] = (3, 6, 9, 12)


# =============================================================================
# Configuration and review records
# =============================================================================


@dataclass(frozen=True)
class FYFConfig:
    """Hyperparameters for the Normal-Normal FYF model.

    Attributes
    ----------
    prior_mean
        Prior mean :math:`\\mu_0` (planned monthly cost in BRL).
    prior_sd
        Prior standard deviation :math:`\\sigma_0`. Must be positive.
    obs_sd
        Observation noise :math:`\\sigma`. Must be positive.
    budget_ceiling
        Annual budget ceiling :math:`B`. ``None`` to disable
        ``p_over_budget`` reporting.
    annual_horizon
        Number of months in the cycle. Defaults to 12.
    """

    prior_mean: float
    prior_sd: float
    obs_sd: float
    budget_ceiling: float | None = None
    annual_horizon: int = ANNUAL_HORIZON

    def __post_init__(self) -> None:
        if self.prior_sd <= 0:
            raise ValueError(f"prior_sd must be > 0; got {self.prior_sd}")
        if self.obs_sd <= 0:
            raise ValueError(f"obs_sd must be > 0; got {self.obs_sd}")
        if self.annual_horizon <= 0:
            raise ValueError(
                f"annual_horizon must be > 0; got {self.annual_horizon}"
            )


@dataclass(frozen=True)
class MonthlyReview:
    """One month of the cycle: actual, posterior, surprise, forecast."""

    month: int
    actual: float
    posterior: NormalPosterior
    surprise_z: float
    cumulative_actual: float
    forecast: YearEndForecast | None  # None at the final month
    p_over_budget: float | None
    is_quarter_end: bool


@dataclass(frozen=True)
class QuarterlyReview:
    """End-of-quarter summary tying posterior, forecast, and diagnostics."""

    quarter: int  # 1..4
    month_end: int  # 3, 6, 9, 12
    posterior: NormalPosterior
    forecast: YearEndForecast | None
    p_over_budget: float | None
    max_abs_surprise: float
    n_surprises_above_2: int
    recommendation: str


# =============================================================================
# The model
# =============================================================================


class FYFModel:
    """Stateful sequential Bayesian engine for the FYF cycle.

    Parameters
    ----------
    config
        :class:`FYFConfig` with prior, observation noise, and budget
        ceiling.
    """

    def __init__(self, config: FYFConfig) -> None:
        self.config = config
        prior = NormalNormalUpdater(
            mu0=config.prior_mean,
            sigma0_sq=config.prior_sd**2,
            sigma_sq=config.obs_sd**2,
        )
        self._engine = SequentialUpdater(prior)
        self._reviews: list[MonthlyReview] = []
        self._cumulative_actual: float = 0.0

    # ------------------------------------------------------------------
    # Per-month and per-quarter
    # ------------------------------------------------------------------

    def process_month(self, actual: float) -> MonthlyReview:
        """Consume one monthly actual; return the resulting review.

        Steps:

        1. Compute the surprise z-score using the *prior* posterior
           predictive (i.e. what the model expected before seeing
           ``actual``).
        2. Feed the actual through the sequential engine; obtain the
           updated posterior.
        3. Compute the year-end forecast and ``P(T > B)`` if a budget
           ceiling is set and any months remain.
        4. Append a :class:`MonthlyReview` to the history.
        """
        if self._engine.n_steps >= self.config.annual_horizon:
            raise RuntimeError(
                f"Annual horizon of {self.config.annual_horizon} months reached."
            )

        prior_post = self._engine.current_posterior()
        if not isinstance(prior_post, NormalPosterior):  # pragma: no cover
            raise TypeError("FYFModel expects Normal-Normal posteriors.")
        surprise_z = self._surprise(prior_post, actual)

        new_post = self._engine.feed(float(actual))
        if not isinstance(new_post, NormalPosterior):  # pragma: no cover
            raise TypeError("FYFModel expects Normal-Normal posteriors.")
        month = self._engine.n_steps
        self._cumulative_actual += float(actual)

        n_remaining = self.config.annual_horizon - month
        forecast: YearEndForecast | None = None
        p_over: float | None = None
        if n_remaining > 0:
            forecast = year_end_predictive_total(
                posterior=new_post,
                sigma_sq=self.config.obs_sd**2,
                n_remaining=n_remaining,
                observed_total=self._cumulative_actual,
            )
            if self.config.budget_ceiling is not None:
                p_over = forecast.prob_above(self.config.budget_ceiling)

        review = MonthlyReview(
            month=month,
            actual=float(actual),
            posterior=new_post,
            surprise_z=surprise_z,
            cumulative_actual=self._cumulative_actual,
            forecast=forecast,
            p_over_budget=p_over,
            is_quarter_end=(month in QUARTER_END_MONTHS),
        )
        self._reviews.append(review)
        return review

    def fyf_review(self, quarter: int) -> QuarterlyReview:
        """Produce the end-of-quarter review for ``quarter`` in 1..4."""
        if quarter not in (1, 2, 3, 4):
            raise ValueError(f"quarter must be in 1..4; got {quarter}")
        month_end = 3 * quarter
        if self._engine.n_steps < month_end:
            raise RuntimeError(
                f"Cannot review quarter {quarter}: only "
                f"{self._engine.n_steps} months consumed."
            )
        end_review = self._reviews[month_end - 1]
        quarter_window = self._reviews[month_end - 3 : month_end]
        max_abs_z = max(abs(r.surprise_z) for r in quarter_window)
        n_above_2 = sum(1 for r in quarter_window if abs(r.surprise_z) > 2.0)
        recommendation = self._recommendation(end_review, max_abs_z, n_above_2)
        return QuarterlyReview(
            quarter=quarter,
            month_end=month_end,
            posterior=end_review.posterior,
            forecast=end_review.forecast,
            p_over_budget=end_review.p_over_budget,
            max_abs_surprise=max_abs_z,
            n_surprises_above_2=n_above_2,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # Bulk convenience
    # ------------------------------------------------------------------

    def annual_cycle(self, monthly_actuals: list[float]) -> list[MonthlyReview]:
        """Feed all 12 actuals; return the full list of monthly reviews."""
        if len(monthly_actuals) != self.config.annual_horizon:
            raise ValueError(
                f"Expected {self.config.annual_horizon} actuals; "
                f"got {len(monthly_actuals)}."
            )
        for x in monthly_actuals:
            self.process_month(float(x))
        return list(self._reviews)

    def reviews(self) -> list[MonthlyReview]:
        """Return a shallow copy of the monthly review history."""
        return list(self._reviews)

    def quarterly_reviews(self) -> list[QuarterlyReview]:
        """Quarterly reviews for every completed quarter."""
        return [
            self.fyf_review(q)
            for q in (1, 2, 3, 4)
            if self._engine.n_steps >= 3 * q
        ]

    def reset(self) -> None:
        """Return the model to its initial state."""
        self._engine.reset()
        self._reviews.clear()
        self._cumulative_actual = 0.0

    def current_posterior(self) -> NormalPosterior:
        """Return the latest posterior, or the prior before any data."""
        post = self._engine.current_posterior()
        if not isinstance(post, NormalPosterior):  # pragma: no cover
            raise TypeError("FYFModel expects Normal-Normal posteriors.")
        return post

    @property
    def n_months_processed(self) -> int:
        return self._engine.n_steps

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _surprise(self, prior_post: NormalPosterior, actual: float) -> float:
        pred_var = prior_post.variance() + self.config.obs_sd**2
        return float((actual - prior_post.mean()) / math.sqrt(pred_var))

    def _recommendation(
        self,
        end_review: MonthlyReview,
        max_abs_z: float,
        n_above_2: int,
    ) -> str:
        if max_abs_z > 3.0:
            return "investigate shock"
        if n_above_2 >= 2:
            return "re-elicit prior or model"
        if (
            end_review.p_over_budget is not None
            and end_review.p_over_budget > 0.50
        ):
            return "request budget revision"
        return "hold"


# =============================================================================
# Scenarios
# =============================================================================


@dataclass(frozen=True)
class Scenario:
    """A named simulated annual cycle."""

    id: str
    name: str
    description: str
    actuals: tuple[float, ...]
    config: FYFConfig
    secondary_config: FYFConfig | None = field(default=None)
    seed: int | None = None


def _default_config() -> FYFConfig:
    return FYFConfig(
        prior_mean=1_050_000.0,
        prior_sd=150_000.0,
        obs_sd=80_000.0,
        budget_ceiling=13_200_000.0,
    )


def _scenario_on_target(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    config = _default_config()
    actuals = rng.normal(config.prior_mean, config.obs_sd, size=12)
    return Scenario(
        id="s1_on_target",
        name="On-target",
        description="Plan is accurate in expectation; posterior confirms it.",
        actuals=tuple(float(x) for x in actuals),
        config=config,
        seed=seed,
    )


def _scenario_optimistic(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    config = _default_config()
    delta = 50_000.0
    actuals = rng.normal(config.prior_mean + delta, config.obs_sd, size=12)
    return Scenario(
        id="s2_optimistic",
        name="Optimistic plan",
        description="Plan systematically underestimates; posterior corrects upward.",
        actuals=tuple(float(x) for x in actuals),
        config=config,
        seed=seed,
    )


def _scenario_shock(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    config = _default_config()
    actuals = list(rng.normal(config.prior_mean, config.obs_sd, size=4))
    actuals.append(1_400_000.0)  # the shock at month 5
    actuals.extend(rng.normal(config.prior_mean + 50_000.0, config.obs_sd, size=7))
    return Scenario(
        id="s3_shock",
        name="Budget shock",
        description="A one-off +4σ event at month 5 followed by a step-up.",
        actuals=tuple(float(x) for x in actuals),
        config=config,
        seed=seed,
    )


def _scenario_seasonal(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    config = _default_config()
    seasonal = np.array(
        [0, 0, 0, 0, 0, 0, -100_000, -150_000, -100_000, 100_000, 150_000, 200_000],
        dtype=float,
    )
    noise = rng.normal(0.0, config.obs_sd, size=12)
    actuals = config.prior_mean + seasonal + noise
    return Scenario(
        id="s4_seasonal",
        name="Seasonal variation",
        description="Q3 below baseline, Q4 above; single-θ model cannot resolve.",
        actuals=tuple(float(x) for x in actuals),
        config=config,
        seed=seed,
    )


def _scenario_prior_sensitivity(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    base_config = _default_config()
    actuals = rng.normal(base_config.prior_mean, base_config.obs_sd, size=12)
    confident = FYFConfig(
        prior_mean=base_config.prior_mean,
        prior_sd=100_000.0,
        obs_sd=base_config.obs_sd,
        budget_ceiling=base_config.budget_ceiling,
    )
    uncertain = FYFConfig(
        prior_mean=base_config.prior_mean,
        prior_sd=300_000.0,
        obs_sd=base_config.obs_sd,
        budget_ceiling=base_config.budget_ceiling,
    )
    return Scenario(
        id="s5_prior_sensitivity",
        name="Prior sensitivity",
        description="Same data; confident vs uncertain prior trajectories.",
        actuals=tuple(float(x) for x in actuals),
        config=confident,
        secondary_config=uncertain,
        seed=seed,
    )


def build_scenarios(base_seed: int = 20260901) -> list[Scenario]:
    """Return the five canonical Phase-5 scenarios with fixed seeds."""
    return [
        _scenario_on_target(base_seed + 1),
        _scenario_optimistic(base_seed + 2),
        _scenario_shock(base_seed + 3),
        _scenario_seasonal(base_seed + 4),
        _scenario_prior_sensitivity(base_seed + 5),
    ]


def run_scenario(scenario: Scenario) -> FYFModel:
    """Build an :class:`FYFModel` and run ``scenario.actuals`` through it."""
    model = FYFModel(scenario.config)
    model.annual_cycle(list(scenario.actuals))
    return model


def run_scenario_secondary(scenario: Scenario) -> FYFModel | None:
    """Run the secondary configuration of ``scenario`` (used by S5)."""
    if scenario.secondary_config is None:
        return None
    model = FYFModel(scenario.secondary_config)
    model.annual_cycle(list(scenario.actuals))
    return model


# =============================================================================
# Helper: predictive next-month from the current posterior
# =============================================================================


def next_month_predictive(model: FYFModel) -> tuple[float, float]:
    """Mean and s.d. of the posterior predictive for ``month_n+1``."""
    post = model.current_posterior()
    pred = posterior_predictive_normal(post, sigma_sq=model.config.obs_sd**2)
    return float(pred.mean()), float(pred.std())


__all__ = [
    "ANNUAL_HORIZON",
    "QUARTER_END_MONTHS",
    "FYFConfig",
    "FYFModel",
    "MonthlyReview",
    "QuarterlyReview",
    "Scenario",
    "build_scenarios",
    "next_month_predictive",
    "run_scenario",
    "run_scenario_secondary",
]
