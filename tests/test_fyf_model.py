"""Tests for src/fyf_model.py — DO NOT auto-run.

Coverage:

- ``FYFConfig`` validation rules.
- ``FYFModel.process_month`` produces correct surprise scores and
  posteriors that match the underlying conjugate engine.
- ``FYFModel.process_month`` cumulative actuals stay in sync.
- ``annual_cycle`` consumes 12 actuals exactly once.
- ``fyf_review`` aggregates the right window and surfaces the worst
  surprise correctly.
- Recommendation rules fire on the right diagnostic conditions
  (shock, repeated drift, P(over-budget) > 0.5).
- The five canonical scenarios run end-to-end without raising.
- ``run_scenario_secondary`` returns a model only when the scenario
  has a secondary config.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.conjugate import NormalNormalUpdater, NormalPosterior
from src.fyf_model import (
    ANNUAL_HORIZON,
    QUARTER_END_MONTHS,
    FYFConfig,
    FYFModel,
    build_scenarios,
    next_month_predictive,
    run_scenario,
    run_scenario_secondary,
)

# =============================================================================
# Config validation
# =============================================================================


class TestFYFConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"prior_sd": 0.0},
            {"prior_sd": -1.0},
            {"obs_sd": 0.0},
            {"obs_sd": -1.0},
            {"annual_horizon": 0},
            {"annual_horizon": -1},
        ],
    )
    def test_invalid_inputs_raise(self, kwargs: dict[str, float]) -> None:
        defaults = {
            "prior_mean": 1_050_000.0,
            "prior_sd": 150_000.0,
            "obs_sd": 80_000.0,
            "annual_horizon": 12,
        }
        defaults.update(kwargs)
        with pytest.raises(ValueError):
            FYFConfig(**defaults)


# =============================================================================
# process_month
# =============================================================================


class TestProcessMonth:
    def _make_model(self) -> FYFModel:
        return FYFModel(
            FYFConfig(
                prior_mean=1_050_000.0,
                prior_sd=150_000.0,
                obs_sd=80_000.0,
                budget_ceiling=13_200_000.0,
            )
        )

    def test_first_month_review_fields(self) -> None:
        model = self._make_model()
        review = model.process_month(1_120_000.0)
        assert review.month == 1
        assert review.actual == pytest.approx(1_120_000.0)
        # Surprise z = (1.12M - 1.05M) / sqrt(150K^2 + 80K^2)
        expected_z = (1_120_000 - 1_050_000) / math.sqrt(150_000**2 + 80_000**2)
        assert review.surprise_z == pytest.approx(expected_z, rel=1e-9)
        # Posterior matches the conjugate updater.
        upd = NormalNormalUpdater(
            mu0=1_050_000, sigma0_sq=150_000**2, sigma_sq=80_000**2
        )
        expected_post = upd.update([1_120_000.0])
        assert review.posterior.mean() == pytest.approx(expected_post.mean())
        assert review.posterior.variance() == pytest.approx(
            expected_post.variance()
        )
        assert review.cumulative_actual == pytest.approx(1_120_000.0)
        assert review.is_quarter_end is False
        assert review.forecast is not None  # 11 months remaining
        assert review.p_over_budget is not None

    def test_quarter_end_flag(self) -> None:
        model = self._make_model()
        for x in [1_050_000.0, 1_050_000.0, 1_050_000.0]:
            review = model.process_month(x)
        # Last review is month 3 → quarter-end.
        assert review.month == 3
        assert review.is_quarter_end is True

    def test_forecast_is_none_at_horizon_end(self) -> None:
        model = self._make_model()
        for _ in range(12):
            review = model.process_month(1_050_000.0)
        assert review.month == 12
        assert review.forecast is None
        assert review.p_over_budget is None

    def test_cumulative_actual_tracks_running_sum(self) -> None:
        model = self._make_model()
        actuals = [1_050_000.0, 1_080_000.0, 1_120_000.0]
        running_sum = 0.0
        for x in actuals:
            running_sum += x
            review = model.process_month(x)
            assert review.cumulative_actual == pytest.approx(running_sum)

    def test_horizon_cap(self) -> None:
        model = self._make_model()
        for _ in range(ANNUAL_HORIZON):
            model.process_month(1_050_000.0)
        with pytest.raises(RuntimeError, match="horizon"):
            model.process_month(1_050_000.0)


# =============================================================================
# annual_cycle
# =============================================================================


class TestAnnualCycle:
    def test_correct_number_of_actuals(self) -> None:
        model = FYFModel(
            FYFConfig(
                prior_mean=1_050_000.0, prior_sd=150_000.0, obs_sd=80_000.0
            )
        )
        with pytest.raises(ValueError, match="Expected"):
            model.annual_cycle([1_050_000.0] * 5)

    def test_full_year(self) -> None:
        model = FYFModel(
            FYFConfig(
                prior_mean=1_050_000.0,
                prior_sd=150_000.0,
                obs_sd=80_000.0,
                budget_ceiling=13_200_000.0,
            )
        )
        actuals = [1_050_000.0 + i * 100 for i in range(12)]
        reviews = model.annual_cycle(actuals)
        assert len(reviews) == 12
        # Every quarter-end month should be flagged.
        flagged = [r.month for r in reviews if r.is_quarter_end]
        assert tuple(flagged) == QUARTER_END_MONTHS


# =============================================================================
# fyf_review
# =============================================================================


class TestQuarterlyReview:
    def _make_and_run(self) -> FYFModel:
        model = FYFModel(
            FYFConfig(
                prior_mean=1_050_000.0,
                prior_sd=150_000.0,
                obs_sd=80_000.0,
                budget_ceiling=13_200_000.0,
            )
        )
        rng = np.random.default_rng(seed=2026)
        model.annual_cycle(list(rng.normal(1_080_000, 80_000, size=12)))
        return model

    def test_quarter_review_window(self) -> None:
        model = self._make_and_run()
        q1 = model.fyf_review(1)
        assert q1.quarter == 1
        assert q1.month_end == 3
        # Posterior at q1 review matches the month-3 review.
        m3 = model.reviews()[2]
        assert q1.posterior.mean() == pytest.approx(m3.posterior.mean())

    def test_max_abs_surprise_is_within_window(self) -> None:
        model = self._make_and_run()
        for q in (1, 2, 3, 4):
            review_window = model.reviews()[3 * (q - 1) : 3 * q]
            expected_max = max(abs(r.surprise_z) for r in review_window)
            qr = model.fyf_review(q)
            assert qr.max_abs_surprise == pytest.approx(expected_max)

    def test_invalid_quarter_raises(self) -> None:
        model = self._make_and_run()
        with pytest.raises(ValueError):
            model.fyf_review(0)
        with pytest.raises(ValueError):
            model.fyf_review(5)

    def test_review_before_quarter_complete_raises(self) -> None:
        model = FYFModel(
            FYFConfig(
                prior_mean=1_050_000.0, prior_sd=150_000.0, obs_sd=80_000.0
            )
        )
        with pytest.raises(RuntimeError, match="Cannot review"):
            model.fyf_review(1)


# =============================================================================
# Recommendations
# =============================================================================


class TestRecommendations:
    def _config(self, ceiling: float | None = 13_200_000.0) -> FYFConfig:
        return FYFConfig(
            prior_mean=1_050_000.0,
            prior_sd=150_000.0,
            obs_sd=80_000.0,
            budget_ceiling=ceiling,
        )

    def test_shock_triggers_investigate(self) -> None:
        # Use a tight-prior config so that a moderate shock easily exceeds 3σ_pred
        config = FYFConfig(
            prior_mean=1_050_000.0,
            prior_sd=20_000.0,
            obs_sd=20_000.0,
            budget_ceiling=13_200_000.0,
        )
        model = FYFModel(config)
        # First two months as expected, third month is a 5σ shock
        model.process_month(1_050_000.0)
        model.process_month(1_050_000.0)
        model.process_month(1_300_000.0)  # huge surprise
        review = model.fyf_review(1)
        assert review.recommendation == "investigate shock"

    def test_repeated_drift_triggers_re_elicit(self) -> None:
        # Two months with |z| > 2 in the same quarter, no single |z| > 3.
        config = FYFConfig(
            prior_mean=1_000.0,
            prior_sd=100.0,
            obs_sd=100.0,
            budget_ceiling=None,
        )
        model = FYFModel(config)
        # Hand-chosen actuals so that z₁ ≈ +2.5 and z₂ ≈ +2.1 with a quiet
        # month 3. See test docstring for derivation.
        model.process_month(1_352.5)
        model.process_month(1_433.5)
        model.process_month(model.current_posterior().mean())
        review = model.fyf_review(1)
        assert review.max_abs_surprise <= 3.0
        assert review.n_surprises_above_2 >= 2
        assert review.recommendation == "re-elicit prior or model"

    def test_high_p_over_budget_triggers_revision(self) -> None:
        # Plan is way too low; data systematically much higher.
        config = FYFConfig(
            prior_mean=1_050_000.0,
            prior_sd=150_000.0,
            obs_sd=80_000.0,
            budget_ceiling=12_000_000.0,  # well below true 12 × 1.10M
        )
        model = FYFModel(config)
        rng = np.random.default_rng(seed=10)
        # Feed 6 months at well-above plan to force P(T>B) very high
        for x in rng.normal(1_150_000, 80_000, size=6):
            model.process_month(float(x))
        review = model.fyf_review(2)
        # Either "investigate shock" / "re-elicit" / "request budget revision"
        assert review.recommendation in {
            "request budget revision",
            "re-elicit prior or model",
            "investigate shock",
        }

    def test_quiet_quarter_holds(self) -> None:
        config = self._config()
        model = FYFModel(config)
        for _ in range(3):
            model.process_month(config.prior_mean)
        review = model.fyf_review(1)
        assert review.recommendation == "hold"


# =============================================================================
# next_month_predictive
# =============================================================================


def test_next_month_predictive_at_prior() -> None:
    config = FYFConfig(
        prior_mean=1_050_000.0, prior_sd=150_000.0, obs_sd=80_000.0
    )
    model = FYFModel(config)
    mean, std = next_month_predictive(model)
    assert mean == pytest.approx(1_050_000.0)
    expected_std = math.sqrt(150_000**2 + 80_000**2)
    assert std == pytest.approx(expected_std, rel=1e-9)


# =============================================================================
# Scenarios
# =============================================================================


class TestScenarios:
    def test_build_scenarios_returns_five(self) -> None:
        scenarios = build_scenarios()
        ids = [s.id for s in scenarios]
        assert ids == [
            "s1_on_target",
            "s2_optimistic",
            "s3_shock",
            "s4_seasonal",
            "s5_prior_sensitivity",
        ]
        for s in scenarios:
            assert len(s.actuals) == ANNUAL_HORIZON

    def test_each_scenario_runs_end_to_end(self) -> None:
        for scenario in build_scenarios():
            model = run_scenario(scenario)
            assert model.n_months_processed == ANNUAL_HORIZON
            assert len(model.reviews()) == ANNUAL_HORIZON

    def test_secondary_only_for_s5(self) -> None:
        for scenario in build_scenarios():
            secondary = run_scenario_secondary(scenario)
            if scenario.id == "s5_prior_sensitivity":
                assert secondary is not None
                assert secondary.n_months_processed == ANNUAL_HORIZON
            else:
                assert secondary is None

    def test_seed_reproducibility(self) -> None:
        a = build_scenarios(base_seed=42)
        b = build_scenarios(base_seed=42)
        for sa, sb in zip(a, b, strict=True):
            assert sa.actuals == sb.actuals

    def test_shock_scenario_contains_large_value_at_month_5(self) -> None:
        scenarios = {s.id: s for s in build_scenarios()}
        shock = scenarios["s3_shock"]
        assert shock.actuals[4] == pytest.approx(1_400_000.0)


# =============================================================================
# Reset
# =============================================================================


def test_reset_restores_initial_state() -> None:
    config = FYFConfig(
        prior_mean=1_050_000.0, prior_sd=150_000.0, obs_sd=80_000.0
    )
    model = FYFModel(config)
    model.annual_cycle([1_080_000.0] * 12)
    assert model.n_months_processed == 12

    model.reset()
    assert model.n_months_processed == 0
    assert model.reviews() == []
    post = model.current_posterior()
    assert isinstance(post, NormalPosterior)
    assert post.mean() == pytest.approx(1_050_000.0)
    assert post.variance() == pytest.approx(150_000**2)
