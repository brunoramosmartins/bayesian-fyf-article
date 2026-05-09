"""Tests for src/priors.py — DO NOT auto-run.

Author runs: ``pytest tests/`` and ``ruff check .`` after each
implementation phase. These tests cover correctness of the closed-form
elicitation helpers and the summary utility.
"""

from __future__ import annotations

import math

import pytest

from src.priors import (
    Prior,
    beta_prior_from_proportion,
    gamma_prior_from_rate,
    normal_prior_from_budget,
    prior_summary,
)


# ---------------------------------------------------------------------------
# normal_prior_from_budget
# ---------------------------------------------------------------------------


class TestNormalPriorFromBudget:
    def test_budget_plan_default_case_matches_closed_form(self) -> None:
        # μ0 = 1.05M, 90 % confidence in ±15 % ⟹ σ0 ≈ 95,753 (z=1.6449)
        p = normal_prior_from_budget(1_050_000, 0.90, 0.15)
        assert p.family == "normal"
        assert p.params["mu"] == pytest.approx(1_050_000)
        # 0.15 * 1_050_000 / 1.6448536269514722 ≈ 95753.2
        assert p.params["sigma"] == pytest.approx(95_753.2, rel=1e-4)

    def test_higher_confidence_widens_sigma(self) -> None:
        # For the same band, demanding 99 % confidence forces a smaller σ
        # (the band has to cover MORE of the distribution).
        p_90 = normal_prior_from_budget(1000, 0.90, 0.10)
        p_99 = normal_prior_from_budget(1000, 0.99, 0.10)
        assert p_99.params["sigma"] < p_90.params["sigma"]

    def test_wider_band_loosens_sigma(self) -> None:
        # For the same confidence, a wider band corresponds to LARGER σ.
        p_narrow = normal_prior_from_budget(1000, 0.90, 0.05)
        p_wide = normal_prior_from_budget(1000, 0.90, 0.25)
        assert p_wide.params["sigma"] > p_narrow.params["sigma"]

    def test_band_recovers_stated_confidence(self) -> None:
        # By construction P(θ ∈ [μ ± w·μ]) = γ. Verify by integrating.
        p = normal_prior_from_budget(1_000_000, 0.90, 0.15)
        rv = p.frozen()
        prob = rv.cdf(1_000_000 * 1.15) - rv.cdf(1_000_000 * 0.85)
        assert prob == pytest.approx(0.90, abs=1e-6)

    def test_negative_plan_value_uses_absolute_band(self) -> None:
        # Sign of plan_value should not flip σ.
        p = normal_prior_from_budget(-1000, 0.90, 0.10)
        assert p.params["sigma"] > 0

    @pytest.mark.parametrize("conf", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_confidence_raises(self, conf: float) -> None:
        with pytest.raises(ValueError, match="confidence_pct"):
            normal_prior_from_budget(1000, conf, 0.10)

    @pytest.mark.parametrize("width", [0.0, -0.1])
    def test_invalid_width_raises(self, width: float) -> None:
        with pytest.raises(ValueError, match="interval_width"):
            normal_prior_from_budget(1000, 0.90, width)

    def test_zero_plan_raises(self) -> None:
        with pytest.raises(ValueError, match="plan_value"):
            normal_prior_from_budget(0.0, 0.90, 0.10)


# ---------------------------------------------------------------------------
# gamma_prior_from_rate
# ---------------------------------------------------------------------------


class TestGammaPriorFromRate:
    def test_default_case_matches_pseudo_observation_form(self) -> None:
        p = gamma_prior_from_rate(expected_rate=3.0, confidence=1.0)
        assert p.family == "gamma"
        assert p.params == {"alpha": 3.0, "beta": 1.0}

    def test_mean_equals_expected_rate(self) -> None:
        # E[λ] = α / β. Verified across several parameterisations.
        for rate, n in [(3.0, 1.0), (0.5, 10.0), (12.0, 4.0)]:
            p = gamma_prior_from_rate(rate, n)
            mean = p.params["alpha"] / p.params["beta"]
            assert mean == pytest.approx(rate)

    def test_higher_confidence_tightens_variance(self) -> None:
        # Var = α / β² = rate / β. Larger confidence ⟹ smaller variance.
        p_weak = gamma_prior_from_rate(3.0, 1.0)
        p_strong = gamma_prior_from_rate(3.0, 10.0)
        var_weak = p_weak.params["alpha"] / p_weak.params["beta"] ** 2
        var_strong = p_strong.params["alpha"] / p_strong.params["beta"] ** 2
        assert var_strong < var_weak

    @pytest.mark.parametrize("rate", [0.0, -1.0])
    def test_invalid_rate_raises(self, rate: float) -> None:
        with pytest.raises(ValueError, match="expected_rate"):
            gamma_prior_from_rate(rate, 1.0)

    @pytest.mark.parametrize("conf", [0.0, -1.0])
    def test_invalid_confidence_raises(self, conf: float) -> None:
        with pytest.raises(ValueError, match="confidence"):
            gamma_prior_from_rate(3.0, conf)


# ---------------------------------------------------------------------------
# beta_prior_from_proportion
# ---------------------------------------------------------------------------


class TestBetaPriorFromProportion:
    def test_default_case_matches_pseudo_count_form(self) -> None:
        p = beta_prior_from_proportion(0.2, 10)
        assert p.family == "beta"
        assert p.params == {"alpha": 2.0, "beta": 8.0}

    def test_mean_equals_expected_proportion(self) -> None:
        for prop, n in [(0.5, 4), (0.20, 10), (0.7, 20)]:
            p = beta_prior_from_proportion(prop, n)
            a, b = p.params["alpha"], p.params["beta"]
            assert a / (a + b) == pytest.approx(prop)
            assert a + b == pytest.approx(n)

    def test_larger_pseudo_n_tightens_prior(self) -> None:
        p_weak = beta_prior_from_proportion(0.5, 2)
        p_strong = beta_prior_from_proportion(0.5, 50)
        # Beta variance = ab / [(a+b)²(a+b+1)] — strictly decreasing in n.
        def _var(p: Prior) -> float:
            a, b = p.params["alpha"], p.params["beta"]
            return (a * b) / (((a + b) ** 2) * (a + b + 1))

        assert _var(p_strong) < _var(p_weak)

    @pytest.mark.parametrize("prop", [0.0, 1.0, -0.1, 1.1])
    def test_invalid_proportion_raises(self, prop: float) -> None:
        with pytest.raises(ValueError, match="expected_prop"):
            beta_prior_from_proportion(prop, 10)

    @pytest.mark.parametrize("n", [0.0, -1.0])
    def test_invalid_pseudo_n_raises(self, n: float) -> None:
        with pytest.raises(ValueError, match="sample_size_equiv"):
            beta_prior_from_proportion(0.5, n)


# ---------------------------------------------------------------------------
# prior_summary
# ---------------------------------------------------------------------------


class TestPriorSummary:
    def test_normal_summary_recovers_hyperparameters(self) -> None:
        p = normal_prior_from_budget(1_050_000, 0.90, 0.15)
        s = prior_summary(p, credible_level=0.95)
        assert s["mean"] == pytest.approx(1_050_000)
        assert s["std"] == pytest.approx(p.params["sigma"], rel=1e-9)
        # 95 % CI for Normal: μ ± 1.96σ
        lo, hi = s["credible_interval"]  # type: ignore[misc]
        assert lo == pytest.approx(1_050_000 - 1.96 * p.params["sigma"], rel=1e-3)
        assert hi == pytest.approx(1_050_000 + 1.96 * p.params["sigma"], rel=1e-3)

    def test_gamma_summary_mean_and_variance(self) -> None:
        p = gamma_prior_from_rate(3.0, 1.0)
        s = prior_summary(p, credible_level=0.90)
        assert s["mean"] == pytest.approx(3.0)
        # Var = α / β² = 3
        assert s["variance"] == pytest.approx(3.0)
        assert s["std"] == pytest.approx(math.sqrt(3.0))

    def test_beta_summary_mean(self) -> None:
        p = beta_prior_from_proportion(0.2, 10)
        s = prior_summary(p)
        assert s["mean"] == pytest.approx(0.2)
        lo, hi = s["credible_interval"]  # type: ignore[misc]
        assert 0.0 < lo < 0.2 < hi < 1.0

    def test_credible_level_must_be_in_unit_interval(self) -> None:
        p = beta_prior_from_proportion(0.2, 10)
        with pytest.raises(ValueError, match="credible_level"):
            prior_summary(p, credible_level=1.5)


# ---------------------------------------------------------------------------
# Exercise 6 (paper) cross-check
# ---------------------------------------------------------------------------


def test_exercise_6_matches_paper_solution() -> None:
    """``exercises/ex01_bayes_foundations.md`` exercise 6: σ₀ ≈ 95,753."""
    p = normal_prior_from_budget(1_050_000, 0.90, 0.15)
    assert p.params["sigma"] == pytest.approx(95_753, abs=1.0)
