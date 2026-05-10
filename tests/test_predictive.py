"""Tests for src/predictive.py — DO NOT auto-run.

Coverage:

- Normal-Normal predictive: mean, variance = sigma_n^2 + sigma^2,
  matches scipy.norm pdf/cdf, exercise-6 numerical answers.
- Gamma-Poisson predictive: matches NegBin mean/variance and the
  variance decomposition from §3.2 of the theory notes.
- Beta-Binomial predictive: matches betabinom mean.
- ``year_end_predictive_total``: variance is quadratic in horizon for
  the parameter term and linear for the noise term; matches the
  exercise-7 expected probability of exceeding budget.
- ``posterior_predictive_sample`` correctness:
  * Sample mean/variance agree with closed form for ``correlated=True``.
  * Multi-period samples within a replication are correlated when
    ``correlated=True`` and uncorrelated when ``correlated=False``.
- ``log_marginal_likelihood_normal_normal``: matches the multivariate
  Normal joint density on a small example.
- Bayes factor computation for two competing Normal priors.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from src.conjugate import (
    BetaPosterior,
    GammaPosterior,
    NormalPosterior,
)
from src.predictive import (
    YearEndForecast,
    bayes_factor,
    log_marginal_likelihood_normal_normal,
    posterior_predictive_beta_binomial,
    posterior_predictive_gamma_poisson,
    posterior_predictive_normal,
    posterior_predictive_sample,
    prob_over_budget,
    year_end_predictive_total,
)

# =============================================================================
# Single-period predictives
# =============================================================================


class TestNormalPredictive:
    def test_mean_matches_posterior_mean(self) -> None:
        post = NormalPosterior(mu=10.0, sigma_sq=4.0)
        pred = posterior_predictive_normal(post, sigma_sq=1.0)
        assert pred.mean() == pytest.approx(10.0)

    def test_variance_is_sum_of_components(self) -> None:
        post = NormalPosterior(mu=10.0, sigma_sq=4.0)
        pred = posterior_predictive_normal(post, sigma_sq=1.0)
        # Total variance = sigma_n^2 + sigma^2 = 4 + 1 = 5
        assert pred.var() == pytest.approx(5.0)

    def test_predictive_wider_than_credible(self) -> None:
        # The predictive interval is always wider than the credible
        # interval — the sampling-noise share never vanishes.
        post = NormalPosterior(mu=0.0, sigma_sq=1.0)
        sigma_sq = 4.0
        pred = posterior_predictive_normal(post, sigma_sq=sigma_sq)
        assert pred.var() > post.variance()

    def test_exercise_6_predictive_interval(self) -> None:
        # exercises/ex04_predictive_inference.md exercise 6:
        # posterior N(1,085,000, 32,000^2), sigma=80,000.
        post = NormalPosterior(mu=1_085_000.0, sigma_sq=32_000.0**2)
        pred = posterior_predictive_normal(post, sigma_sq=80_000.0**2)
        assert pred.mean() == pytest.approx(1_085_000.0)
        assert pred.std() == pytest.approx(86_162.0, abs=2.0)
        lo, hi = pred.ppf(0.025), pred.ppf(0.975)
        assert lo == pytest.approx(916_123.0, abs=200.0)
        assert hi == pytest.approx(1_253_877.0, abs=200.0)

    def test_invalid_sigma_sq_raises(self) -> None:
        post = NormalPosterior(mu=0.0, sigma_sq=1.0)
        with pytest.raises(ValueError, match="sigma_sq"):
            posterior_predictive_normal(post, sigma_sq=0.0)


class TestGammaPoissonPredictive:
    def test_mean_matches_posterior_mean(self) -> None:
        post = GammaPosterior(alpha=20.0, beta=7.0)
        pred = posterior_predictive_gamma_poisson(post)
        assert pred.mean() == pytest.approx(20.0 / 7.0, rel=1e-12)

    def test_variance_decomposition(self) -> None:
        # Var(x̃) = E[λ] + Var(λ) for the Gamma-Poisson predictive.
        post = GammaPosterior(alpha=20.0, beta=7.0)
        pred = posterior_predictive_gamma_poisson(post)
        e_lambda = post.mean()
        var_lambda = post.variance()
        assert pred.var() == pytest.approx(e_lambda + var_lambda, rel=1e-12)

    def test_overdispersion_holds(self) -> None:
        # Gamma-Poisson predictive has Var > mean (Poisson would have ==).
        post = GammaPosterior(alpha=5.0, beta=2.0)
        pred = posterior_predictive_gamma_poisson(post)
        assert pred.var() > pred.mean()


class TestBetaBinomialPredictive:
    def test_mean_matches_n_trials_times_proportion(self) -> None:
        post = BetaPosterior(alpha=14.0, beta=46.0)
        n_trials = 100
        pred = posterior_predictive_beta_binomial(post, n_trials=n_trials)
        expected_mean = n_trials * post.mean()
        assert pred.mean() == pytest.approx(expected_mean, rel=1e-12)

    def test_zero_trials_returns_zero(self) -> None:
        post = BetaPosterior(alpha=2.0, beta=8.0)
        pred = posterior_predictive_beta_binomial(post, n_trials=0)
        # 0-trial Binomial is degenerate at 0; betabinom inherits that.
        assert pred.pmf(0) == pytest.approx(1.0, rel=1e-9)

    def test_negative_trials_raises(self) -> None:
        post = BetaPosterior(alpha=2.0, beta=8.0)
        with pytest.raises(ValueError, match="n_trials"):
            posterior_predictive_beta_binomial(post, n_trials=-1)


# =============================================================================
# Year-end forecast and over-budget probability
# =============================================================================


class TestYearEndForecast:
    def test_variance_is_quadratic_in_horizon_for_param_term(self) -> None:
        # Var = h^2 sigma_n^2 + h sigma^2.
        post = NormalPosterior(mu=1_085_000.0, sigma_sq=32_000.0**2)
        sigma_sq = 80_000.0**2
        for h in (1, 3, 6, 12):
            f = year_end_predictive_total(post, sigma_sq, n_remaining=h)
            expected_var = (h * h) * post.variance() + h * sigma_sq
            assert f.variance == pytest.approx(expected_var, rel=1e-12)

    def test_naive_iid_formula_understates_variance(self) -> None:
        # Compare correct quadratic-h variance to the naïve linear-h one.
        post = NormalPosterior(mu=1_085_000.0, sigma_sq=32_000.0**2)
        sigma_sq = 80_000.0**2
        h = 6
        correct = year_end_predictive_total(post, sigma_sq, n_remaining=h)
        naive_var = h * (post.variance() + sigma_sq)
        assert correct.variance > naive_var

    def test_exercise_7_probability(self) -> None:
        # exercises/ex04: mid-year forecast, B=13.2M.
        post = NormalPosterior(mu=1_085_000.0, sigma_sq=32_000.0**2)
        sigma_sq = 80_000.0**2
        observed_total = 6_510_000.0
        budget = 13_200_000.0

        f = year_end_predictive_total(
            posterior=post,
            sigma_sq=sigma_sq,
            n_remaining=6,
            observed_total=observed_total,
        )
        assert f.mean == pytest.approx(13_020_000.0, rel=1e-9)
        assert f.std() == pytest.approx(274_343.0, abs=2.0)

        p = f.prob_above(budget)
        assert p == pytest.approx(0.256, abs=2e-3)

    def test_prob_over_budget_matches_forecast(self) -> None:
        post = NormalPosterior(mu=1_085_000.0, sigma_sq=32_000.0**2)
        sigma_sq = 80_000.0**2
        p1 = prob_over_budget(post, sigma_sq, 6, 13_200_000.0, 6_510_000.0)
        f = year_end_predictive_total(post, sigma_sq, 6, 6_510_000.0)
        p2 = f.prob_above(13_200_000.0)
        assert p1 == pytest.approx(p2, rel=1e-12)

    def test_credible_interval_symmetric(self) -> None:
        f = YearEndForecast(
            mean=10_000_000.0,
            variance=1e10,
            n_remaining=3,
            observed_total=0.0,
        )
        lo, hi = f.credible_interval(0.95)
        assert (lo + hi) / 2.0 == pytest.approx(10_000_000.0, rel=1e-9)
        # Width = 2 · 1.96 · σ
        assert hi - lo == pytest.approx(2.0 * 1.96 * f.std(), rel=2e-3)

    def test_invalid_inputs_raise(self) -> None:
        post = NormalPosterior(mu=0.0, sigma_sq=1.0)
        with pytest.raises(ValueError, match="sigma_sq"):
            year_end_predictive_total(post, sigma_sq=0.0, n_remaining=6)
        with pytest.raises(ValueError, match="n_remaining"):
            year_end_predictive_total(post, sigma_sq=1.0, n_remaining=-1)


# =============================================================================
# Monte Carlo posterior predictive sampling
# =============================================================================


class TestPosteriorPredictiveSample:
    def test_correlated_sample_mean_variance_match_closed_form(self) -> None:
        # Single period — correlated and uncorrelated should agree
        # in distribution; only multi-period changes.
        post = NormalPosterior(mu=10.0, sigma_sq=4.0)
        sigma_sq = 1.0
        samples = posterior_predictive_sample(
            post, sigma_sq=sigma_sq, n_samples=200_000, seed=42
        )
        assert samples.mean() == pytest.approx(post.mean(), abs=0.05)
        # Closed-form predictive variance: 4 + 1 = 5.
        assert samples.var() == pytest.approx(5.0, rel=5e-2)

    def test_correlated_multi_period_preserves_total_variance(self) -> None:
        # h=6 future periods. Var(sum) = h^2 σ_n^2 + h σ^2.
        post = NormalPosterior(mu=10.0, sigma_sq=4.0)
        sigma_sq = 1.0
        h = 6
        samples = posterior_predictive_sample(
            post, sigma_sq=sigma_sq, n_samples=50_000, n_periods=h, seed=7
        )
        sums = samples.sum(axis=1)
        expected_var = (h * h) * post.variance() + h * sigma_sq
        assert sums.var() == pytest.approx(expected_var, rel=5e-2)

    def test_uncorrelated_multi_period_underestimates_total_variance(self) -> None:
        # The naïve "iid future months" sample variance for the SUM
        # equals h · marginal predictive variance — not the correct one.
        post = NormalPosterior(mu=10.0, sigma_sq=4.0)
        sigma_sq = 1.0
        h = 6
        samples = posterior_predictive_sample(
            post, sigma_sq=sigma_sq, n_samples=50_000, n_periods=h,
            seed=11, correlated=False,
        )
        sums = samples.sum(axis=1)
        naive_var = h * (post.variance() + sigma_sq)
        correct_var = (h * h) * post.variance() + h * sigma_sq
        assert sums.var() == pytest.approx(naive_var, rel=5e-2)
        assert sums.var() < correct_var * 0.99  # strictly less

    def test_seed_reproducibility(self) -> None:
        post = NormalPosterior(mu=0.0, sigma_sq=1.0)
        a = posterior_predictive_sample(post, sigma_sq=1.0, n_samples=100, seed=2026)
        b = posterior_predictive_sample(post, sigma_sq=1.0, n_samples=100, seed=2026)
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize(
        ("kw", "value"),
        [("n_samples", 0), ("n_samples", -1), ("n_periods", 0), ("sigma_sq", 0.0)],
    )
    def test_invalid_inputs_raise(self, kw: str, value: object) -> None:
        post = NormalPosterior(mu=0.0, sigma_sq=1.0)
        kwargs: dict[str, object] = {"n_samples": 10, "n_periods": 1, "sigma_sq": 1.0}
        kwargs[kw] = value
        with pytest.raises(ValueError):
            posterior_predictive_sample(post, **kwargs)  # type: ignore[arg-type]


# =============================================================================
# Bayes factor / log marginal likelihood
# =============================================================================


class TestLogMarginalAndBayesFactor:
    def test_two_observation_known_answer(self) -> None:
        # Computed by hand in notes (matches multivariate Normal joint).
        # x=[1,2], μ_0=0, σ_0²=1, σ²=1 ⟹ log p(x) = -log(2π) - 0.5 log 3 - 1.
        log_p = log_marginal_likelihood_normal_normal(
            data=[1.0, 2.0], mu0=0.0, sigma0_sq=1.0, sigma_sq=1.0
        )
        expected = -math.log(2.0 * math.pi) - 0.5 * math.log(3.0) - 1.0
        assert log_p == pytest.approx(expected, rel=1e-12)

    def test_matches_multivariate_normal_density(self) -> None:
        # Verify against scipy's multivariate_normal density on the
        # joint x ~ N(μ_0·1, σ²·I + σ_0²·J).
        rng = np.random.default_rng(seed=2026)
        x = rng.normal(loc=2.0, scale=0.7, size=5)
        mu0, sigma0_sq, sigma_sq = 0.5, 1.5, 0.8

        n = len(x)
        Sigma = sigma_sq * np.eye(n) + sigma0_sq * np.ones((n, n))
        mvn = stats.multivariate_normal(mean=np.full(n, mu0), cov=Sigma)
        expected = float(mvn.logpdf(x))

        actual = log_marginal_likelihood_normal_normal(
            data=x, mu0=mu0, sigma0_sq=sigma0_sq, sigma_sq=sigma_sq
        )
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_empty_data_returns_zero(self) -> None:
        assert log_marginal_likelihood_normal_normal(
            data=[], mu0=0.0, sigma0_sq=1.0, sigma_sq=1.0
        ) == pytest.approx(0.0)

    def test_bayes_factor_matches_definition(self) -> None:
        log_a = -3.0
        log_b = -5.0
        bf = bayes_factor(log_a, log_b)
        assert bf == pytest.approx(math.exp(2.0), rel=1e-12)

    def test_bayes_factor_for_two_normal_priors(self) -> None:
        # Two Normal priors centred at the truth vs centred 5σ_0 away.
        rng = np.random.default_rng(seed=11)
        true_mean = 1_080_000.0
        x = rng.normal(true_mean, 80_000.0, size=12)

        log_close = log_marginal_likelihood_normal_normal(
            data=x, mu0=true_mean, sigma0_sq=150_000**2, sigma_sq=80_000**2
        )
        log_far = log_marginal_likelihood_normal_normal(
            data=x, mu0=true_mean + 5 * 150_000, sigma0_sq=150_000**2,
            sigma_sq=80_000**2,
        )
        # Prior closer to the truth must have higher marginal likelihood.
        assert log_close > log_far
        bf = bayes_factor(log_close, log_far)
        assert bf > 1.0
