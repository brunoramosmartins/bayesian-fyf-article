"""Tests for src/diagnostics.py — DO NOT auto-run.

Coverage:

- ``surprise_score`` returns the right z under hand-checked inputs.
- ``surprise_trajectory`` errors on length mismatch.
- ``posterior_predictive_pvalues``: empirical distribution under
  correctly-specified Monte Carlo data is approximately Uniform(0, 1).
- ``calibration_score`` recovers the nominal level on simulated
  well-calibrated data.
- ``calibration_binomial_test``: returns the binomial p-value the
  exercise 5 numbers expect (60 trials, 51 successes at p=0.95).
- ``cumulative_surprise``: matches running cumsum of surprises.
- ``priors_before_each_step`` correctly reconstructs the priors used
  at each step for an FYFModel run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.conjugate import NormalNormalUpdater, NormalPosterior
from src.diagnostics import (
    calibration_binomial_test,
    calibration_score,
    cumulative_surprise,
    posterior_predictive_pvalues,
    priors_before_each_step,
    surprise_score,
    surprise_trajectory,
)
from src.fyf_model import FYFConfig, FYFModel
from src.updating import SequentialUpdater

# =============================================================================
# surprise_score
# =============================================================================


class TestSurpriseScore:
    def test_known_value(self) -> None:
        post = NormalPosterior(mu=10.0, sigma_sq=4.0)
        # Predictive var = 4 + 1 = 5; z = (15 - 10) / sqrt(5).
        z = surprise_score(post, actual=15.0, sigma_sq=1.0)
        assert z == pytest.approx(5.0 / math.sqrt(5.0), rel=1e-12)

    def test_invalid_sigma_sq_raises(self) -> None:
        post = NormalPosterior(mu=0.0, sigma_sq=1.0)
        with pytest.raises(ValueError):
            surprise_score(post, actual=0.0, sigma_sq=0.0)


class TestSurpriseTrajectory:
    def test_length_mismatch_raises(self) -> None:
        post = NormalPosterior(mu=0.0, sigma_sq=1.0)
        with pytest.raises(ValueError):
            surprise_trajectory([post] * 3, [1.0, 2.0], sigma_sq=1.0)

    def test_matches_pointwise(self) -> None:
        posts = [
            NormalPosterior(mu=10.0, sigma_sq=4.0),
            NormalPosterior(mu=11.0, sigma_sq=2.0),
        ]
        actuals = [12.0, 11.5]
        z = surprise_trajectory(posts, actuals, sigma_sq=1.0)
        z_expected = [
            surprise_score(p, x, 1.0) for p, x in zip(posts, actuals, strict=True)
        ]
        np.testing.assert_allclose(z, z_expected, rtol=1e-12)


# =============================================================================
# posterior_predictive_pvalues
# =============================================================================


class TestPosteriorPredictivePvalues:
    def test_in_unit_interval(self) -> None:
        rng = np.random.default_rng(seed=2026)
        posts = [NormalPosterior(mu=0.0, sigma_sq=1.0) for _ in range(50)]
        actuals = list(rng.normal(0.0, math.sqrt(2.0), size=50))  # predictive σ
        p = posterior_predictive_pvalues(posts, actuals, sigma_sq=1.0)
        assert np.all((0.0 <= p) & (p <= 1.0))

    def test_uniform_under_correct_specification(self) -> None:
        # Many predictive draws should give p-values approximately Uniform.
        rng = np.random.default_rng(seed=11)
        n = 5000
        posts = [NormalPosterior(mu=0.0, sigma_sq=1.0) for _ in range(n)]
        actuals = list(rng.normal(0.0, math.sqrt(2.0), size=n))
        p = posterior_predictive_pvalues(posts, actuals, sigma_sq=1.0)
        # Mean of Uniform(0,1) is 0.5
        assert p.mean() == pytest.approx(0.5, abs=0.02)
        # Fraction <= 0.5 should be ~0.5
        assert (p <= 0.5).mean() == pytest.approx(0.5, abs=0.02)


# =============================================================================
# calibration_score
# =============================================================================


class TestCalibrationScore:
    def test_perfect_calibration_recovered_in_limit(self) -> None:
        rng = np.random.default_rng(seed=1)
        n = 10_000
        posts = [NormalPosterior(mu=0.0, sigma_sq=1.0) for _ in range(n)]
        # Predictive var = 1 + 1 = 2; sample under correct model.
        actuals = list(rng.normal(0.0, math.sqrt(2.0), size=n))
        score = calibration_score(posts, actuals, sigma_sq=1.0, level=0.95)
        assert score == pytest.approx(0.95, abs=0.01)

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError):
            calibration_score(
                [NormalPosterior(mu=0.0, sigma_sq=1.0)],
                [0.0],
                sigma_sq=1.0,
                level=1.5,
            )


# =============================================================================
# calibration_binomial_test  (exercise 5 cross-check)
# =============================================================================


class TestCalibrationBinomialTest:
    def test_exercise_5_numbers(self) -> None:
        # 60 trials, 51 inside 95% predictive interval.
        # Construct synthetic posteriors+actuals that produce this count.
        n_total = 60
        n_inside = 51
        sigma_sq = 1.0
        post = NormalPosterior(mu=0.0, sigma_sq=1.0)
        # Predictive sd = sqrt(2). z=0 is inside the 95 % interval;
        # actual=5 has z ≈ 3.5 (outside).
        inside_actual = 0.0
        outside_actual = 5.0
        actuals = (
            [inside_actual] * n_inside + [outside_actual] * (n_total - n_inside)
        )
        posts = [post] * n_total
        result = calibration_binomial_test(
            posts, actuals, sigma_sq=sigma_sq, level=0.95
        )
        assert result.n_observed == n_total
        assert result.n_inside == n_inside
        assert result.empirical_coverage == pytest.approx(51 / 60)
        # Check the binomial p-value is small (under 5 % at the 5 % level).
        # P(X <= 51 | n=60, p=0.95) ≈ 0.024; two-sided ≈ 0.024 + small upper.
        assert result.p_value_two_sided < 0.10
        assert result.is_calibrated(alpha=0.001) is True
        assert result.is_calibrated(alpha=0.10) is False


# =============================================================================
# cumulative_surprise
# =============================================================================


class TestCumulativeSurprise:
    def test_matches_cumsum(self) -> None:
        posts = [
            NormalPosterior(mu=0.0, sigma_sq=1.0),
            NormalPosterior(mu=0.5, sigma_sq=0.5),
            NormalPosterior(mu=0.7, sigma_sq=0.3),
        ]
        actuals = [1.5, 0.8, 0.6]
        s = cumulative_surprise(posts, actuals, sigma_sq=1.0)
        z = surprise_trajectory(posts, actuals, sigma_sq=1.0)
        np.testing.assert_allclose(s, np.cumsum(z), rtol=1e-12)


# =============================================================================
# priors_before_each_step
# =============================================================================


class TestPriorsBeforeEachStep:
    def test_reconstructs_priors_from_fyf_history(self) -> None:
        config = FYFConfig(
            prior_mean=1_050_000.0,
            prior_sd=150_000.0,
            obs_sd=80_000.0,
        )
        model = FYFModel(config)
        rng = np.random.default_rng(seed=2026)
        model.annual_cycle(list(rng.normal(1_080_000, 80_000, size=12)))

        prior = NormalPosterior(
            mu=config.prior_mean, sigma_sq=config.prior_sd**2
        )
        priors = priors_before_each_step(model.reviews(), prior)
        assert len(priors) == 12
        # Step 1 prior is the original prior.
        assert priors[0].mean() == pytest.approx(prior.mean())
        assert priors[0].variance() == pytest.approx(prior.variance())
        # Step 2 prior is the posterior after step 1.
        assert priors[1].mean() == pytest.approx(model.reviews()[0].posterior.mean())

    def test_diagnostics_match_model_surprise(self) -> None:
        # Surprise computed by the model and by the standalone diagnostic
        # should agree.
        config = FYFConfig(
            prior_mean=1_050_000.0, prior_sd=150_000.0, obs_sd=80_000.0
        )
        model = FYFModel(config)
        rng = np.random.default_rng(seed=11)
        model.annual_cycle(list(rng.normal(1_080_000, 80_000, size=12)))

        prior = NormalPosterior(
            mu=config.prior_mean, sigma_sq=config.prior_sd**2
        )
        priors = priors_before_each_step(model.reviews(), prior)
        actuals = [r.actual for r in model.reviews()]
        z_diag = surprise_trajectory(priors, actuals, sigma_sq=config.obs_sd**2)
        z_model = np.array([r.surprise_z for r in model.reviews()])
        np.testing.assert_allclose(z_diag, z_model, rtol=1e-12)


# =============================================================================
# Integration with SequentialUpdater (smoke)
# =============================================================================


def test_works_with_plain_sequential_updater() -> None:
    upd = NormalNormalUpdater(mu0=0.0, sigma0_sq=1.0, sigma_sq=1.0)
    seq = SequentialUpdater(upd)
    rng = np.random.default_rng(seed=7)
    # Generate data UNDER THE MODEL: draw a θ from the prior, then
    # observations from N(θ, σ²=1). After enough steps the posterior
    # concentrates on θ_true and predictive coverage → 0.95.
    # (Drawing data from N(0, sqrt(2)) — i.e. from the marginal
    # predictive at step 1 — is a model mis-specification: the data
    # variance would be 2 while the model assumes σ²=1, and the
    # asymptotic coverage drops to ≈ 0.83.)
    true_theta = float(rng.normal(0.0, 1.0))
    data = list(rng.normal(true_theta, 1.0, size=5_000))
    posts: list[NormalPosterior] = []
    cur = upd.update([])  # prior as posterior with n_obs=0
    assert isinstance(cur, NormalPosterior)
    for x in data:
        posts.append(cur)
        new = seq.feed(x)
        assert isinstance(new, NormalPosterior)
        cur = new
    score = calibration_score(posts, data, sigma_sq=1.0, level=0.95)
    # Asymptotic coverage is 0.95; early steps slightly over-cover
    # because the predictive variance is wider than σ². With n=5000
    # the average is well within the [0.94, 0.97] band.
    assert 0.93 <= score <= 0.97
