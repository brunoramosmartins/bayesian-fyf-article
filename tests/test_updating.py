"""Tests for src/updating.py — DO NOT auto-run.

The author runs ``pytest tests/`` and ``ruff check .`` after each
phase. Tests cover:

- the central sequential = batch identity (Phase 3 theorem) for all
  four conjugate pairs;
- history bookkeeping (length, prior_weight monotonicity);
- the closed-form shrinkage helpers ``normal_normal_shrinkage_weight``
  and ``months_to_data_weight`` against §2.3 of the theory notes;
- ``reset()`` returns the engine to its starting state;
- the recursive Kalman-gain form for Normal-Normal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

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
from src.updating import (
    SequentialUpdater,
    months_to_data_weight,
    normal_normal_shrinkage_weight,
)


# =============================================================================
# Sequential = Batch theorem
# =============================================================================


class TestSequentialEqualsBatch:
    def test_normal_normal(self) -> None:
        prior = NormalNormalUpdater(
            mu0=1_050_000, sigma0_sq=150_000**2, sigma_sq=80_000**2
        )
        data = [1_120_000.0, 1_080_000.0, 1_095_000.0, 1_103_000.0, 1_088_000.0]
        seq = SequentialUpdater(prior)
        post_seq = seq.feed_batch(data)
        post_bat = prior.update(data)
        assert isinstance(post_seq, NormalPosterior)
        assert isinstance(post_bat, NormalPosterior)
        assert post_seq.mean() == pytest.approx(post_bat.mean(), rel=1e-12)
        assert post_seq.variance() == pytest.approx(post_bat.variance(), rel=1e-12)

    def test_gamma_poisson(self) -> None:
        prior = GammaPoissonUpdater(alpha0=3.0, beta0=1.0)
        data = [2, 4, 1, 3, 5, 2]
        seq = SequentialUpdater(prior)
        post_seq = seq.feed_batch(data)
        post_bat = prior.update(data)
        assert isinstance(post_seq, GammaPosterior)
        assert post_seq.alpha == pytest.approx(post_bat.alpha, rel=1e-12)
        assert post_seq.beta == pytest.approx(post_bat.beta, rel=1e-12)

    def test_beta_binomial(self) -> None:
        prior = BetaBinomialUpdater(alpha0=2.0, beta0=8.0)
        batches = [(3, 5), (4, 5), (2, 5)]
        seq = SequentialUpdater(prior)
        post_seq = seq.feed_batch(batches)
        successes = [s for s, _ in batches]
        trials = [t for _, t in batches]
        post_bat = prior.update(successes=successes, trials=trials)
        assert isinstance(post_seq, BetaPosterior)
        assert post_seq.alpha == pytest.approx(post_bat.alpha, rel=1e-12)
        assert post_seq.beta == pytest.approx(post_bat.beta, rel=1e-12)

    def test_normal_inverse_gamma(self) -> None:
        prior = NormalInverseGammaUpdater(
            mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0
        )
        rng = np.random.default_rng(seed=42)
        data = list(rng.normal(loc=2.0, scale=1.5, size=20))
        seq = SequentialUpdater(prior)
        post_seq = seq.feed_batch(data)
        post_bat = prior.update(data)
        assert isinstance(post_seq, NormalInverseGammaPosterior)
        assert post_seq.mu == pytest.approx(post_bat.mu, rel=1e-10)
        assert post_seq.kappa == pytest.approx(post_bat.kappa, rel=1e-12)
        assert post_seq.alpha == pytest.approx(post_bat.alpha, rel=1e-12)
        # NIG beta_n has a more involved formula; relax the tolerance.
        assert post_seq.beta == pytest.approx(post_bat.beta, rel=1e-9)


# =============================================================================
# History and current_posterior
# =============================================================================


class TestHistoryBookkeeping:
    def test_history_length_matches_step_count(self) -> None:
        prior = NormalNormalUpdater(mu0=0.0, sigma0_sq=1.0, sigma_sq=1.0)
        seq = SequentialUpdater(prior)
        seq.feed_batch([1.0, 2.0, 3.0, 4.0])
        assert len(seq.history()) == 4
        assert seq.n_steps == 4

    def test_history_steps_are_one_indexed(self) -> None:
        prior = NormalNormalUpdater(mu0=0.0, sigma0_sq=1.0, sigma_sq=1.0)
        seq = SequentialUpdater(prior)
        seq.feed_batch([1.0, 2.0, 3.0])
        steps = [entry.step for entry in seq.history()]
        assert steps == [1, 2, 3]

    def test_current_posterior_is_prior_initially(self) -> None:
        prior = NormalNormalUpdater(mu0=10.0, sigma0_sq=4.0, sigma_sq=1.0)
        seq = SequentialUpdater(prior)
        post = seq.current_posterior()
        assert isinstance(post, NormalPosterior)
        assert post.mean() == pytest.approx(10.0)
        assert post.variance() == pytest.approx(4.0)

    def test_current_posterior_advances_with_feed(self) -> None:
        prior = NormalNormalUpdater(mu0=0.0, sigma0_sq=1.0, sigma_sq=1.0)
        seq = SequentialUpdater(prior)
        seq.feed(2.0)
        # After 1 obs N(0,1) prior + N(theta, 1) likelihood with x=2:
        # μ_1 = 1, σ²_1 = 0.5
        post = seq.current_posterior()
        assert post.mean() == pytest.approx(1.0, rel=1e-12)
        assert post.variance() == pytest.approx(0.5, rel=1e-12)


# =============================================================================
# Shrinkage weights
# =============================================================================


class TestShrinkageWeights:
    def test_weights_match_closed_form_normal_normal(self) -> None:
        prior = NormalNormalUpdater(
            mu0=1_050_000, sigma0_sq=150_000**2, sigma_sq=80_000**2
        )
        data = [1_120_000.0] * 6
        seq = SequentialUpdater(prior)
        seq.feed_batch(data)
        weights = seq.shrinkage_weights()
        for n, w in enumerate(weights, start=1):
            expected = normal_normal_shrinkage_weight(150_000, 80_000, n)
            assert w == pytest.approx(expected, rel=1e-12)

    def test_weights_strictly_decrease(self) -> None:
        prior = NormalNormalUpdater(mu0=0.0, sigma0_sq=1.0, sigma_sq=1.0)
        seq = SequentialUpdater(prior)
        seq.feed_batch(list(range(20)))
        weights = seq.shrinkage_weights()
        diffs = np.diff(weights)
        assert np.all(diffs < 0.0)

    def test_weights_for_gamma_poisson(self) -> None:
        prior = GammaPoissonUpdater(alpha0=3.0, beta0=1.0)
        seq = SequentialUpdater(prior)
        seq.feed_batch([2, 4, 1, 3, 5, 2])
        weights = seq.shrinkage_weights()
        # w(n) = beta0 / (beta0 + n) = 1 / (1 + n)
        expected = [1.0 / (1.0 + n) for n in range(1, 7)]
        for w, exp in zip(weights, expected, strict=True):
            assert w == pytest.approx(exp, rel=1e-12)

    def test_weights_for_beta_binomial_account_for_trials(self) -> None:
        prior = BetaBinomialUpdater(alpha0=2.0, beta0=8.0)
        seq = SequentialUpdater(prior)
        seq.feed_batch([(3, 5), (4, 5)])  # 5 + 5 = 10 trials total
        weights = seq.shrinkage_weights()
        # After step 1 (5 trials): w = 10 / (10 + 5) = 2/3
        # After step 2 (10 trials): w = 10 / (10 + 10) = 1/2
        assert weights[0] == pytest.approx(10.0 / 15.0, rel=1e-12)
        assert weights[1] == pytest.approx(10.0 / 20.0, rel=1e-12)


# =============================================================================
# Posterior variance / Kalman gain
# =============================================================================


class TestPosteriorVarianceMonotone:
    def test_normal_normal_variance_decreases(self) -> None:
        prior = NormalNormalUpdater(mu0=0.0, sigma0_sq=1.0, sigma_sq=1.0)
        seq = SequentialUpdater(prior)
        seq.feed_batch([1.0] * 30)
        variances = [e.posterior.variance() for e in seq.history()]
        diffs = np.diff(variances)
        assert np.all(diffs < 0.0)


class TestKalmanGainRecursion:
    def test_recursive_form_matches_closed_form(self) -> None:
        # μ_n = μ_{n-1} + K_n · (x_n - μ_{n-1}),
        #   K_n = σ²_{n-1} / (σ²_{n-1} + σ²)
        sigma_sq = 4.0
        prior = NormalNormalUpdater(mu0=0.0, sigma0_sq=2.0, sigma_sq=sigma_sq)
        seq = SequentialUpdater(prior)

        rng = np.random.default_rng(seed=7)
        observations = list(rng.normal(loc=3.0, scale=2.0, size=10))

        # Manual Kalman recursion
        mu_manual = 0.0
        var_manual = 2.0
        manual_means = []
        for x in observations:
            K = var_manual / (var_manual + sigma_sq)
            mu_manual = mu_manual + K * (x - mu_manual)
            # Posterior variance: σ²_n = σ²_{n-1} · σ² / (σ²_{n-1} + σ²)
            var_manual = var_manual * sigma_sq / (var_manual + sigma_sq)
            manual_means.append(mu_manual)

        seq.feed_batch(observations)
        engine_means = [e.posterior.mean() for e in seq.history()]

        for me, mm in zip(engine_means, manual_means, strict=True):
            assert me == pytest.approx(mm, rel=1e-12)


# =============================================================================
# reset()
# =============================================================================


class TestReset:
    def test_reset_clears_history_and_state(self) -> None:
        prior = NormalNormalUpdater(mu0=10.0, sigma0_sq=4.0, sigma_sq=1.0)
        seq = SequentialUpdater(prior)
        seq.feed_batch([12.0, 11.0])
        assert seq.n_steps == 2

        seq.reset()
        assert seq.n_steps == 0
        post = seq.current_posterior()
        assert post.mean() == pytest.approx(10.0)
        assert post.variance() == pytest.approx(4.0)

    def test_reset_then_feed_matches_fresh_engine(self) -> None:
        prior = NormalNormalUpdater(mu0=0.0, sigma0_sq=1.0, sigma_sq=1.0)
        data = [1.0, 2.0, 3.0]

        seq_a = SequentialUpdater(prior)
        seq_a.feed_batch([5.0, 6.0])  # arbitrary first run
        seq_a.reset()
        post_a = seq_a.feed_batch(data)

        seq_b = SequentialUpdater(prior)
        post_b = seq_b.feed_batch(data)

        assert post_a.mean() == pytest.approx(post_b.mean(), rel=1e-12)
        assert post_a.variance() == pytest.approx(post_b.variance(), rel=1e-12)


# =============================================================================
# Closed-form helpers
# =============================================================================


class TestShrinkageHelpers:
    def test_shrinkage_weight_matches_table(self) -> None:
        # Table from notes/phase3-sequential-updating.md §2.3
        sigma0, sigma = 150_000, 80_000
        cases = {
            1: 0.2213,
            2: 0.1245,
            3: 0.0866,
            6: 0.0452,
            9: 0.0306,
            12: 0.0231,
        }
        for n, expected in cases.items():
            w = normal_normal_shrinkage_weight(sigma0, sigma, n)
            assert w == pytest.approx(expected, abs=1e-4)

    def test_months_to_data_weight_threshold(self) -> None:
        # 80 % at n = 2; 95 % at n = 6 (Phase 3 §2.3).
        assert months_to_data_weight(150_000, 80_000, 0.80) == 2
        assert months_to_data_weight(150_000, 80_000, 0.95) == 6

    def test_helper_validation(self) -> None:
        with pytest.raises(ValueError):
            normal_normal_shrinkage_weight(-1.0, 80_000, 5)
        with pytest.raises(ValueError):
            months_to_data_weight(150_000, 80_000, 0.0)
        with pytest.raises(ValueError):
            months_to_data_weight(150_000, 80_000, 1.0)


# =============================================================================
# Exercise 7 cross-check
# =============================================================================


def test_exercise_7_two_priors_converge() -> None:
    """exercises/ex03_sequential_updating.md exercise 7."""
    prior_a = NormalNormalUpdater(mu0=1_050_000, sigma0_sq=150_000**2, sigma_sq=80_000**2)
    prior_b = NormalNormalUpdater(mu0=1_200_000, sigma0_sq=150_000**2, sigma_sq=80_000**2)

    # Same data: 6 actuals with mean 1,085,000.
    data = [1_085_000.0] * 6

    seq_a = SequentialUpdater(prior_a)
    seq_b = SequentialUpdater(prior_b)
    post_a = seq_a.feed_batch(data)
    post_b = seq_b.feed_batch(data)

    gap_n = post_b.mean() - post_a.mean()
    gap_0 = 150_000.0
    w_6 = normal_normal_shrinkage_weight(150_000, 80_000, 6)

    # Theorem from Phase 3 §3.1: gap_n = w_0(n) · gap_0
    assert gap_n == pytest.approx(w_6 * gap_0, rel=1e-12)

    # Numerical sanity: gap shrinks to ~ R$ 6,780.
    assert gap_n == pytest.approx(6_780, abs=20)


# =============================================================================
# Module guard
# =============================================================================


def test_unsupported_updater_type_raises() -> None:
    with pytest.raises(TypeError, match="Unsupported"):
        SequentialUpdater("not an updater")  # type: ignore[arg-type]


def test_no_observations_returns_prior_unchanged() -> None:
    prior = NormalNormalUpdater(mu0=5.0, sigma0_sq=1.0, sigma_sq=1.0)
    seq = SequentialUpdater(prior)
    post = seq.feed_batch([])
    assert isinstance(post, NormalPosterior)
    assert post.mean() == pytest.approx(5.0)
    assert post.variance() == pytest.approx(1.0)
    assert math.isfinite(post.mean())
