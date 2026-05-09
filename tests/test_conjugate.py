"""Tests for src/conjugate.py — DO NOT auto-run.

The author runs ``pytest tests/`` and ``ruff check .`` after each
phase. Tests cover:

- closed-form correctness against textbook-simple cases (small integer
  hyperparameters where the answer is computable by hand);
- the FYF reference scenario from ``docs/model-design.md``;
- monotone shrinkage of posterior variance with n;
- asymptotic concentration on the data;
- input validation contracts.
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


# =============================================================================
# Normal-Normal
# =============================================================================


class TestNormalNormalUpdater:
    def test_empty_data_returns_prior(self) -> None:
        upd = NormalNormalUpdater(mu0=10.0, sigma0_sq=4.0, sigma_sq=1.0)
        post = upd.update([])
        assert post.mean() == pytest.approx(10.0)
        assert post.variance() == pytest.approx(4.0)
        assert post.n_obs == 0

    def test_textbook_case_n_eq_1(self) -> None:
        # Prior N(0, 1), σ²=1, x=2 ⟹ posterior N(1, 0.5).
        upd = NormalNormalUpdater(mu0=0.0, sigma0_sq=1.0, sigma_sq=1.0)
        post = upd.update([2.0])
        assert post.mean() == pytest.approx(1.0, rel=1e-12)
        assert post.variance() == pytest.approx(0.5, rel=1e-12)

    def test_precision_additivity(self) -> None:
        # τ_n = τ_0 + n·τ should hold exactly (modulo float).
        upd = NormalNormalUpdater(mu0=10.0, sigma0_sq=4.0, sigma_sq=1.0)
        for n in (1, 5, 20, 100):
            data = np.full(n, 12.0)
            post = upd.update(data)
            tau0 = 1.0 / 4.0
            tau = 1.0 / 1.0
            assert post.precision() == pytest.approx(tau0 + n * tau, rel=1e-12)

    def test_posterior_variance_monotone_in_n(self) -> None:
        upd = NormalNormalUpdater(mu0=0.0, sigma0_sq=1.0, sigma_sq=1.0)
        rng = np.random.default_rng(seed=42)
        data = rng.normal(loc=2.0, scale=1.0, size=200)
        variances = [upd.update(data[:n]).variance() for n in range(1, 200)]
        diffs = np.diff(variances)
        assert np.all(diffs <= 0.0), "posterior variance must be non-increasing in n"

    def test_posterior_mean_converges_to_sample_mean(self) -> None:
        # For large n, μ_n should equal the sample mean to high precision.
        upd = NormalNormalUpdater(mu0=0.0, sigma0_sq=1.0, sigma_sq=1.0)
        rng = np.random.default_rng(seed=7)
        data = rng.normal(loc=5.0, scale=2.0, size=10_000)
        post = upd.update(data)
        assert post.mean() == pytest.approx(float(data.mean()), rel=1e-3)

    def test_fyf_reference_q1(self) -> None:
        # docs/model-design.md §5: Q1 actuals
        upd = NormalNormalUpdater(
            mu0=1_050_000, sigma0_sq=150_000**2, sigma_sq=80_000**2
        )
        post = upd.update([1_120_000, 1_080_000, 1_095_000])
        # σ_3² = σ²σ₀²/(σ²+3σ₀²)
        expected_var = (80_000**2 * 150_000**2) / (80_000**2 + 3 * 150_000**2)
        assert post.variance() == pytest.approx(expected_var, rel=1e-12)
        # μ_3 closed-form
        x_bar = (1_120_000 + 1_080_000 + 1_095_000) / 3.0
        expected_mu = (80_000**2 * 1_050_000 + 3 * 150_000**2 * x_bar) / (
            80_000**2 + 3 * 150_000**2
        )
        assert post.mean() == pytest.approx(expected_mu, rel=1e-12)

    def test_credible_interval_is_symmetric_for_normal(self) -> None:
        post = NormalPosterior(mu=10.0, sigma_sq=4.0, n_obs=5)
        lo, hi = post.credible_interval(0.95)
        assert (lo + hi) / 2.0 == pytest.approx(10.0, rel=1e-12)
        assert hi - lo == pytest.approx(2.0 * 1.959963984540054 * 2.0, rel=1e-6)

    def test_summary_keys(self) -> None:
        post = NormalPosterior(mu=0.0, sigma_sq=1.0, n_obs=3)
        s = post.summary(level=0.90)
        assert set(s.keys()) >= {
            "mean",
            "variance",
            "std",
            "precision",
            "credible_interval",
            "credible_level",
            "n_obs",
        }

    @pytest.mark.parametrize("sigma0_sq", [0.0, -1.0])
    def test_invalid_prior_variance_raises(self, sigma0_sq: float) -> None:
        with pytest.raises(ValueError, match="sigma0_sq"):
            NormalNormalUpdater(mu0=0.0, sigma0_sq=sigma0_sq, sigma_sq=1.0)

    @pytest.mark.parametrize("sigma_sq", [0.0, -1.0])
    def test_invalid_sampling_variance_raises(self, sigma_sq: float) -> None:
        with pytest.raises(ValueError, match="sigma_sq"):
            NormalNormalUpdater(mu0=0.0, sigma0_sq=1.0, sigma_sq=sigma_sq)


# =============================================================================
# Gamma-Poisson
# =============================================================================


class TestGammaPoissonUpdater:
    def test_empty_data_returns_prior(self) -> None:
        upd = GammaPoissonUpdater(alpha0=3.0, beta0=1.0)
        post = upd.update([])
        assert post.alpha == pytest.approx(3.0)
        assert post.beta == pytest.approx(1.0)
        assert post.n_obs == 0

    def test_textbook_case(self) -> None:
        # Prior Gamma(2, 1), x=[3, 5, 7] ⟹ posterior Gamma(17, 4).
        upd = GammaPoissonUpdater(alpha0=2.0, beta0=1.0)
        post = upd.update([3, 5, 7])
        assert post.alpha == pytest.approx(17.0, rel=1e-12)
        assert post.beta == pytest.approx(4.0, rel=1e-12)
        assert post.mean() == pytest.approx(17.0 / 4.0, rel=1e-12)
        assert post.variance() == pytest.approx(17.0 / 16.0, rel=1e-12)

    def test_pseudo_observation_interpretation(self) -> None:
        # Prior Gamma(3, 1) (3 events / 1 month), data {2,4,1,3,5,2}.
        # Posterior Gamma(20, 7).
        upd = GammaPoissonUpdater(alpha0=3.0, beta0=1.0)
        post = upd.update([2, 4, 1, 3, 5, 2])
        assert post.alpha == pytest.approx(20.0)
        assert post.beta == pytest.approx(7.0)
        assert post.mean() == pytest.approx(20.0 / 7.0)

    def test_posterior_concentrates_with_more_data(self) -> None:
        upd = GammaPoissonUpdater(alpha0=1.0, beta0=1.0)
        rng = np.random.default_rng(seed=2026)
        data = rng.poisson(lam=2.5, size=5_000)
        post = upd.update(data)
        assert post.mean() == pytest.approx(2.5, rel=2e-2)
        assert post.std() < 0.05  # should be very tight

    def test_negative_count_raises(self) -> None:
        upd = GammaPoissonUpdater(alpha0=1.0, beta0=1.0)
        with pytest.raises(ValueError, match="non-negative"):
            upd.update([1, 2, -3])

    def test_non_integer_count_raises(self) -> None:
        upd = GammaPoissonUpdater(alpha0=1.0, beta0=1.0)
        with pytest.raises(ValueError, match="integers"):
            upd.update([1, 2.5, 3])

    @pytest.mark.parametrize(
        ("alpha", "beta"),
        [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)],
    )
    def test_invalid_prior_raises(self, alpha: float, beta: float) -> None:
        with pytest.raises(ValueError):
            GammaPoissonUpdater(alpha0=alpha, beta0=beta)

    def test_credible_interval_in_unit_range(self) -> None:
        post = GammaPosterior(alpha=5.0, beta=2.0)
        lo, hi = post.credible_interval(0.95)
        assert 0.0 < lo < hi
        # Mean = 2.5; 95 % CI must straddle the mean for a unimodal Gamma
        # with these parameters.
        assert lo < post.mean() < hi


# =============================================================================
# Beta-Binomial
# =============================================================================


class TestBetaBinomialUpdater:
    def test_empty_data_returns_prior(self) -> None:
        upd = BetaBinomialUpdater(alpha0=2.0, beta0=2.0)
        post = upd.update(successes=[], trials=[])
        assert post.alpha == pytest.approx(2.0)
        assert post.beta == pytest.approx(2.0)
        assert post.n_trials == 0

    def test_textbook_case(self) -> None:
        # Beta(1,1) + 7 successes in 10 ⟹ Beta(8, 4).
        upd = BetaBinomialUpdater(alpha0=1.0, beta0=1.0)
        post = upd.update(successes=7, trials=10)
        assert post.alpha == pytest.approx(8.0)
        assert post.beta == pytest.approx(4.0)
        assert post.mean() == pytest.approx(8.0 / 12.0)

    def test_multiple_batches_accumulate(self) -> None:
        # Two independent batches: 3/5 and 4/5 ⟹ same as 7/10.
        upd_split = BetaBinomialUpdater(alpha0=1.0, beta0=1.0)
        post_split = upd_split.update(successes=[3, 4], trials=[5, 5])
        upd_one = BetaBinomialUpdater(alpha0=1.0, beta0=1.0)
        post_one = upd_one.update(successes=7, trials=10)
        assert post_split.alpha == pytest.approx(post_one.alpha)
        assert post_split.beta == pytest.approx(post_one.beta)

    def test_overtime_default_prior(self) -> None:
        # docs/model-design.md: Beta(2, 8) for 20 % overtime.
        upd = BetaBinomialUpdater(alpha0=2.0, beta0=8.0)
        post = upd.update(successes=12, trials=50)
        assert post.alpha == pytest.approx(14.0)
        assert post.beta == pytest.approx(46.0)
        assert post.mean() == pytest.approx(14.0 / 60.0)

    def test_successes_cannot_exceed_trials(self) -> None:
        upd = BetaBinomialUpdater(alpha0=1.0, beta0=1.0)
        with pytest.raises(ValueError, match="successes cannot exceed trials"):
            upd.update(successes=11, trials=10)

    def test_negative_inputs_raise(self) -> None:
        upd = BetaBinomialUpdater(alpha0=1.0, beta0=1.0)
        with pytest.raises(ValueError, match="non-negative"):
            upd.update(successes=-1, trials=10)

    def test_shape_mismatch_raises(self) -> None:
        upd = BetaBinomialUpdater(alpha0=1.0, beta0=1.0)
        with pytest.raises(ValueError, match="same shape"):
            upd.update(successes=[1, 2], trials=[3, 4, 5])

    def test_credible_interval_in_unit_interval(self) -> None:
        post = BetaPosterior(alpha=8.0, beta=4.0)
        lo, hi = post.credible_interval(0.95)
        assert 0.0 < lo < post.mean() < hi < 1.0


# =============================================================================
# Normal-Inverse-Gamma
# =============================================================================


class TestNormalInverseGammaUpdater:
    def test_empty_data_returns_prior(self) -> None:
        upd = NormalInverseGammaUpdater(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        post = upd.update([])
        assert (post.mu, post.kappa, post.alpha, post.beta) == pytest.approx(
            (0.0, 1.0, 2.0, 1.0)
        )

    def test_simple_closed_form(self) -> None:
        # Prior NIG(0, 1, 2, 1), x = [1, 2, 3].
        # n=3, x_bar=2, S = (1-2)²+0+(3-2)² = 2.
        # κ_n = 1+3 = 4
        # μ_n = (1·0 + 3·2)/4 = 1.5
        # α_n = 2 + 3/2 = 3.5
        # β_n = 1 + 0.5·2 + 0.5·(1·3/4)·(2-0)² = 1 + 1 + 1.5 = 3.5
        upd = NormalInverseGammaUpdater(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        post = upd.update([1.0, 2.0, 3.0])
        assert post.mu == pytest.approx(1.5, rel=1e-12)
        assert post.kappa == pytest.approx(4.0, rel=1e-12)
        assert post.alpha == pytest.approx(3.5, rel=1e-12)
        assert post.beta == pytest.approx(3.5, rel=1e-12)

    def test_marginal_concentrates_on_data(self) -> None:
        # As n → ∞ with iid N(theta_star, sigma_star²) data, the marginal
        # mean of theta should approach theta_star.
        upd = NormalInverseGammaUpdater(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        rng = np.random.default_rng(seed=11)
        data = rng.normal(loc=5.0, scale=1.5, size=2_000)
        post = upd.update(data)
        # μ_n approaches x_bar (which approaches 5 by LLN)
        assert post.mu == pytest.approx(float(data.mean()), rel=1e-3)
        # σ² posterior mean approaches sample variance, which approaches 1.5²
        assert post.mean_sigma_sq() == pytest.approx(1.5**2, rel=5e-2)

    def test_credible_interval_uses_t_marginal(self) -> None:
        # For NIG(μ=0, κ=1, α=10, β=10), the marginal t has df=20, scale=1.
        # 95% CI ≈ μ ± t_{0.975, df=20} · scale ≈ ±2.086.
        post = NormalInverseGammaPosterior(mu=0.0, kappa=1.0, alpha=10.0, beta=10.0)
        lo, hi = post.credible_interval(0.95)
        assert (hi - lo) / 2.0 == pytest.approx(2.086, rel=1e-3)

    def test_summary_includes_sigma_marginal_when_defined(self) -> None:
        post = NormalInverseGammaPosterior(mu=0.0, kappa=1.0, alpha=3.0, beta=2.0)
        s = post.summary()
        assert "sigma_sq_mean" in s
        assert s["sigma_sq_mean"] == pytest.approx(2.0 / (3.0 - 1.0))

    def test_summary_omits_sigma_marginal_when_undefined(self) -> None:
        # alpha = 1 ⟹ Inverse-Gamma mean undefined.
        post = NormalInverseGammaPosterior(mu=0.0, kappa=1.0, alpha=1.0, beta=2.0)
        s = post.summary()
        assert "sigma_sq_mean" not in s

    @pytest.mark.parametrize(
        ("kappa", "alpha", "beta"),
        [
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 1.0),
        ],
    )
    def test_invalid_hyperparams_raise(
        self, kappa: float, alpha: float, beta: float
    ) -> None:
        with pytest.raises(ValueError):
            NormalInverseGammaUpdater(
                mu0=0.0, kappa0=kappa, alpha0=alpha, beta0=beta
            )


# =============================================================================
# Cross-pair sanity: shrinkage trajectory matches exercise 7
# =============================================================================


def test_exercise_7_trajectory_matches_paper() -> None:
    """``exercises/ex02_conjugate_families.md`` exercise 7 expected values."""
    upd = NormalNormalUpdater(
        mu0=1_050_000, sigma0_sq=150_000**2, sigma_sq=80_000**2
    )
    cases = [
        (1, 1_120_000, 1_104_500, 70_600),
        (3, 1_095_000, 1_091_100, 44_100),
        (6, 1_085_000, 1_083_400, 31_900),
        (12, 1_078_000, 1_077_400, 22_800),
    ]
    for n, x_bar, exp_mu, exp_sigma in cases:
        # Each row uses its own n with a given sample mean — simulate by
        # passing a constant array of length n with that mean.
        post = upd.update(np.full(n, float(x_bar)))
        assert post.mean() == pytest.approx(exp_mu, abs=200), (
            f"n={n}: μ off by more than R$ 200"
        )
        assert post.std() == pytest.approx(exp_sigma, abs=200), (
            f"n={n}: σ off by more than R$ 200"
        )


def test_exercise_8_gamma_poisson_matches_paper() -> None:
    """``exercises/ex02_conjugate_families.md`` exercise 8."""
    upd = GammaPoissonUpdater(alpha0=3.0, beta0=1.0)
    post = upd.update([2, 4, 1, 3, 5, 2])
    assert post.alpha == pytest.approx(20.0)
    assert post.beta == pytest.approx(7.0)
    assert post.mean() == pytest.approx(20.0 / 7.0)
    # Shrinkage weight w0 = beta0 / (beta0 + n) = 1/7
    w0 = 1.0 / 7.0
    expected = w0 * 3.0 + (1.0 - w0) * (17.0 / 6.0)
    assert post.mean() == pytest.approx(expected, rel=1e-12)


# =============================================================================
# Module-level
# =============================================================================


def test_math_module_used_consistently() -> None:
    """Smoke test: posteriors must not return NaN/Inf for valid inputs."""
    nn = NormalNormalUpdater(mu0=0.0, sigma0_sq=1.0, sigma_sq=1.0).update([1.0, 2.0])
    gp = GammaPoissonUpdater(alpha0=1.0, beta0=1.0).update([1, 2, 3])
    bb = BetaBinomialUpdater(alpha0=1.0, beta0=1.0).update(successes=3, trials=5)
    nig = NormalInverseGammaUpdater(
        mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0
    ).update([1.0, 2.0, 3.0])

    for post in (nn, gp, bb, nig):
        assert math.isfinite(post.mean())
        assert math.isfinite(post.variance())
