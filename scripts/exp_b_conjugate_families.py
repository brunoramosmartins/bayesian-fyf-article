"""Experiment B — all four conjugate pairs in a 2×2 grid.

For each pair, plot the prior and the posterior after a small batch
of synthetic data, with the hyperparameter update annotated. The
figure shows that the **same kernel-matching argument** produces
closed-form posteriors across very different distributional families.

Output: ``figures/exp_b_conjugate_families.png`` at 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.conjugate import (
    BetaBinomialUpdater,
    GammaPoissonUpdater,
    NormalInverseGammaUpdater,
    NormalNormalUpdater,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = REPO_ROOT / "figures" / "exp_b_conjugate_families.png"
SEED = 20260910


def _panel_normal_normal(ax: plt.Axes) -> None:
    mu0, sigma0, sigma = 0.0, 1.0, 1.0
    rng = np.random.default_rng(SEED)
    x = rng.normal(0.5, sigma, size=10)
    upd = NormalNormalUpdater(mu0=mu0, sigma0_sq=sigma0**2, sigma_sq=sigma**2)
    post = upd.update(x)
    grid = np.linspace(-3, 3, 600)
    ax.plot(grid, stats.norm(mu0, sigma0).pdf(grid),
            color="steelblue", lw=2, label=r"Prior $N(0, 1)$")
    ax.plot(grid, stats.norm(post.mean(), post.std()).pdf(grid),
            color="crimson", lw=2,
            label=fr"Posterior $N({post.mean():.2f}, {post.variance():.3f})$")
    ax.set_title("Normal–Normal (mean, σ² known)")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("density")
    ax.legend(fontsize=8.5)


def _panel_normal_inverse_gamma(ax: plt.Axes) -> None:
    upd = NormalInverseGammaUpdater(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
    rng = np.random.default_rng(SEED + 1)
    x = rng.normal(0.7, 1.2, size=15)
    post = upd.update(x)

    # Marginal posterior on θ is Student-t with df=2α, loc=μ, scale=√(β/(ακ))
    df_pri, df_post = 2 * upd.alpha0, 2 * post.alpha
    scale_pri = np.sqrt(upd.beta0 / (upd.alpha0 * upd.kappa0))
    scale_post = np.sqrt(post.beta / (post.alpha * post.kappa))

    grid = np.linspace(-3, 3, 600)
    ax.plot(grid, stats.t(df=df_pri, loc=upd.mu0, scale=scale_pri).pdf(grid),
            color="steelblue", lw=2,
            label=fr"Prior marginal  $t_{{{df_pri:.0f}}}(0, {scale_pri:.2f})$")
    ax.plot(grid, stats.t(df=df_post, loc=post.mu, scale=scale_post).pdf(grid),
            color="crimson", lw=2,
            label=fr"Posterior marginal  $t_{{{df_post:.0f}}}({post.mu:.2f}, {scale_post:.2f})$")
    ax.set_title("Normal–Inverse-Gamma (marginal on θ is Student-t)")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("density")
    ax.legend(fontsize=8.5)


def _panel_gamma_poisson(ax: plt.Axes) -> None:
    upd = GammaPoissonUpdater(alpha0=3.0, beta0=1.0)
    rng = np.random.default_rng(SEED + 2)
    x = rng.poisson(2.5, size=8)
    post = upd.update(x)
    grid = np.linspace(0, 8, 600)
    ax.plot(grid, stats.gamma(upd.alpha0, scale=1.0 / upd.beta0).pdf(grid),
            color="steelblue", lw=2,
            label=fr"Prior  Gamma({upd.alpha0:.0f}, {upd.beta0:.0f})")
    ax.plot(grid, stats.gamma(post.alpha, scale=1.0 / post.beta).pdf(grid),
            color="crimson", lw=2,
            label=fr"Posterior  Gamma({post.alpha:.0f}, {post.beta:.0f})")
    ax.set_title("Gamma–Poisson (rate)")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("density")
    ax.legend(fontsize=8.5)


def _panel_beta_binomial(ax: plt.Axes) -> None:
    upd = BetaBinomialUpdater(alpha0=2.0, beta0=8.0)
    rng = np.random.default_rng(SEED + 3)
    trials = 50
    successes = int(rng.binomial(trials, 0.30))
    post = upd.update(successes=successes, trials=trials)
    grid = np.linspace(0.001, 0.999, 600)
    ax.plot(grid, stats.beta(upd.alpha0, upd.beta0).pdf(grid),
            color="steelblue", lw=2,
            label=fr"Prior  Beta({upd.alpha0:.0f}, {upd.beta0:.0f})")
    ax.plot(grid, stats.beta(post.alpha, post.beta).pdf(grid),
            color="crimson", lw=2,
            label=fr"Posterior  Beta({post.alpha:.0f}, {post.beta:.0f})")
    ax.set_title("Beta–Binomial (proportion)")
    ax.set_xlabel(r"$p$")
    ax.set_ylabel("density")
    ax.legend(fontsize=8.5)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    _panel_normal_normal(axes[0, 0])
    _panel_normal_inverse_gamma(axes[0, 1])
    _panel_gamma_poisson(axes[1, 0])
    _panel_beta_binomial(axes[1, 1])
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Experiment B — four conjugate pairs, one update rule "
        "(prior × likelihood ⟶ same-family posterior)",
        fontsize=12,
    )
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300)
    print(
        f"Saved {FIG_PATH.relative_to(REPO_ROOT)}  "
        f"({FIG_PATH.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
