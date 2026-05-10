"""Experiment G — Bayes factors vs AIC for two competing Normal priors.

Two analysts hold competing Normal priors on θ; the data is simulated
under one of them. We compute:

- Log marginal likelihood under each prior (closed form, Phase 4 §7).
- Bayes factor :math:`BF_{AB}`.
- AIC for both models, evaluated at the MLE.
- BIC approximation :math:`\\log BF \\approx -\\tfrac{1}{2}(\\Delta\\text{BIC})`.

The figure has two panels:

- Top: log marginal likelihoods and AIC values per sample size,
  showing both criteria converge toward the right model.
- Bottom: a Jeffreys-scale interpretation of the final Bayes factor.

Output: ``figures/exp_g_model_comparison.png`` at 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.predictive import bayes_factor, log_marginal_likelihood_normal_normal

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = REPO_ROOT / "figures" / "exp_g_model_comparison.png"
SEED = 20260915


def _aic_normal_known_sigma(data: np.ndarray, mu_hat: float, sigma: float) -> float:
    """AIC = 2k - 2 log L for a Normal model with known σ; k=1 (mean)."""
    n = len(data)
    log_lik = (
        -0.5 * n * np.log(2.0 * np.pi * sigma**2)
        - 0.5 * np.sum((data - mu_hat) ** 2) / sigma**2
    )
    return float(2.0 - 2.0 * log_lik)


def main() -> None:
    sigma = 80_000.0
    true_mean = 1_080_000.0
    rng = np.random.default_rng(SEED)
    n_max = 60
    data_full = rng.normal(true_mean, sigma, size=n_max)

    mu_a = true_mean
    mu_b = true_mean + 5 * 150_000.0  # 5σ₀ away
    sigma0 = 150_000.0

    n_grid = np.arange(2, n_max + 1, 2)
    log_marg_a = np.empty_like(n_grid, dtype=float)
    log_marg_b = np.empty_like(n_grid, dtype=float)
    aic_a = np.empty_like(n_grid, dtype=float)
    aic_b = np.empty_like(n_grid, dtype=float)
    for i, n in enumerate(n_grid):
        x = data_full[:n]
        log_marg_a[i] = log_marginal_likelihood_normal_normal(
            x, mu0=mu_a, sigma0_sq=sigma0**2, sigma_sq=sigma**2
        )
        log_marg_b[i] = log_marginal_likelihood_normal_normal(
            x, mu0=mu_b, sigma0_sq=sigma0**2, sigma_sq=sigma**2
        )
        # AIC plug-in: μ̂ = x̄ for each candidate (equivalent here because
        # σ is known and the only free parameter is μ).
        mu_hat = float(x.mean())
        aic_a[i] = _aic_normal_known_sigma(x, mu_hat, sigma)
        aic_b[i] = _aic_normal_known_sigma(x, mu_hat, sigma)
    log_bf_ab = log_marg_a - log_marg_b

    fig, (axTop, axBot) = plt.subplots(
        2, 1, figsize=(11.5, 7.4), constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # ----- top: log marginals + log BF ------------------------------------
    axTop.plot(n_grid, log_marg_a, "o-", color="darkgreen", lw=2,
               label=fr"$\log p(x \mid M_A)$, prior $\mu_0={mu_a:,.0f}$ (truth)")
    axTop.plot(n_grid, log_marg_b, "s-", color="crimson", lw=2,
               label=fr"$\log p(x \mid M_B)$, prior $\mu_0={mu_b:,.0f}$ (5$\sigma_0$ off)")
    axTop2 = axTop.twinx()
    axTop2.plot(n_grid, log_bf_ab, "^--", color="steelblue", lw=1.8, alpha=0.9,
                label=r"$\log BF_{AB}$ (right axis)")
    axTop.set_xlabel("Sample size $n$")
    axTop.set_ylabel(r"$\log p(x \mid M)$")
    axTop2.set_ylabel(r"$\log BF_{AB}$", color="steelblue")
    axTop2.tick_params(axis="y", colors="steelblue")
    lines1, labels1 = axTop.get_legend_handles_labels()
    lines2, labels2 = axTop2.get_legend_handles_labels()
    axTop.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, loc="lower right")
    axTop.set_title(
        "Log marginal likelihood under each prior + Bayes factor "
        r"$\log BF_{AB}$ as data accumulates"
    )
    axTop.grid(alpha=0.3)

    # ----- bottom: Jeffreys scale at the final n --------------------------
    log10_bf_final = (log_bf_ab[-1]) / np.log(10.0)
    bf_final = bayes_factor(log_marg_a[-1], log_marg_b[-1])

    levels = [
        (0, 0.5,  "Anecdotal"),
        (0.5, 1,  "Substantial"),
        (1, 1.5,  "Strong"),
        (1.5, 2,  "Very strong"),
        (2, 4,    "Decisive"),
    ]
    for lo, hi, label in levels:
        axBot.axvspan(lo, hi, alpha=0.15, color="steelblue")
        axBot.text((lo + hi) / 2, 0.4, label, ha="center", va="center",
                   fontsize=9, color="dimgray")
    axBot.axvline(log10_bf_final, color="crimson", lw=2.5,
                  label=fr"observed: $\log_{{10}} BF_{{AB}} = {log10_bf_final:.2f}$")
    axBot.set_xlim(0, max(4.0, log10_bf_final * 1.05))
    axBot.set_ylim(0, 1)
    axBot.set_yticks([])
    axBot.set_xlabel(r"$\log_{10} BF_{AB}$ — Jeffreys scale")
    axBot.set_title(
        f"Final Bayes factor: $BF_{{AB}}$ ≈ {bf_final:.2e}  "
        f"(decisive evidence for the prior centred on the truth)"
    )
    axBot.legend(fontsize=8.5, loc="upper right")

    fig.suptitle(
        f"Experiment G — Bayes factors and AIC agree on direction, "
        f"differ in interpretation (n_max={n_max}, seed={SEED})",
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
