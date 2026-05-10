"""Generate one PNG per Phase-5 canonical scenario.

Outputs (under ``figures/``):

- ``fyf_cycle_scenario_s1_on_target.png``
- ``fyf_cycle_scenario_s2_optimistic.png``
- ``fyf_cycle_scenario_s3_shock.png``
- ``fyf_cycle_scenario_s4_seasonal.png``
- ``fyf_cycle_scenario_s5_prior_sensitivity.png``

For S1–S4 the figure has four panels — posterior trajectory, surprise
z-scores, year-end total forecast, P(over-budget) — driven by the
shared visualisation helpers in ``src/visualization.py``. S5 is a
two-panel comparison of the confident vs uncertain priors.

Usage
-----
From the repository root::

    python scripts/fig_fyf_scenarios.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.fyf_model import (
    FYFModel,
    Scenario,
    build_scenarios,
    run_scenario,
    run_scenario_secondary,
)
from src.visualization import (
    plot_fyf_comparison,
    plot_p_over_budget,
    plot_posterior_evolution,
    plot_surprise_scores,
    plot_year_end_forecast,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "figures"


def _save(fig: plt.Figure, scenario: Scenario) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"fyf_cycle_scenario_{scenario.id}.png"
    fig.savefig(out, dpi=300)
    print(f"  saved {out.relative_to(REPO_ROOT)}  ({out.stat().st_size / 1024:.1f} KB)")
    plt.close(fig)


def _figure_four_panels(scenario: Scenario, model: FYFModel) -> plt.Figure:
    config = scenario.config
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), constrained_layout=True)
    plot_posterior_evolution(
        axes[0, 0],
        model.reviews(),
        prior_mean=config.prior_mean,
        title=f"{scenario.name} — posterior trajectory",
    )
    plot_surprise_scores(
        axes[0, 1],
        model.reviews(),
        title=f"{scenario.name} — surprise z-scores",
    )
    plot_year_end_forecast(
        axes[1, 0],
        model.reviews(),
        budget_ceiling=config.budget_ceiling,
        title=f"{scenario.name} — year-end total forecast",
    )
    plot_p_over_budget(
        axes[1, 1],
        model.reviews(),
        title=f"{scenario.name} — P(annual total > B)",
    )
    fig.suptitle(
        f"{scenario.id.upper()} · {scenario.name}\n"
        f"{scenario.description}",
        fontsize=11.5,
    )
    return fig


def _figure_prior_sensitivity(
    scenario: Scenario,
    model_primary: FYFModel,
    model_secondary: FYFModel,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    plot_fyf_comparison(
        axes[0],
        models=[model_primary, model_secondary],
        labels=[
            fr"Confident prior  $\sigma_0={model_primary.config.prior_sd:,.0f}$",
            fr"Uncertain prior  $\sigma_0={model_secondary.config.prior_sd:,.0f}$",
        ],
        title="Posterior mean trajectory",
    )

    rev_a = model_primary.reviews()
    rev_b = model_secondary.reviews()
    months = [r.month for r in rev_a]
    means_a = [r.posterior.mean() for r in rev_a]
    means_b = [r.posterior.mean() for r in rev_b]
    stds_a = [r.posterior.std() for r in rev_a]
    stds_b = [r.posterior.std() for r in rev_b]

    ax = axes[1]
    ax.plot(months, [a - b for a, b in zip(means_a, means_b, strict=True)],
            "o-", color="crimson", lw=2,
            label=r"Mean gap $\mu_n^{\mathrm{conf}}-\mu_n^{\mathrm{unc}}$")
    ax.plot(months, [a - b for a, b in zip(stds_a, stds_b, strict=True)],
            "s--", color="darkgreen", lw=1.5,
            label=r"S.d. gap $\sigma_n^{\mathrm{conf}}-\sigma_n^{\mathrm{unc}}$")
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xlabel("Month $n$")
    ax.set_ylabel("Difference (R\\$)")
    ax.set_xticks(months)
    ax.ticklabel_format(style="plain", axis="y")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(r"Disagreement between priors $\to 0$")

    fig.suptitle(
        f"{scenario.id.upper()} · {scenario.name}\n"
        f"{scenario.description}",
        fontsize=11.5,
    )
    return fig


def main() -> None:
    print("Running 5 canonical FYF scenarios...")
    for scenario in build_scenarios():
        print(f"\n[{scenario.id}] {scenario.name}")
        if scenario.id == "s5_prior_sensitivity":
            model_primary = run_scenario(scenario)
            model_secondary = run_scenario_secondary(scenario)
            assert model_secondary is not None
            fig = _figure_prior_sensitivity(scenario, model_primary, model_secondary)
        else:
            model = run_scenario(scenario)
            fig = _figure_four_panels(scenario, model)
            qr = model.fyf_review(4)
            print(
                f"  Year-end posterior: μ={qr.posterior.mean():,.0f}, "
                f"σ={qr.posterior.std():,.0f}; recommendation: {qr.recommendation}"
            )
        _save(fig, scenario)
    print("\nDone.")


if __name__ == "__main__":
    main()
