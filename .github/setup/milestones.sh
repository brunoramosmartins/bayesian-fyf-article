#!/bin/bash
set -euo pipefail
REPO="${1:?Usage: bash milestones.sh owner/repo}"

echo "Creating milestones for $REPO..."

gh api "repos/$REPO/milestones" -f title="Phase 0 — Foundation" \
  -f description="Thesis, FYF model design, project scaffold." --silent
gh api "repos/$REPO/milestones" -f title="Phase 1 — Bayesian Foundations" \
  -f description="Bayes' theorem, priors, posteriors, credible intervals." --silent
gh api "repos/$REPO/milestones" -f title="Phase 2 — Conjugate Families" \
  -f description="Normal-Normal, NIG, Gamma-Poisson, Beta-Binomial derivations." --silent
gh api "repos/$REPO/milestones" -f title="Phase 3 — Sequential Updating" \
  -f description="Sequential=batch proof, shrinkage, prior sensitivity." --silent
gh api "repos/$REPO/milestones" -f title="Phase 4 — Predictive Inference" \
  -f description="Posterior predictive, year-end forecast, Bayes factors." --silent
gh api "repos/$REPO/milestones" -f title="Phase 5 — Applied FYF Model" \
  -f description="Full FYF model, 5 scenarios, diagnostics." --silent
gh api "repos/$REPO/milestones" -f title="Phase 6 — Experiments & Visualizations" \
  -f description="All experiments, publication figures, animated GIF." --silent
gh api "repos/$REPO/milestones" -f title="Phase 7 — Article Writing" \
  -f description="Full article in English and Portuguese." --silent
gh api "repos/$REPO/milestones" -f title="Phase 8 — Review & Publish" \
  -f description="Validation, reproducibility, publication." --silent

echo "All milestones created."
