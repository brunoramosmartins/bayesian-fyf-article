#!/bin/bash
set -euo pipefail
REPO="${1:?Usage: bash issues.sh owner/repo}"

echo "Creating issues for $REPO..."

# Phase 0
gh issue create --repo "$REPO" --title "[Phase 0] Write thesis and define scope" \
  --label "phase:0,type:documentation,priority:high" --milestone "Phase 0 — Foundation" \
  --body "## Context
The thesis argues that FYF revisions are Bayesian updates.

## Tasks
- [ ] Draft thesis (v0.1)
- [ ] Define scope and anti-scope
- [ ] Write abstract

## Definition of Done
- [ ] \`docs/thesis.md\` complete

## References
- Gelman et al., BDA Ch. 1–3"

gh issue create --repo "$REPO" --title "[Phase 0] Design FYF model and prior specification" \
  --label "phase:0,type:documentation,priority:high" --milestone "Phase 0 — Foundation" \
  --body "## Context
The FYF model defines the revision calendar, cost components, and conjugate pairs.

## Tasks
- [ ] Define revision calendar (4 FYF cycles)
- [ ] Define cost model and conjugate pairs
- [ ] Choose default parameters
- [ ] Work through concrete example

## Definition of Done
- [ ] \`docs/model-design.md\` complete

## References
- FYF Model Design in roadmap"

gh issue create --repo "$REPO" --title "[Phase 0] Configure repository, GitHub scripts, and Claude Code rules" \
  --label "phase:0,type:infrastructure,priority:high" --milestone "Phase 0 — Foundation" \
  --body "## Context
Professional repo setup with bash scripts and Claude Code rules.

## Tasks
- [ ] Initialize directories
- [ ] Create \`.claude/CLAUDE.md\`, \`.github/\` templates and scripts
- [ ] Write \`pyproject.toml\` (no requirements.txt)
- [ ] Create \`til/README.md\` with TIL index
- [ ] Write \`README.md\`

## Definition of Done
- [ ] All scripts run, \`pip install -e .\` works

## References
- Roadmap configuration sections"

# Phase 1
gh issue create --repo "$REPO" --title "[Phase 1] Derive Bayes' theorem and posterior properties" \
  --label "phase:1,type:theory,priority:critical" --milestone "Phase 1 — Bayesian Foundations" \
  --body "## Context
Bayes' theorem for continuous parameters is the article's foundation.

## Tasks
- [ ] Derive π(θ|x) ∝ f(x|θ)π(θ)
- [ ] Derive marginal likelihood
- [ ] Define MAP, posterior mean, credible intervals
- [ ] Contrast credible vs confidence intervals
- [ ] Discuss prior elicitation
- [ ] Two worked examples (discrete + Normal-Normal preview)

## Definition of Done
- [ ] All in \`notes/phase1-bayes-foundations.md\`

## References
- Gelman et al., BDA Ch. 1–2"

gh issue create --repo "$REPO" --title "[Phase 1] Implement prior specification module" \
  --label "phase:1,type:code,priority:high" --milestone "Phase 1 — Bayesian Foundations" \
  --body "## Context
Translates budget plan information into mathematical priors.

## Tasks
- [ ] Implement \`src/priors.py\`
- [ ] Create tests (DO NOT RUN)

## Definition of Done
- [ ] Prior construction from budget plans implemented
- [ ] Tests created

## References
- Theory from Issue #4"

# Phase 2
gh issue create --repo "$REPO" --title "[Phase 2] Derive all four conjugate pairs" \
  --label "phase:2,type:theory,priority:critical" --milestone "Phase 2 — Conjugate Families" \
  --body "## Context
The mathematical core: every FYF update is a conjugate posterior.

## Tasks
- [ ] Normal-Normal (known σ²): precision-weighted mean
- [ ] Normal-Inverse Gamma (unknown μ, σ²): four hyperparameters
- [ ] Gamma-Poisson: posterior and pseudo-observations
- [ ] Beta-Binomial: posterior
- [ ] Each: worked example with FYF context

## Definition of Done
- [ ] All four pairs derived step-by-step
- [ ] Precision-weighted mean is central insight

## References
- Gelman et al., BDA Ch. 2–3"

gh issue create --repo "$REPO" --title "[Phase 2] Implement conjugate updating module" \
  --label "phase:2,type:code,priority:high" --milestone "Phase 2 — Conjugate Families" \
  --body "## Context
Wraps closed-form posterior computations for all four families.

## Tasks
- [ ] Implement \`src/conjugate.py\` with all four updaters
- [ ] Create tests (DO NOT RUN)

## Definition of Done
- [ ] All updaters implemented, tests created

## References
- Theory from Issue #6"

# Phase 3
gh issue create --repo "$REPO" --title "[Phase 3] Prove sequential=batch and derive shrinkage" \
  --label "phase:3,type:theory,priority:critical" --milestone "Phase 3 — Sequential Updating" \
  --body "## Context
Sequential updating = FYF cycle. Shrinkage = the math of forecast revision.

## Tasks
- [ ] Prove sequential = batch for Normal-Normal and exp. families
- [ ] Derive shrinkage weight w₀ = τ₀/(τ₀ + nτ)
- [ ] Prove w₀ → 0 and σ_n → 0
- [ ] Analyse prior sensitivity
- [ ] Define prior-data conflict

## Definition of Done
- [ ] All proofs in \`notes/phase3-sequential-updating.md\`

## References
- Gelman et al., BDA Ch. 2.6"

gh issue create --repo "$REPO" --title "[Phase 3] Implement sequential updating engine" \
  --label "phase:3,type:code,priority:high" --milestone "Phase 3 — Sequential Updating" \
  --body "## Context
Processes monthly actuals sequentially, tracking posterior history.

## Tasks
- [ ] Implement \`src/updating.py\`
- [ ] Create tests (DO NOT RUN)

## Definition of Done
- [ ] Sequential engine works for all conjugate pairs
- [ ] History tracking complete

## References
- Theory from Issue #8"

# Phase 4
gh issue create --repo "$REPO" --title "[Phase 4] Derive posterior predictive and Bayes factors" \
  --label "phase:4,type:theory,priority:critical" --milestone "Phase 4 — Predictive Inference" \
  --body "## Context
The posterior predictive answers the business question: what will future months cost?

## Tasks
- [ ] Derive predictive for Normal-Normal, Gamma-Poisson, Beta-Binomial
- [ ] Derive year-end total prediction with P(overbudget)
- [ ] Connect to Monte Carlo (Article 1)
- [ ] Define Bayes factors

## Definition of Done
- [ ] Predictive derived for all pairs
- [ ] Year-end prediction applied to FYF

## References
- Gelman et al., BDA Ch. 2.5, 7"

gh issue create --repo "$REPO" --title "[Phase 4] Implement predictive module" \
  --label "phase:4,type:code,priority:high" --milestone "Phase 4 — Predictive Inference" \
  --body "## Context
Computes posterior predictive distributions and year-end forecasts.

## Tasks
- [ ] Implement \`src/predictive.py\`
- [ ] Create tests (DO NOT RUN)

## Definition of Done
- [ ] Year-end forecast matches analytical formula

## References
- Theory from Issue #10"

# Phase 5
gh issue create --repo "$REPO" --title "[Phase 5] Build FYF model and simulate 5 scenarios" \
  --label "phase:5,type:code,type:experiment,priority:critical" --milestone "Phase 5 — Applied FYF Model" \
  --body "## Context
Complete annual FYF cycle as sequential Bayesian updating.

## Tasks
- [ ] Implement \`src/fyf_model.py\` and \`src/visualization.py\`
- [ ] Simulate 5 scenarios (on-target, optimistic, shock, seasonal, sensitivity)
- [ ] Create figures
- [ ] Tests (DO NOT RUN)

## Definition of Done
- [ ] All scenarios analysed with figures

## References
- All theory Phases 1–4"

gh issue create --repo "$REPO" --title "[Phase 5] Implement diagnostics and model checking" \
  --label "phase:5,type:code,priority:high" --milestone "Phase 5 — Applied FYF Model" \
  --body "## Context
Checks whether model predictions are consistent with actuals.

## Tasks
- [ ] Implement \`src/diagnostics.py\`
- [ ] Calibration plot and surprise detection
- [ ] Tests (DO NOT RUN)

## Definition of Done
- [ ] Diagnostics implemented and validated

## References
- Gelman et al., BDA Ch. 6"

# Phase 6
gh issue create --repo "$REPO" --title "[Phase 6] Run all experiments and create publication figures" \
  --label "phase:6,type:experiment,priority:critical" --milestone "Phase 6 — Experiments & Visualizations" \
  --body "## Context
All 8 experiments with figures and animated GIF.

## Tasks
- [ ] Experiments A–H
- [ ] 300 DPI, fixed seeds
- [ ] Animated posterior evolution GIF

## Definition of Done
- [ ] All figures in \`figures/\`, GIF < 5 MB

## References
- All theory Phases 1–5"

# Phase 7
gh issue create --repo "$REPO" --title "[Phase 7] Write article sections 1–5 (English)" \
  --label "phase:7,type:writing,priority:high" --milestone "Phase 7 — Article Writing" \
  --body "## Context
First half: Bayes' theorem through predictive inference.

## Tasks
- [ ] Sections 1–5 in \`article/bayesian-fyf.md\`
- [ ] Self-contained derivations, consistent notation

## Definition of Done
- [ ] Sections 1–5 complete in English

## References
- Theory notes Phases 1–4"

gh issue create --repo "$REPO" --title "[Phase 7] Write article sections 6–11 (English)" \
  --label "phase:7,type:writing,priority:high" --milestone "Phase 7 — Article Writing" \
  --body "## Context
Second half: FYF model, experiments, diagnostics, conclusion.

## Tasks
- [ ] Sections 6–11 in \`article/bayesian-fyf.md\`
- [ ] Connection to Articles 1–3

## Definition of Done
- [ ] Full article complete in English

## References
- Phase 5 model, Phase 6 figures"

gh issue create --repo "$REPO" --title "[Phase 7] Translate article to Portuguese" \
  --label "phase:7,type:writing,priority:medium" --milestone "Phase 7 — Article Writing" \
  --body "## Context
Portuguese version for review. Translation, not rewrite.

## Tasks
- [ ] Translate to \`article/bayesian-fyf-ptbr.md\`

## Definition of Done
- [ ] Full translation, math unchanged

## References
- English article"

# Phase 8
gh issue create --repo "$REPO" --title "[Phase 8] Mathematical validation and code reproducibility" \
  --label "phase:8,type:review,priority:critical" --milestone "Phase 8 — Review & Publish" \
  --body "## Context
Final quality gate.

## Tasks
- [ ] Review all derivations
- [ ] Author runs: \`pip install -e .\` → scripts → tests → ruff

## Definition of Done
- [ ] Zero errors, tests pass, ruff clean

## References
- Full article"

gh issue create --repo "$REPO" --title "[Phase 8] Publish to GitHub Pages and Medium" \
  --label "phase:8,type:writing,type:infrastructure,priority:high" --milestone "Phase 8 — Review & Publish" \
  --body "## Context
Publication and distribution.

## Tasks
- [ ] Copy to github.io, run pipeline, verify
- [ ] Medium cross-post, LinkedIn post
- [ ] Update README

## Definition of Done
- [ ] Article live, Medium published, LinkedIn drafted

## References
- Author's github.io pipeline"

echo "All issues created."
echo "Verify milestone IDs: gh api repos/$REPO/milestones --jq '.[] | {number, title}'"
