#!/bin/bash
set -euo pipefail
REPO="${1:?Usage: bash labels.sh owner/repo}"

echo "Creating labels for $REPO..."

gh label create "phase:0" --color "0E8A16" --description "Phase 0 — Foundation" --repo "$REPO" --force
gh label create "phase:1" --color "1D76DB" --description "Phase 1 — Bayesian Foundations" --repo "$REPO" --force
gh label create "phase:2" --color "5319E7" --description "Phase 2 — Conjugate Families" --repo "$REPO" --force
gh label create "phase:3" --color "D93F0B" --description "Phase 3 — Sequential Updating" --repo "$REPO" --force
gh label create "phase:4" --color "FBCA04" --description "Phase 4 — Predictive Inference" --repo "$REPO" --force
gh label create "phase:5" --color "B60205" --description "Phase 5 — Applied FYF Model" --repo "$REPO" --force
gh label create "phase:6" --color "006B75" --description "Phase 6 — Experiments & Visualizations" --repo "$REPO" --force
gh label create "phase:7" --color "1D76DB" --description "Phase 7 — Article Writing" --repo "$REPO" --force
gh label create "phase:8" --color "0E8A16" --description "Phase 8 — Review & Publish" --repo "$REPO" --force

gh label create "type:theory" --color "C5DEF5" --description "Mathematical derivation or proof" --repo "$REPO" --force
gh label create "type:code" --color "BFD4F2" --description "Implementation task" --repo "$REPO" --force
gh label create "type:experiment" --color "D4C5F9" --description "Experimental validation" --repo "$REPO" --force
gh label create "type:writing" --color "FEF2C0" --description "Article writing task" --repo "$REPO" --force
gh label create "type:documentation" --color "0075CA" --description "Planning or project docs" --repo "$REPO" --force
gh label create "type:infrastructure" --color "E4E669" --description "Repo setup, CI, tooling" --repo "$REPO" --force
gh label create "type:review" --color "F9D0C4" --description "Review or validation" --repo "$REPO" --force
gh label create "type:bug" --color "D73A4A" --description "Something is broken" --repo "$REPO" --force
gh label create "type:content" --color "BFDADC" --description "LinkedIn, Medium, social" --repo "$REPO" --force

gh label create "priority:critical" --color "B60205" --description "Must be done, blocks other work" --repo "$REPO" --force
gh label create "priority:high" --color "D93F0B" --description "Important, do soon" --repo "$REPO" --force
gh label create "priority:medium" --color "FBCA04" --description "Can wait but should be done" --repo "$REPO" --force
gh label create "priority:low" --color "0E8A16" --description "Nice to have" --repo "$REPO" --force

echo "All labels created."
