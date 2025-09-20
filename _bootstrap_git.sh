#!/usr/bin/env bash
set -euo pipefail

cd /Users/georgekurchey/CL_Model

# Init repo (ok if already done)
git init

# Ensure branch name is main (ignore if it already is)
git branch -M main || true

# First commit (ignore if nothing to commit)
git add .
git commit -m "chore: initial commit" || true

# Point to your GitHub repo (replace URL if needed)
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/georgekurchey/CL_Model.git

# Push
git push -u origin main
