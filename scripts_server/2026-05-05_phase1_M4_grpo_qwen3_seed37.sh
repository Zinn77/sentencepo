#!/usr/bin/env bash
# Phase 1 / M4 — GRPO baseline × Qwen3 × seed=37 × ep3
set -e
cd "$HOME/sentencepo_v1-5"

SEED=37 EPOCHS=3 bash test_GRPO-metrics2.sh
