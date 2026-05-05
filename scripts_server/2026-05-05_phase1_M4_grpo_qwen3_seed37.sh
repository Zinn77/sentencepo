#!/usr/bin/env bash
# Phase 1 / M4 — GRPO baseline × Qwen3 × seed=37 × ep1
# multi-seed 第三个 seed（已有 seed=42 ep3，本次补 ep1 第三 seed）。
set -e
cd "$HOME/sentencepo_v1-5"

SEED=37 EPOCHS=1 bash test_GRPO-metrics2.sh
