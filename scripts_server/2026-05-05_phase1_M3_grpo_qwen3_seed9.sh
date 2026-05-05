#!/usr/bin/env bash
# Phase 1 / M3 — GRPO baseline × Qwen3 × seed=9 × ep1
# multi-seed 第二个 seed（已有 seed=42 ep3，可截 ep1 用）。
set -e
cd "$HOME/sentencepo_v1-5"

SEED=9 EPOCHS=1 bash test_GRPO-metrics2.sh
