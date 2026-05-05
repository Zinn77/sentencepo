#!/usr/bin/env bash
# Phase 1 / M5 — GSPO baseline × Qwen3 × seed=9 × ep3
# multi-seed 第三个 seed（已有 42 + 37）。
set -e
cd "$HOME/sentencepo_v1-5"

SEED=9 EPOCHS=3 bash test_GSPO-metrics2.sh
