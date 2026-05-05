#!/usr/bin/env bash
# Phase 1 / M5 — GSPO baseline × Qwen3 × seed=9 × ep1
# multi-seed 第三个 seed（已有 seed=42 + seed=37 ep3，本次补 ep1 第三 seed）。
set -e
cd "$HOME/sentencepo_v1-5"

SEED=9 EPOCHS=1 bash test_GSPO-metrics2.sh
