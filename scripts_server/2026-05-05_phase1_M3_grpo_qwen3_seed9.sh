#!/usr/bin/env bash
# Phase 1 / M3 — GRPO baseline × Qwen3 × seed=9 × ep3
# 使用现有 baseline 脚本 test_GRPO-metrics2.sh，仅传 SEED + EPOCHS。
set -e
cd "$HOME/sentencepo_v1-5"

SEED=9 EPOCHS=3 bash test_GRPO-metrics2.sh
