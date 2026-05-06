#!/usr/bin/env bash
# 2026-05-07 / M4 — GRPO × Llama-3.2-3B-Instruct × seed=9 × ep1
# 注意：GRPO Llama 在 ep1 内会 length collapse（peak 0.20 → end 0.09 on seed=42）。
# 本 run 验证 collapse 是否 seed-stable，作为论文 cross-model robustness 的反例。
set -e
cd "$HOME/sentencepo_v1-5"
SEED=9 EPOCHS=1 \
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  MODEL_NAME=llama3.2-3b \
  bash test_GRPO-metrics2.sh
