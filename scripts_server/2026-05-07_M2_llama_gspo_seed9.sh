#!/usr/bin/env bash
# 2026-05-07 / M2 — GSPO × Llama-3.2-3B-Instruct × seed=9 × ep1
# Llama baseline 多 seed 补全（已有 GSPO Llama seed=42 from rerun M12，补 seed=9）。
set -e
cd "$HOME/sentencepo_v1-5"
SEED=9 EPOCHS=1 \
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  MODEL_NAME=llama3.2-3b \
  bash test_GSPO-metrics2.sh
