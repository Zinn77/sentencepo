#!/usr/bin/env bash
# 2026-05-07 / M3 — GSPO × Llama-3.2-3B-Instruct × seed=37 × ep1
set -e
cd "$HOME/sentencepo_v1-5"
SEED=37 EPOCHS=1 \
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  MODEL_NAME=llama3.2-3b \
  bash test_GSPO-metrics2.sh
