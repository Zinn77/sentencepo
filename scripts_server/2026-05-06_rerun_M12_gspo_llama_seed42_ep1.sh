#!/usr/bin/env bash
# Rerun / M12 — GSPO Llama × seed=42 × ep1 native
#
# 输出 dir：gspo_math_llama3.2-3b_ep1_rand42
set -e
cd "$HOME/sentencepo_v1-5"
SEED=42 EPOCHS=1 \
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  MODEL_NAME=llama3.2-3b \
  bash test_GSPO-metrics2.sh
