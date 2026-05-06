#!/usr/bin/env bash
# 2026-05-07 / M1 — v1-5 winner_config × Llama-3.2-3B-Instruct × seed=37 × ep1
#
# Llama 多 seed 补全：已有 winner_llama seed=42/9，本次补 seed=37 凑齐 3 seed。
# winner_config = smaller_alpha (SLPA 0.02/SCR 0.01 + L=-18 + decay=none).
set -e
cd "$HOME/sentencepo_v1-5"

source scripts_server/2026-05-05_phase1_winner_config.sh

EXP_TAG="2026-05-07_llama_v15_winner" \
  MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct \
  MODEL_NAME=llama3.2-3b \
  MAX_RESP_LEN=4096 \
  EPOCHS=1 \
  SEED=37 \
  bash scripts_server/2026-05-03_v1-5-hidden_run.sh
